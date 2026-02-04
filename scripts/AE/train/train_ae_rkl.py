"""
Conditional AutoEncoder for Knowledge Distillation (with LayerNorm)

이 스크립트는 Teacher 모델의 Hidden States를 압축하는 Conditional AutoEncoder를 학습합니다.

주요 개념:
----------
1. **Conditional AutoEncoder 구조**:
   - Encoder: [hidden_state, teacher_embed(y)] -> latent_code (차원 축소)
   - Decoder: [latent_code, teacher_embed(y)] -> reconstructed_hidden_state
   
   여기서 y는 다음 토큰 예측 (teacher의 argmax prediction)입니다.

2. **데이터 처리**:
   - .bin 파일 (MMapIndexedDataset)에서 토큰 시퀀스 로드
   - Teacher 모델로 각 시퀀스의 hidden states 추출
   - 각 위치(pos)에서:
     * x_t: 현재 토큰 (position pos의 토큰)
     * y_t: 다음 토큰 예측 (teacher가 position pos에서 예측한 토큰)
     * h_t: 현재 위치의 hidden state

3. **Loss 함수**:
   - MSE Loss: hidden state 재구성 오차
   - Cosine Loss: 방향성 보존 (선택적)
   - Logit Reverse KL Loss: KL(recon || teacher)
   - Logit MSE Loss: Logit 값 직접 매칭 (선택적)

4. **사용 목적**:
   - Teacher의 hidden state를 작은 latent space로 압축
   - Knowledge Distillation에서 사용할 수 있는 압축된 표현 학습
   - 다음 토큰 정보를 condition으로 사용하여 더 나은 압축 성능 달성

사용 예시:
----------
python train_ae.py \
    --data_path /home/jiwonyoon/data1/data/pile_dataset/data_0 \
    --teacher_path /path/to/teacher \
    --latent_dim 4 \
    --train_samples 10000 \
    --batch_size 256 \
    --epochs 30
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import hashlib
from pathlib import Path
import sys
import numpy as np
import time

# Add parent directory to path for data_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_utils.indexed_dataset import MMapIndexedDataset


# ========================================================================================
# AutoEncoder Model
# ========================================================================================
#
# 이 코드는 Conditional AutoEncoder를 사용하여 Teacher 모델의 Hidden States를 압축합니다.
#
# 구조:
#   1. Encoder: [hidden_state, teacher_embed(y)] -> latent_code (차원 축소)
#   2. Decoder: [latent_code, teacher_embed(y)] -> reconstructed_hidden_state
#
# 목적:
#   - Teacher의 hidden state를 작은 latent space로 압축
#   - 다음 토큰 예측(y)을 condition으로 사용하여 더 나은 압축 성능 달성
#   - Knowledge Distillation에서 사용할 수 있는 압축된 표현 학습
#
# Loss:
#   - MSE Loss: hidden state 재구성 오차
#   - Cosine Loss: 방향성 보존
#   - Logit Reverse KL Loss: KL(recon || teacher)
#   - Logit MSE Loss: Logit 값 직접 매칭
#
# ========================================================================================

class ConditionalAutoEncoder(nn.Module):
    """AutoEncoder with Y condition (using teacher embedding)"""
    def __init__(self, input_dim=1600, latent_dim=8, teacher_embed=None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Y condition embedding: use teacher's word embedding (frozen)
        self.y_embed = teacher_embed  # teacher.transformer.wte
        for p in self.y_embed.parameters():
            p.requires_grad = False  # frozen
        
        # Encoder input: hidden + teacher_embed(y) = input_dim * 2
        enc_input_dim = input_dim * 2  # 3200 for GPT2-XL
        
        # Encoder: enc_input_dim -> latent_dim
        # 먼저 input_dim으로 축소한 후 latent_dim으로
        enc_dims = self._get_dims(enc_input_dim, input_dim)
        if enc_dims[-1] != input_dim:
            enc_dims.append(input_dim)
        # input_dim에서 latent_dim으로
        if latent_dim < input_dim:
            latent_dims = self._get_dims(input_dim, latent_dim)
            enc_dims.extend(latent_dims[1:])  # input_dim 제외하고 추가
        elif latent_dim > input_dim:
            enc_dims.append(latent_dim)
        # latent_dim == input_dim이면 그대로
        
        # Encoder layers
        encoder_layers = []
        for i in range(len(enc_dims) - 1):
            encoder_layers.append(nn.Linear(enc_dims[i], enc_dims[i+1]))
            if i < len(enc_dims) - 2:
                encoder_layers.append(nn.LayerNorm(enc_dims[i+1]))
                encoder_layers.append(nn.GELU())
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder input: latent + teacher_embed(y) = latent_dim + input_dim
        dec_input_dim = latent_dim + input_dim  # 16 + 1600 for GPT2-XL
        
        # Decoder: dec_input_dim -> input_dim
        decoder_dims = self._get_dims(dec_input_dim, input_dim)
        
        # Decoder layers
        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i+1]))
            if i < len(decoder_dims) - 2:
                decoder_layers.append(nn.LayerNorm(decoder_dims[i+1]))
                decoder_layers.append(nn.GELU())
        decoder_layers.append(nn.LayerNorm(input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def _get_dims(self, input_dim, latent_dim):
        """input_dim에서 latent_dim까지 점진적으로 줄어드는 차원 리스트"""
        if latent_dim >= input_dim:
            return [input_dim, latent_dim]
        
        dims = [input_dim]
        current = input_dim
        
        # 절반씩 줄여가면서 latent_dim보다 클 때만 추가
        while current // 2 > latent_dim:
            current = current // 2
            dims.append(current)
        
        dims.append(latent_dim)
        return dims
    
    def forward(self, hidden, y_token):
        # Y condition embedding (match dtype with hidden)
        cond = self.y_embed(y_token).to(hidden.dtype)
        
        # Encode
        enc_input = torch.cat([hidden, cond], dim=-1)
        z = self.encoder(enc_input)
        
        # Decode
        dec_input = torch.cat([z, cond], dim=-1)
        recon = self.decoder(dec_input)
        
        return recon, z


# ========================================================================================
# Helper Functions
# ========================================================================================

def get_lm_logits_from_hidden(model, hidden_states):
    """Get logits from hidden states"""
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        ln_f = model.transformer.ln_f
        target_dtype = ln_f.weight.dtype
        hidden_states = hidden_states.to(target_dtype)
        hidden_norm = ln_f(hidden_states)
    else:
        hidden_norm = hidden_states
    
    lm_head = model.lm_head
    head_dtype = lm_head.weight.dtype
    hidden_norm = hidden_norm.to(head_dtype)
    logits = lm_head(hidden_norm)
    return logits


# ========================================================================================
# Dataset
# ========================================================================================

class TeacherPredictionDataset(Dataset):
    """Dataset with teacher prediction as condition"""
    def __init__(
        self,
        data_path,
        tokenizer,
        teacher,
        start_idx=0,
        max_samples=None,
        max_length=512,
        device="cuda",
        cache_dir=None,
        pre_extracted_path=None
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.teacher = teacher
        self.max_length = max_length
        self.device = device
        self.data_path = data_path
        self.start_idx = start_idx
        self.max_samples = max_samples

        # Try to load from pre-extracted file first
        if pre_extracted_path is not None and os.path.exists(pre_extracted_path):
            print(f"Loading hidden states from pre-extracted file: {pre_extracted_path}")
            cached_data = torch.load(pre_extracted_path, map_location='cpu')
            self.hiddens = cached_data['hiddens']
            self.x_tokens = cached_data['x_tokens']
            self.y_tokens = cached_data['y_tokens']
            print(f"Loaded {self.hiddens.shape[0]:,} tokens from pre-extracted file")
            return

        # Generate cache file path
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(data_path), ".cache_hidden_states")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create cache key from data path, start_idx, max_samples, max_length
        cache_key = f"{os.path.basename(data_path)}_{start_idx}_{max_samples}_{max_length}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"hidden_states_{cache_hash}.pt")
        
        # Try to load from cache
        if os.path.exists(cache_file):
            print(f"Loading hidden states from cache: {cache_file}")
            cached_data = torch.load(cache_file, map_location='cpu')
            self.hiddens = cached_data['hiddens']
            self.x_tokens = cached_data['x_tokens']
            self.y_tokens = cached_data['y_tokens']
            print(f"Loaded {self.hiddens.shape[0]:,} tokens from cache")
            return

        if max_samples == 0:
            print(f"Skipping data loading (max_samples=0)")
            sequences = []
        else:
            # Load from .bin file (MMapIndexedDataset)
            print(f"Loading data from {data_path}...")
            dataset = MMapIndexedDataset(data_path, skip_warmup=True)
            total_sequences = len(dataset)
            
            # Determine range
            end_idx = start_idx + max_samples if max_samples is not None else total_sequences
            end_idx = min(end_idx, total_sequences)
            
            print(f"Loading sequences {start_idx} ~ {end_idx-1} (total: {total_sequences})...")
            sequences = []
            for i in range(start_idx, end_idx):
                token_ids = dataset[i].astype(np.int64)
                if len(token_ids) > 0:
                    sequences.append(token_ids)
            print(f"Loaded {len(sequences)} sequences")

        print("Extracting features with teacher predictions...")
        start_time = time.time()
        self.teacher.eval()
        
        hidden_list = []
        x_token_list = []
        y_token_list = []
        
        # Batch processing parameters (cache_teacher_logits_mp.py 방식)
        batch_size = 32  # Process multiple sequences at once
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        with torch.no_grad():
            batch_input_ids = []
            batch_seq_indices = []  # 원본 시퀀스 인덱스 추적
            
            for seq_idx, seq_tokens in enumerate(tqdm(sequences, desc="Extracting hidden states")):
                # Convert numpy array and truncate
                data = seq_tokens.astype(np.int64)
                data = data[:max_length + 1]  # max_length + 1 (마지막 토큰 제외용)
                
                if len(data) <= 1:
                    continue
                
                # 마지막 토큰 제외 (position pos는 pos+1을 예측)
                batch_input_ids.append(data[:-1])
                batch_seq_indices.append(seq_idx)
                
                # 배치 처리 조건: 배치 사이즈 도달 또는 마지막 시퀀스
                is_last_seq = (seq_idx == len(sequences) - 1)
                if len(batch_input_ids) >= batch_size or is_last_seq:
                    actual_batch_size = len(batch_input_ids)
                    
                    # Pad batch (numpy로 먼저 패딩, cache_teacher_logits_mp.py 방식)
                    max_len = max(len(seq) for seq in batch_input_ids)
                    max_len = min(max_len, max_length)  # MAX_LENGTH 제한 적용
                    
                    padded_batch = np.full((actual_batch_size, max_len), pad_id, dtype=np.int64)
                    attention_mask = np.zeros((actual_batch_size, max_len), dtype=np.int64)
                    
                    # 각 시퀀스의 실제 유효 길이 저장
                    actual_seq_lens = []
                    for i, seq in enumerate(batch_input_ids):
                        seq_len = min(len(seq), max_len)
                        actual_seq_lens.append(seq_len)
                        padded_batch[i, :seq_len] = seq[:seq_len]
                        attention_mask[i, :seq_len] = 1
                    
                    # Convert to torch tensors
                    input_ids = torch.tensor(padded_batch, device=device, dtype=torch.long)
                    attn_mask = torch.tensor(attention_mask, device=device, dtype=torch.long)
                    
                    # Forward pass
                    outputs = self.teacher(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True, use_cache=False)
                    last_hidden = outputs.hidden_states[-1]  # [B, max_len, hidden_dim]
                    teacher_logits = outputs.logits  # [B, max_len, vocab_size]
                    
                    # Process each sequence in batch
                    for i in range(actual_batch_size):
                        actual_len = actual_seq_lens[i]
                        if actual_len <= 1:
                            continue
                        
                        # Extract valid parts (exclude padding)
                        seq_hidden = last_hidden[i, :actual_len].cpu()  # [actual_len, hidden_dim]
                        seq_logits = teacher_logits[i, :actual_len].cpu()  # [actual_len, vocab_size]
                        seq_input_ids = input_ids[i, :actual_len].cpu().long()  # [actual_len]
                        
                        # Extract data for each position (exclude last)
                        # Position pos predicts token at pos+1
                        for pos in range(actual_len - 1):
                            h_t = seq_hidden[pos]  # [hidden_dim]
                            x_t = seq_input_ids[pos].item()  # Current token at position pos
                            
                            # Teacher prediction for y (position pos predicts token at pos+1)
                            y_t = torch.argmax(seq_logits[pos], dim=-1).item()
                            
                            hidden_list.append(h_t)
                            x_token_list.append(x_t)
                            y_token_list.append(y_t)
                    
                    # Clear batch
                    batch_input_ids = []
                    batch_seq_indices = []
                    
                    # Memory cleanup
                    del input_ids, attn_mask, outputs, last_hidden, teacher_logits
                    torch.cuda.empty_cache()
        
        if len(hidden_list) == 0:
            raise ValueError("No valid tokens found.")

        self.hiddens = torch.stack(hidden_list, dim=0)
        self.x_tokens = torch.stack(x_token_list, dim=0)
        self.y_tokens = torch.stack(y_token_list, dim=0)

        elapsed_time = time.time() - start_time
        print(f"Total tokens: {self.hiddens.shape[0]}")
        print(f"Extracting hidden states took {elapsed_time/60:.2f} minutes ({elapsed_time:.2f} seconds)")
        
        # Save to cache
        print(f"Saving hidden states to cache: {cache_file}")
        torch.save({
            'hiddens': self.hiddens,
            'x_tokens': self.x_tokens,
            'y_tokens': self.y_tokens
        }, cache_file)
        print(f"Cache saved successfully")

    def __len__(self):
        return self.hiddens.size(0)
    
    def __getitem__(self, idx):
        return {
            'hidden': self.hiddens[idx],
            'x_token': self.x_tokens[idx],
            'y_token': self.y_tokens[idx]
        }


def collate_fn(batch):
    """Custom collate function"""
    return {
        'hidden': torch.stack([item['hidden'] for item in batch]),
        'x_token': torch.stack([item['x_token'] for item in batch]),
        'y_token': torch.stack([item['y_token'] for item in batch])
    }


# ========================================================================================
# Training
# ========================================================================================

def train_autoencoder(
    ae_model,
    teacher_model,
    train_loader,
    val_loader,
    model_name,
    output_dir=".",
    epochs=30,
    lr=5e-4,
    device="cuda",
    patience=10,
    alpha_mse=2.0,
    alpha_cosine=0.0,
    alpha_logit=1.0,
    alpha_logit_mse=0.0,
):
    """Train autoencoder"""
    ae_model = ae_model.to(device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    
    optimizer = torch.optim.AdamW(ae_model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    max_grad_norm = 1.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Loss weights (shared between train and validation)
    # These are passed as function arguments
    
    for epoch in range(epochs):
        # Train
        ae_model.train()
        train_loss = 0.0
        train_batches = 0
         
        for batch in tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1}/{epochs}"):
            h = batch['hidden'].to(device).float()  # Convert to float32 for AE training
            y_token = batch['y_token'].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            recon, z = ae_model(h, y_token=y_token)
            
            # MSE loss (hidden reconstruction)
            mse_loss = F.mse_loss(recon, h)
            
            # Cosine similarity loss (1 - cosine_similarity)
            cosine_sim = F.cosine_similarity(recon, h, dim=-1)  # [B]
            cosine_loss = (1 - cosine_sim).mean()  # 1 - similarity를 최소화
            
            # Logit loss (reverse KL: KL(recon || teacher))
            # Convert to appropriate dtype for teacher model
            teacher_dtype = next(teacher_model.parameters()).dtype
            h_for_logits = h.to(teacher_dtype)
            recon_for_logits = recon.to(teacher_dtype)
            teacher_logits = get_lm_logits_from_hidden(teacher_model, h_for_logits)
            recon_logits = get_lm_logits_from_hidden(teacher_model, recon_for_logits)
            temperature = 1.0
            # Reverse KL: input=log(teacher), target=recon => KL(recon || teacher)
            logit_loss = F.kl_div(
                F.log_softmax(teacher_logits / temperature, dim=-1),
                F.softmax(recon_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature ** 2)
            
            # Logit MSE loss
            logit_mse_loss = F.mse_loss(recon_logits, teacher_logits)
            
            # Combined loss
            loss = (alpha_mse * mse_loss + alpha_cosine * cosine_loss + 
                    alpha_logit * logit_loss + alpha_logit_mse * logit_mse_loss)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae_model.parameters(), max_grad_norm)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else float('inf')
        
        # Validation
        ae_model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                h = batch['hidden'].to(device).float()  # Convert to float32 for AE
                y_token = batch['y_token'].to(device)
                
                recon, z = ae_model(h, y_token=y_token)
                
                # MSE loss
                mse_loss = F.mse_loss(recon, h)
                
                # Cosine similarity loss
                cosine_sim = F.cosine_similarity(recon, h, dim=-1)
                cosine_loss = (1 - cosine_sim).mean()
                
                # Logit loss (reverse KL: KL(recon || teacher))
                teacher_dtype = next(teacher_model.parameters()).dtype
                h_for_logits = h.to(teacher_dtype)
                recon_for_logits = recon.to(teacher_dtype)
                teacher_logits = get_lm_logits_from_hidden(teacher_model, h_for_logits)
                recon_logits = get_lm_logits_from_hidden(teacher_model, recon_for_logits)
                temperature = 1.0
                logit_loss = F.kl_div(
                    F.log_softmax(teacher_logits / temperature, dim=-1),
                    F.softmax(recon_logits / temperature, dim=-1),
                    reduction="batchmean",
                ) * (temperature ** 2)
                
                # Logit MSE loss
                logit_mse_loss = F.mse_loss(recon_logits, teacher_logits)
                
                # Combined loss (same as training)
                loss = (alpha_mse * mse_loss + alpha_cosine * cosine_loss +
                        alpha_logit * logit_loss + alpha_logit_mse * logit_mse_loss)
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()
                    val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if old_lr != new_lr:
            print(f"  → LR reduced: {old_lr:.6f} → {new_lr:.6f}")
        
        print(
            f"[{model_name}] Epoch {epoch+1}/{epochs} | "
            f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}"
        )
        
        # Save best
        if avg_val_loss < best_val_loss and not torch.isnan(torch.tensor(avg_val_loss)):
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            os.makedirs(output_dir, exist_ok=True)
            save_name = os.path.join(output_dir, f"best_ae_{model_name}.pt")
            # Exclude y_embed (teacher embedding) from saved state_dict
            state_dict = {k: v for k, v in ae_model.state_dict().items() if 'y_embed' not in k}
            torch.save(state_dict, save_name)
            print(f"  → Saved best (val_loss: {best_val_loss:.6f}) to {save_name}")
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return ae_model


# ========================================================================================
# Main
# ========================================================================================

def main():
    parser = argparse.ArgumentParser(description="Train AutoEncoder for KD")
    parser.add_argument("--data_path", type=str, 
                        default="/home/jiwonyoon/data1/data/pile_dataset/data_0",
                        help="Path to training data (.bin file without extension, e.g., data_0). Default: data_0.bin")
    parser.add_argument("--val_data_path", type=str, default=None,
                        help="Path to validation data (.bin file). If not provided, split from training data.")
    parser.add_argument("--teacher_path", type=str, required=True,
                        help="Path to teacher model")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to tokenizer (default: same as teacher_path)")
    parser.add_argument("--latent_dim", type=int, default=4,
                        help="Latent dimension for AE (default: 4)")
    parser.add_argument("--train_samples", type=int, default=None,
                        help="Number of training samples (default: None, use all)")
    parser.add_argument("--val_samples", type=int, default=None,
                        help="Number of validation samples (default: None, use all)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size (default: 256)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs (default: 30)")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate (default: 5e-4)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (default: 10)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (default: cuda:0)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max sequence length (default: 512)")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save the model (default: current directory)")
    parser.add_argument("--alpha_mse", type=float, default=2.0,
                        help="Weight for MSE loss (default: 2.0)")
    parser.add_argument("--alpha_cosine", type=float, default=0.0,
                        help="Weight for cosine similarity loss (default: 0.0)")
    parser.add_argument("--alpha_logit", type=float, default=1.0,
                        help="Weight for logit reverse KL loss KL(recon||teacher) (default: 1.0)")
    parser.add_argument("--alpha_logit_mse", type=float, default=0.0,
                        help="Weight for logit MSE loss (default: 0.0)")
    parser.add_argument("--pre_extracted_path", type=str, default=None,
                        help="Path to pre-extracted hidden states file (.pt). If provided, will use this instead of extracting.")
    
    args = parser.parse_args()
    
    # Use tokenizer_path if provided, otherwise use teacher_path
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.teacher_path
    
    print("Loading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_path,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    teacher_model.eval()
    
    # Get hidden dimension (support both GPT-style and Qwen-style configs)
    if hasattr(teacher_model.config, 'n_embd'):
        hidden_dim = teacher_model.config.n_embd
    elif hasattr(teacher_model.config, 'hidden_size'):
        hidden_dim = teacher_model.config.hidden_size
    else:
        raise ValueError(f"Unknown config type: {type(teacher_model.config)}. Expected n_embd or hidden_size attribute.")
    
    # Get teacher's word embedding for condition (support both GPT-style and Qwen-style)
    if hasattr(teacher_model, 'transformer') and hasattr(teacher_model.transformer, 'wte'):
        # GPT-style models (GPT2, etc.)
        teacher_embed = teacher_model.transformer.wte
    elif hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'embed_tokens'):
        # Qwen2-style models
        teacher_embed = teacher_model.model.embed_tokens
    else:
        raise ValueError(f"Unknown model structure. Cannot find word embedding. Model type: {type(teacher_model)}")
    print(f"Hidden dim: {hidden_dim}, Teacher embed dim: {teacher_embed.weight.shape}")
    
    # Prepare datasets
    print("\nPreparing datasets...")
    
    # Training dataset
    train_dataset = TeacherPredictionDataset(
        args.data_path, tokenizer, teacher_model,
        start_idx=0, max_samples=args.train_samples,
        max_length=args.max_length, device=args.device,
        pre_extracted_path=args.pre_extracted_path
    )
    
    # Validation dataset
    if args.val_data_path is not None:
        # Use separate validation file
        print(f"\nUsing separate validation file: {args.val_data_path}")
        val_dataset = TeacherPredictionDataset(
            args.val_data_path, tokenizer, teacher_model,
            start_idx=0, max_samples=args.val_samples,
            max_length=args.max_length, device=args.device,
            pre_extracted_path=None  # Validation uses separate file, no pre-extracted path
        )
    elif args.val_samples is not None and args.val_samples > 0:
        # Split from training data (use samples after train_samples)
        val_start_idx = args.train_samples if args.train_samples else 0
        val_dataset = TeacherPredictionDataset(
            args.data_path, tokenizer, teacher_model,
            start_idx=val_start_idx, max_samples=args.val_samples,
            max_length=args.max_length, device=args.device,
            pre_extracted_path=None  # Validation split, no pre-extracted path
        )
    else:
        # No validation
        print("\nNo validation dataset specified.")
        val_dataset = TeacherPredictionDataset(
            args.data_path, tokenizer, teacher_model,
            start_idx=0, max_samples=0,
            max_length=args.max_length, device=args.device,
            pre_extracted_path=None
        )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, collate_fn=collate_fn
    )
    
    model_name = f"y_ld{args.latent_dim}"
    
    ae_model = ConditionalAutoEncoder(
        input_dim=hidden_dim,
        latent_dim=args.latent_dim,
        teacher_embed=teacher_embed
    )
    
    trainable_params = sum(p.numel() for p in ae_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in ae_model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"Total parameters (incl. frozen): {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Compression ratio: {hidden_dim / args.latent_dim:.1f}x")
    
    # Train
    print(f"\n{'='*100}")
    print(f"Training: Condition=y, Latent={args.latent_dim}")
    print(f"{'='*100}")
    ae_model = train_autoencoder(
        ae_model, teacher_model,
        train_loader, val_loader,
        model_name,
        output_dir=args.output_dir,
        epochs=args.epochs, lr=args.lr,
        device=args.device, patience=args.patience,
        alpha_mse=args.alpha_mse,
        alpha_cosine=args.alpha_cosine,
        alpha_logit=args.alpha_logit,
        alpha_logit_mse=args.alpha_logit_mse,
    )
    
    save_path = os.path.join(args.output_dir, f"best_ae_{model_name}.pt")
    print(f"\nTraining completed! Best model saved as: {save_path}")


if __name__ == "__main__":
    main()

