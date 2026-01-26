import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed

# Add parent directory to path for data_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from data_utils.indexed_dataset import MMapIndexedDataset
except ImportError:
    print("Warning: data_utils not found. Ensure the path is correct.")

# ========================================================================================
# AutoEncoder Model
# ========================================================================================

class ConditionalAutoEncoder(nn.Module):
    """AutoEncoder with Y condition (using teacher embedding)"""
    def __init__(self, input_dim=1600, latent_dim=8, teacher_embed=None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Y condition embedding: use teacher's word embedding (frozen)
        # nn.Embedding을 복사하지 않고 참조만 하거나, 가중치만 가져옵니다.
        # DDP 환경에서는 파라미터 등록을 피하기 위해 buffer로 등록하거나 forward에서 처리
        self.y_embed = teacher_embed 
        # 주의: teacher_embed는 외부 모델의 모듈이므로 여기서 grad 설정을 다시 할 필요는 없으나 안전장치
        for p in self.y_embed.parameters():
            p.requires_grad = False
        
        # Encoder input: hidden + teacher_embed(y)
        enc_input_dim = input_dim * 2 
        
        # Encoder Structure Construction
        enc_dims = self._get_dims(enc_input_dim, input_dim)
        if enc_dims[-1] != input_dim:
            enc_dims.append(input_dim)
        if latent_dim < input_dim:
            latent_dims = self._get_dims(input_dim, latent_dim)
            enc_dims.extend(latent_dims[1:])
        elif latent_dim > input_dim:
            enc_dims.append(latent_dim)
        
        encoder_layers = []
        for i in range(len(enc_dims) - 1):
            encoder_layers.append(nn.Linear(enc_dims[i], enc_dims[i+1]))
            if i < len(enc_dims) - 2:
                encoder_layers.append(nn.LayerNorm(enc_dims[i+1]))
                encoder_layers.append(nn.GELU())
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder Structure Construction
        dec_input_dim = latent_dim + input_dim
        decoder_dims = self._get_dims(dec_input_dim, input_dim)
        
        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i+1]))
            if i < len(decoder_dims) - 2:
                decoder_layers.append(nn.LayerNorm(decoder_dims[i+1]))
                decoder_layers.append(nn.GELU())
        decoder_layers.append(nn.LayerNorm(input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def _get_dims(self, input_dim, latent_dim):
        if latent_dim >= input_dim:
            return [input_dim, latent_dim]
        dims = [input_dim]
        current = input_dim
        while current // 2 > latent_dim:
            current = current // 2
            dims.append(current)
        dims.append(latent_dim)
        return dims
    
    def forward(self, hidden, y_token):
        # Teacher Embed는 이미 GPU에 올라가 있다고 가정 (Accelerator가 관리하거나 Teacher가 같은 GPU에 있음)
        cond = self.y_embed(y_token).to(hidden.dtype)
        
        enc_input = torch.cat([hidden, cond], dim=-1)
        z = self.encoder(enc_input)
        
        dec_input = torch.cat([z, cond], dim=-1)
        recon = self.decoder(dec_input)
        
        return recon, z

# ========================================================================================
# Helper Functions
# ========================================================================================

def get_lm_logits_from_hidden(model, hidden_states):
    """Get logits from hidden states"""
    # DDP unwrap (if needed, but usually model.module handles it or access directly)
    if hasattr(model, "module"):
        actual_model = model.module
    else:
        actual_model = model
    
    if hasattr(actual_model, "transformer") and hasattr(actual_model.transformer, "ln_f"):
        ln_f = actual_model.transformer.ln_f
        target_dtype = ln_f.weight.dtype
        hidden_states = hidden_states.to(target_dtype)
        hidden_norm = ln_f(hidden_states)
    else:
        hidden_norm = hidden_states
    
    lm_head = actual_model.lm_head
    head_dtype = lm_head.weight.dtype
    hidden_norm = hidden_norm.to(head_dtype)
    logits = lm_head(hidden_norm)
    return logits

# ========================================================================================
# Optimized Dataset & Collate (B200 Sequence Processing)
# ========================================================================================

class TeacherPredictionDatasetOptimized(Dataset):
    """
    토큰 단위가 아닌 시퀀스 인덱스만 반환하는 가벼운 데이터셋
    실제 데이터 로딩과 Teacher Inference는 collate_fn에서 배치 단위로 수행
    """
    def __init__(self, data_path, max_samples=None, max_length=512):
        self.max_length = max_length
        print(f"Loading data from {data_path}...")
        try:
            self.dataset = MMapIndexedDataset(data_path, skip_warmup=True)
            total_sequences = len(self.dataset)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            # Dummy implementation for testing without real data
            self.dataset = [np.random.randint(0, 50257, (1024,)) for _ in range(1000)]
            total_sequences = 1000

        if max_samples:
            end_idx = min(max_samples, total_sequences)
            self.valid_indices = range(end_idx)
        else:
            self.valid_indices = range(total_sequences)
            
        print(f"Optimized Dataset: {len(self.valid_indices)} sequences ready.")

    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # 인덱스만 반환
        return self.valid_indices[idx]

def collate_fn_optimized(batch_indices, dataset, teacher_model, tokenizer, device, max_length):
    """
    1. 시퀀스 배치 로드
    2. Teacher Inference (Sequence Level)
    3. Valid Tokens Flattening -> AE Training Batch
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    # 1. Load Data
    batch_tokens = []
    
    for idx in batch_indices:
        if isinstance(dataset.dataset, list): # For dummy data
            tokens = dataset.dataset[idx]
        else: # Real MMapDataset
            tokens = dataset.dataset[idx].astype(np.int64)
            
        tokens = tokens[:max_length + 1] # y(next_token)을 위해 +1 길이
        if len(tokens) > 1:
            batch_tokens.append(tokens)
    
    if not batch_tokens:
        return None

    # Pad Batch
    batch_size = len(batch_tokens)
    max_batch_len = max(len(t) for t in batch_tokens)
    
    # input_ids: 0 ~ L-1
    # target_ids(y): 1 ~ L
    input_ids = np.full((batch_size, max_batch_len-1), pad_id, dtype=np.int64)
    target_ids = np.full((batch_size, max_batch_len-1), pad_id, dtype=np.int64)
    attention_mask = np.zeros((batch_size, max_batch_len-1), dtype=np.int64)
    
    for i, tokens in enumerate(batch_tokens):
        seq_len = len(tokens) - 1
        input_ids[i, :seq_len] = tokens[:seq_len]
        target_ids[i, :seq_len] = tokens[1:] 
        attention_mask[i, :seq_len] = 1
        
    input_tensor = torch.tensor(input_ids, device=device, dtype=torch.long)
    mask_tensor = torch.tensor(attention_mask, device=device, dtype=torch.long)
    
    # 2. Teacher Inference
    # Teacher is already on the correct device (managed by Accelerator or manually set)
    with torch.no_grad():
        outputs = teacher_model(
            input_ids=input_tensor,
            attention_mask=mask_tensor,
            output_hidden_states=True,
            use_cache=False
        )
        last_hidden = outputs.hidden_states[-1] # [B, L, H]
        teacher_logits = outputs.logits  # [B, L, vocab_size]
    
    # 3. Extract y tokens (teacher's argmax prediction) and flatten valid tokens
    bool_mask = mask_tensor.bool()
    valid_hidden = last_hidden[bool_mask]  # [N, H]
    
    # Get teacher predictions for y (argmax of logits)
    teacher_predictions = torch.argmax(teacher_logits, dim=-1)  # [B, L]
    valid_y = teacher_predictions[bool_mask]  # [N]
    
    # GPU 메모리 절약을 위해 여기서 필요한 것만 리턴
    return {
        'hidden': valid_hidden,
        'y_token': valid_y
    }

# ========================================================================================
# Training & Evaluation
# ========================================================================================

def evaluate_model(
    ae_model, teacher_model, val_loader, 
    alpha_mse, alpha_cosine, alpha_logit, alpha_logit_mse,
    accelerator=None
):
    """Evaluate autoencoder on validation set"""
    ae_model.eval()
    val_loss = 0.0
    total_samples = 0
    
    is_main_process = accelerator.is_main_process if accelerator is not None else True
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating", disable=not is_main_process, leave=False)
        for batch in pbar:
            if batch is None: continue
            
            h = batch['hidden'] # Already on device
            y_token = batch['y_token']
            
            recon, z = ae_model(h, y_token=y_token)
            
            # MSE loss
            mse_loss = F.mse_loss(recon, h)
            
            # Cosine similarity loss
            cosine_sim = F.cosine_similarity(recon, h, dim=-1)
            cosine_loss = (1 - cosine_sim).mean()
            
            # Logit Loss Calculation
            # Cast to teacher dtype
            teacher_dtype = teacher_model.dtype if hasattr(teacher_model, 'dtype') else next(teacher_model.parameters()).dtype
            
            h_for_logits = h.to(teacher_dtype)
            recon_for_logits = recon.to(teacher_dtype)
            
            teacher_logits = get_lm_logits_from_hidden(teacher_model, h_for_logits)
            recon_logits = get_lm_logits_from_hidden(teacher_model, recon_for_logits)
            
            temperature = 1.0
            logit_loss = F.kl_div(
                F.log_softmax(recon_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature ** 2)
            
            logit_mse_loss = F.mse_loss(recon_logits, teacher_logits)
            
            loss = (alpha_mse * mse_loss + alpha_cosine * cosine_loss +
                    alpha_logit * logit_loss + alpha_logit_mse * logit_mse_loss)
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                # Bug Fix: Weighted average based on sample count
                batch_samples = h.size(0)
                val_loss += loss.item() * batch_samples
                total_samples += batch_samples
                
                # Update tqdm with current loss
                pbar.set_postfix({'loss': loss.item(), 'avg_loss': val_loss / total_samples if total_samples > 0 else 0.0})
    
    avg_val_loss = val_loss / total_samples if total_samples > 0 else float('inf')
    return avg_val_loss

def train_autoencoder_distributed(
    args, ae_model, teacher_model, tokenizer, accelerator
):
    # Datasets
    train_dataset = TeacherPredictionDatasetOptimized(
        args.data_path, max_samples=args.train_samples, max_length=args.max_length
    )
    
    if args.val_data_path:
        val_dataset = TeacherPredictionDatasetOptimized(
            args.val_data_path, max_samples=args.val_samples, max_length=args.max_length
        )
    elif args.val_samples and args.val_samples > 0:
        # Simple split logic (overlap might happen if not careful, but okay for large datasets)
        val_dataset = TeacherPredictionDatasetOptimized(
            args.data_path, max_samples=args.val_samples, max_length=args.max_length
        )
    else:
        val_dataset = None

    # Collate function wrappers
    def train_collate_wrapper(batch):
        return collate_fn_optimized(
            batch, train_dataset, teacher_model, tokenizer, accelerator.device, args.max_length
        )
    
    def val_collate_wrapper(batch):
        return collate_fn_optimized(
            batch, val_dataset, teacher_model, tokenizer, accelerator.device, args.max_length
        )

    # DataLoaders
    # Note: num_workers=0 because collate_fn uses CUDA (teacher forward pass)
    # CUDA cannot be used in forked subprocess, so we disable multiprocessing
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=0, collate_fn=train_collate_wrapper, pin_memory=False
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, 
            shuffle=False, num_workers=0, collate_fn=val_collate_wrapper, pin_memory=False
        )

    # Optimizer
    optimizer = torch.optim.AdamW(ae_model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.patience//2
    )

    # Prepare with Accelerator
    # IMPORTANT: Do NOT prepare teacher_model (it freezes layers strangely in some versions)
    # We just want AE, Optimizer, Dataloader to be distributed
    ae_model, optimizer, train_loader, val_loader = accelerator.prepare(
        ae_model, optimizer, train_loader, val_loader
    )
    
    # Evaluation Schedule
    total_steps = len(train_loader) * args.epochs
    eval_interval = max(1, total_steps // 10) # 10 evals total
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    current_step = 0
    
    if accelerator.is_main_process:
        print(f"Start Training on {accelerator.num_processes} GPUs")
        print(f"Total steps: {total_steps}, Eval interval: {eval_interval}")
    
    # Initial evaluation (before training)
    if val_loader is not None and accelerator.is_main_process:
        print("\n" + "="*100)
        print("Initial Evaluation (Before Training)")
        print("="*100)
        initial_val_loss = evaluate_model(
            ae_model, teacher_model, val_loader,
            args.alpha_mse, args.alpha_cosine, args.alpha_logit, args.alpha_logit_mse,
            accelerator=accelerator
        )
        print(f"Initial Val Loss: {initial_val_loss:.6f}")
        best_val_loss = initial_val_loss
        accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print("\n" + "="*100)
        print("Starting Training...")
        print("="*100)

    for epoch in range(args.epochs):
        ae_model.train()
        train_loss = 0.0
        train_samples = 0
        
        # Train Loop
        with tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch+1}") as pbar:
            for batch in pbar:
                if batch is None: continue
                
                h = batch['hidden'] # Already on device
                y_token = batch['y_token']
                
                optimizer.zero_grad()
                
                recon, z = ae_model(h, y_token=y_token)
                
                # Losses
                mse_loss = F.mse_loss(recon, h)
                cosine_sim = F.cosine_similarity(recon, h, dim=-1)
                cosine_loss = (1 - cosine_sim).mean()
                
                # Logit Loss
                teacher_dtype = teacher_model.dtype if hasattr(teacher_model, 'dtype') else torch.bfloat16
                h_for_logits = h.to(teacher_dtype)
                recon_for_logits = recon.to(teacher_dtype)
                
                with torch.no_grad():
                    teacher_logits = get_lm_logits_from_hidden(teacher_model, h_for_logits)
                recon_logits = get_lm_logits_from_hidden(teacher_model, recon_for_logits)
                
                temperature = 1.0
                logit_loss = F.kl_div(
                    F.log_softmax(recon_logits / temperature, dim=-1),
                    F.softmax(teacher_logits / temperature, dim=-1),
                    reduction="batchmean",
                ) * (temperature ** 2)
                
                logit_mse_loss = F.mse_loss(recon_logits, teacher_logits)
                
                loss = (args.alpha_mse * mse_loss + args.alpha_cosine * cosine_loss + 
                        args.alpha_logit * logit_loss + args.alpha_logit_mse * logit_mse_loss)
                
                # Backward via Accelerator
                accelerator.backward(loss)
                
                torch.nn.utils.clip_grad_norm_(ae_model.parameters(), 1.0)
                optimizer.step()
                
                batch_size = h.size(0)
                train_loss += loss.item() * batch_size
                train_samples += batch_size
                current_step += 1
                
                avg_train_loss = train_loss / train_samples if train_samples > 0 else 0.0
                pbar.set_postfix({
                    'loss': loss.item(),
                    'avg_loss': avg_train_loss,
                    'step': current_step
                })
                
                # Periodic Evaluation
                if current_step % eval_interval == 0 and val_loader is not None:
                    avg_val_loss = evaluate_model(
                        ae_model, teacher_model, val_loader,
                        args.alpha_mse, args.alpha_cosine, args.alpha_logit, args.alpha_logit_mse,
                        accelerator=accelerator
                    )
                    
                    # Scheduler Update (Using raw val loss)
                    scheduler.step(avg_val_loss)
                    
                    if accelerator.is_main_process:
                        print(f"\nStep {current_step} | Val Loss: {avg_val_loss:.6f}")
                        
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            epochs_no_improve = 0
                            
                            # Save Best Model
                            accelerator.wait_for_everyone()
                            unwrapped_model = accelerator.unwrap_model(ae_model)
                            os.makedirs(args.output_dir, exist_ok=True)
                            save_path = os.path.join(args.output_dir, f"best_ae_ld{args.latent_dim}.pt")
                            
                            # y_embed 제외하고 저장
                            state_dict = {k: v for k, v in unwrapped_model.state_dict().items() if 'y_embed' not in k}
                            torch.save(state_dict, save_path)
                            print(f"Saved best model to {save_path}")
                        else:
                            epochs_no_improve += 1
                    
                    ae_model.train() # Return to train mode

        # Epoch Summary
        if accelerator.is_main_process:
            avg_train_loss = train_loss / train_samples if train_samples > 0 else float('inf')
            print(f"Epoch {epoch+1} finished. Avg Train Loss: {avg_train_loss:.6f}")
            
            if epochs_no_improve >= args.patience:
                print("Early stopping triggered.")
                break

# ========================================================================================
# Main
# ========================================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data_0")
    parser.add_argument("--val_data_path", type=str, default=None)
    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--latent_dim", type=int, default=4)
    parser.add_argument("--train_samples", type=int, default=None)
    parser.add_argument("--val_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32, help="Sequence batch size per GPU")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--alpha_mse", type=float, default=2.0)
    parser.add_argument("--alpha_cosine", type=float, default=0.0)
    parser.add_argument("--alpha_logit", type=float, default=1.0)
    parser.add_argument("--alpha_logit_mse", type=float, default=0.0)
    
    args = parser.parse_args()
    
    # 1. Initialize Accelerator
    accelerator = Accelerator()
    set_seed(42)
    
    if accelerator.is_main_process:
        print(f"Running on {accelerator.num_processes} GPUs with Accelerate.")
    
    # 2. Load Teacher (Frozen)
    if accelerator.is_main_process:
        print("Loading teacher model...")
    
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.teacher_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_path,
        torch_dtype=torch.bfloat16,
    )
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    
    # Move teacher to device (Accelerate handles the rest via prepare, but teacher is kept separate)
    teacher_model.to(accelerator.device)
    
    # 3. Model Config Extraction
    if hasattr(teacher_model.config, 'n_embd'):
        hidden_dim = teacher_model.config.n_embd
    elif hasattr(teacher_model.config, 'hidden_size'):
        hidden_dim = teacher_model.config.hidden_size
    else:
        hidden_dim = 4096 # fallback
        
    if hasattr(teacher_model, 'transformer') and hasattr(teacher_model.transformer, 'wte'):
        teacher_embed = teacher_model.transformer.wte
    elif hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'embed_tokens'):
        teacher_embed = teacher_model.model.embed_tokens
    else:
        # Fallback or error
        teacher_embed = teacher_model.get_input_embeddings()

    # 4. Initialize AE
    ae_model = ConditionalAutoEncoder(
        input_dim=hidden_dim,
        latent_dim=args.latent_dim,
        teacher_embed=teacher_embed
    )
    
    # 5. Start Training
    train_autoencoder_distributed(args, ae_model, teacher_model, tokenizer, accelerator)

if __name__ == "__main__":
    main()