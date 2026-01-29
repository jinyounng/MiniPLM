import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed
import math

# Add parent directory to path for data_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from data_utils.indexed_dataset import MMapIndexedDataset
except ImportError:
    print("Warning: data_utils not found. Ensure the path is correct.")

# ========================================================================================
# AutoEncoder Model with Cross-Attention Decoder
# ========================================================================================

class ConditionalAutoEncoderCrossAttn(nn.Module):
    """AutoEncoder with Y condition and Cross-Attention Decoder
    Decoder uses Z as query to attend over Y_emb (key/value)
    """
    def __init__(self, input_dim=1600, latent_dim=8, teacher_embed=None, num_heads=8):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert input_dim % num_heads == 0, f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
        
        # Y condition embedding: use teacher's word embedding (frozen)
        if teacher_embed is not None:
            self.register_buffer('y_embed_weight', teacher_embed.weight.data.clone())
            self.y_embed_num_embeddings = teacher_embed.num_embeddings
            self.y_embed_embedding_dim = teacher_embed.embedding_dim
        else:
            self.y_embed_weight = None
            self.y_embed_num_embeddings = None
            self.y_embed_embedding_dim = None
        
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
            encoder_layers.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
            if i < len(enc_dims) - 2:
                encoder_layers.append(nn.LayerNorm(enc_dims[i + 1]))
                encoder_layers.append(nn.GELU())
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder: Cross-Attention based
        # Z (latent) → Query, Y_emb → Key/Value
        self.q_proj = nn.Linear(latent_dim, input_dim)  # Z → Query
        self.k_proj = nn.Linear(input_dim, input_dim)   # Y_emb → Key
        self.v_proj = nn.Linear(input_dim, input_dim)   # Y_emb → Value
        self.out_proj = nn.Linear(input_dim, input_dim)  # Output projection
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def _embed_y(self, y_token):
        """Separate method for y embedding (DDP-safe)"""
        return F.embedding(y_token, self.y_embed_weight)
    
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
    
    def _cross_attention(self, z, y_emb):
        """
        Cross-Attention: Z attends to Y_emb
        Args:
            z: [B, latent_dim] - Query source
            y_emb: [B, input_dim] - Key/Value source
        Returns:
            out: [B, input_dim] - Attended output
        """
        B = z.size(0)
        
        # Project to Q, K, V
        # For cross-attention, we need to handle single-vector case
        # Expand Z to have sequence dimension for multi-head attention
        z_expanded = z.unsqueeze(1)  # [B, 1, latent_dim]
        y_emb_expanded = y_emb.unsqueeze(1)  # [B, 1, input_dim]
        
        Q = self.q_proj(z_expanded)  # [B, 1, input_dim]
        K = self.k_proj(y_emb_expanded)  # [B, 1, input_dim]
        V = self.v_proj(y_emb_expanded)  # [B, 1, input_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, 1, head_dim]
        K = K.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, 1, head_dim]
        V = V.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, 1, head_dim]
        
        # Scaled dot-product attention
        # Since we have single query and single key/value, attention is just a weighted combination
        # But we can still use the attention mechanism for flexibility
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, num_heads, 1, 1]
        attn_weights = F.softmax(scores, dim=-1)  # [B, num_heads, 1, 1]
        attn_output = torch.matmul(attn_weights, V)  # [B, num_heads, 1, head_dim]
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, 1, num_heads, head_dim]
        attn_output = attn_output.view(B, 1, self.input_dim)  # [B, 1, input_dim]
        attn_output = attn_output.squeeze(1)  # [B, input_dim]
        
        # Output projection
        out = self.out_proj(attn_output)  # [B, input_dim]
        out = self.layer_norm(out)
        
        return out
    
    def forward(self, hidden, y_token):
        # Use buffer-based embedding (DDP-safe)
        cond = self._embed_y(y_token).float()
        
        # Ensure hidden is also float32 for consistency
        hidden_f32 = hidden.float()
        
        # Encoder: [hidden + Y_emb] → Z
        enc_input = torch.cat([hidden_f32, cond], dim=-1)
        z = self.encoder(enc_input)  # [B, latent_dim]
        
        # Decoder: Cross-Attention (Z attends to Y_emb)
        recon = self._cross_attention(z, cond)  # [B, input_dim]
        
        return recon, z

# ========================================================================================
# Helper Functions
# ========================================================================================

def get_lm_logits_from_hidden(model, hidden_states, force_float32=False):
    """Get logits from hidden states
    
    Args:
        model: Teacher model
        hidden_states: Input hidden states
        force_float32: If True, use float32 for computation (for gradient flow compatibility)
    """
    # DDP unwrap (if needed, but usually model.module handles it or access directly)
    if hasattr(model, "module"):
        actual_model = model.module
    else:
        actual_model = model
    
    if hasattr(actual_model, "transformer") and hasattr(actual_model.transformer, "ln_f"):
        ln_f = actual_model.transformer.ln_f
        if force_float32:
            # Use float32 for computation
            # CRITICAL: Detach teacher weights to prevent gradient flow through frozen teacher model
            hidden_states_f32 = hidden_states.float()
            with torch.no_grad():
                ln_f_weight_f32 = ln_f.weight.float().detach()
                ln_f_bias_f32 = ln_f.bias.float().detach() if ln_f.bias is not None else None
            hidden_norm = F.layer_norm(
                hidden_states_f32, 
                (hidden_states_f32.size(-1),),
                weight=ln_f_weight_f32,
                bias=ln_f_bias_f32,
                eps=ln_f.eps
            )
        else:
            target_dtype = ln_f.weight.dtype
            hidden_states = hidden_states.to(target_dtype)
            hidden_norm = ln_f(hidden_states)
    else:
        hidden_norm = hidden_states.float() if force_float32 else hidden_states
    
    lm_head = actual_model.lm_head
    if force_float32:
        # Use float32 for computation
        # CRITICAL: Detach teacher weights to prevent gradient flow through frozen teacher model
        with torch.no_grad():
            lm_head_weight_f32 = lm_head.weight.float().detach()
        logits = F.linear(hidden_norm, lm_head_weight_f32, bias=None)
    else:
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

def collate_fn_optimized(batch_indices, dataset, teacher_model, tokenizer, device, max_length, accelerator=None):
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
    # Ensure teacher model is in eval mode
    teacher_model.eval()
    
    # NOTE: Do NOT use accelerator.wait_for_everyone() in collate_fn
    # collate_fn runs in DataLoader context and may cause NCCL timeout
    # Each rank processes its batch independently
    
    with torch.no_grad():
        # Use torch.inference_mode() for better performance and safety
        with torch.inference_mode():
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
    
    # Ensure all processes participate in evaluation
    if accelerator is not None:
        accelerator.wait_for_everyone()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating", disable=not is_main_process, leave=False)
        for batch in pbar:
            if batch is None: continue
            
            h = batch['hidden'] # Already on device (may be bfloat16 from teacher)
            y_token = batch['y_token']
            
            # Convert to float32 immediately
            h_f32 = h.float()
            
            recon, z = ae_model(h_f32, y_token=y_token)
            # recon is already float32 from AE model forward
            
            # MSE loss - all in float32
            mse_loss = F.mse_loss(recon, h_f32)
            
            # Cosine similarity loss
            cosine_sim = F.cosine_similarity(recon, h_f32, dim=-1)
            cosine_loss = (1 - cosine_sim).mean()
            
            # Logit Loss Calculation
            # Use float32 for both to avoid dtype mismatch
            h_for_logits_f32 = h_f32
            recon_for_logits_f32 = recon
            
            # No synchronization needed here - each rank processes its own batch independently
            with torch.no_grad():
                teacher_logits_f32 = get_lm_logits_from_hidden(teacher_model, h_for_logits_f32, force_float32=True).detach()
            
            recon_logits_f32 = get_lm_logits_from_hidden(teacher_model, recon_for_logits_f32, force_float32=True)
            
            temperature = 1.0
            logit_loss = F.kl_div(
                F.log_softmax(recon_logits_f32 / temperature, dim=-1),
                F.softmax(teacher_logits_f32 / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature ** 2)
            
            logit_mse_loss = F.mse_loss(recon_logits_f32, teacher_logits_f32)
            
            loss = (alpha_mse * mse_loss + alpha_cosine * cosine_loss +
                    alpha_logit * logit_loss + alpha_logit_mse * logit_mse_loss)
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                # Bug Fix: Weighted average based on sample count
                batch_samples = h.size(0)
                val_loss += loss.item() * batch_samples
                total_samples += batch_samples
                
                # Update tqdm with current loss
                pbar.set_postfix({'loss': loss.item(), 'avg_loss': val_loss / total_samples if total_samples > 0 else 0.0})
    
    # Distributed reduction for accurate average across all processes
    if accelerator is not None and accelerator.num_processes > 1:
        # Use accelerator.gather() for safer distributed reduction
        total_samples_tensor = torch.tensor([total_samples], device=accelerator.device, dtype=torch.float32)
        val_loss_tensor = torch.tensor([val_loss], device=accelerator.device, dtype=torch.float32)
        
        # Gather from all processes and sum
        gathered_samples = accelerator.gather(total_samples_tensor)
        gathered_loss = accelerator.gather(val_loss_tensor)
        
        total_samples = int(gathered_samples.sum().item())
        val_loss = gathered_loss.sum().item()
        
        accelerator.wait_for_everyone()
    
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
        val_dataset = TeacherPredictionDatasetOptimized(
            args.data_path, max_samples=args.val_samples, max_length=args.max_length
        )
    else:
        val_dataset = None

    def train_collate_wrapper(batch):
        return collate_fn_optimized(
            batch, train_dataset, teacher_model, tokenizer, accelerator.device, args.max_length, accelerator=accelerator
        )
    
    def val_collate_wrapper(batch):
        return collate_fn_optimized(
            batch, val_dataset, teacher_model, tokenizer, accelerator.device, args.max_length, accelerator=accelerator
        )

    # DataLoaders
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

    optimizer = torch.optim.AdamW(ae_model.parameters(), lr=args.lr, weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.patience//2
    )

    ae_model, optimizer, train_loader, val_loader = accelerator.prepare(
        ae_model, optimizer, train_loader, val_loader
    )
    
    # [수정] Step 단위 평가 로직 제거 -> Epoch 단위로 변경
    best_val_loss = float('inf')
    epochs_no_improve = 0
    epochs_no_improve_tensor = torch.tensor([0], device=accelerator.device, dtype=torch.int)
    
    if accelerator.is_main_process:
        print(f"Start Training on {accelerator.num_processes} GPUs")
        print("Evaluation will be performed at the end of each epoch to prevent DDP desynchronization.")
        print(f"Decoder: Cross-Attention (Z → Y_emb)")
    
    # Initial evaluation
    if val_loader is not None:
        if accelerator.is_main_process:
            print("\n" + "="*100)
            print("Initial Evaluation (Before Training)")
            print("="*100)
        initial_val_loss = evaluate_model(
            ae_model, teacher_model, val_loader,
            args.alpha_mse, args.alpha_cosine, args.alpha_logit, args.alpha_logit_mse,
            accelerator=accelerator
        )
        if accelerator.is_main_process:
            print(f"Initial Val Loss: {initial_val_loss:.6f}")
        best_val_loss = initial_val_loss
        accelerator.wait_for_everyone()
    
    for epoch in range(args.epochs):
        ae_model.train()
        train_loss = 0.0
        train_samples = 0
        
        # Train Loop
        with tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch+1}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # [중요] 여기서 continue를 타면 rank 간 step 차이가 발생함
                # 따라서 이 루프 안에는 동기화(Broadcast, Barrier) 로직이 절대 있으면 안 됨
                if batch is None: continue
                
                h = batch['hidden']
                y_token = batch['y_token']
                h_f32 = h.float()
                
                recon, z = ae_model(h_f32, y_token=y_token)
                
                mse_loss = F.mse_loss(recon, h_f32)
                cosine_sim = F.cosine_similarity(recon, h_f32, dim=-1)
                cosine_loss = (1 - cosine_sim).mean()
                
                h_for_logits_f32 = h_f32
                recon_for_logits_f32 = recon
                
                with torch.no_grad():
                    teacher_logits_f32 = get_lm_logits_from_hidden(teacher_model, h_for_logits_f32, force_float32=True).detach()
                
                recon_logits_f32 = get_lm_logits_from_hidden(teacher_model, recon_for_logits_f32, force_float32=True)
                
                temperature = 1.0
                logit_loss = F.kl_div(
                    F.log_softmax(recon_logits_f32 / temperature, dim=-1),
                    F.softmax(teacher_logits_f32 / temperature, dim=-1),
                    reduction="batchmean",
                ) * (temperature ** 2)
                
                logit_mse_loss = F.mse_loss(recon_logits_f32, teacher_logits_f32)
                
                loss = (args.alpha_mse * mse_loss + args.alpha_cosine * cosine_loss + 
                        args.alpha_logit * logit_loss + args.alpha_logit_mse * logit_mse_loss)
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(ae_model.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                batch_size = h.size(0)
                train_loss += loss.item() * batch_size
                train_samples += batch_size
                
                avg_train_loss = train_loss / train_samples if train_samples > 0 else 0.0
                pbar.set_postfix({'loss': loss.item(), 'avg_loss': avg_train_loss})

        # [수정] Epoch가 끝난 후 안전하게 평가 수행 (모든 Rank가 여기서 만남)
        if val_loader is not None:
            # 모든 프로세스가 Epoch를 마칠 때까지 대기
            accelerator.wait_for_everyone()
            
            avg_val_loss = evaluate_model(
                ae_model, teacher_model, val_loader,
                args.alpha_mse, args.alpha_cosine, args.alpha_logit, args.alpha_logit_mse,
                accelerator=accelerator
            )
            
            scheduler.step(avg_val_loss)
            
            if accelerator.is_main_process:
                avg_train_loss = train_loss / train_samples if train_samples > 0 else float('inf')
                print(f"\nEpoch {epoch+1} Finished | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(ae_model)
                    os.makedirs(args.output_dir, exist_ok=True)
                    save_path = os.path.join(args.output_dir, f"best_ae_crossattn_ld{args.latent_dim}.pt")
                    torch.save(unwrapped_model.state_dict(), save_path)
                    print(f"Saved best model to {save_path}")
                else:
                    epochs_no_improve += 1
                
                epochs_no_improve_tensor[0] = epochs_no_improve
            
            # Broadcast early stopping status
            if accelerator.num_processes > 1:
                accelerator.wait_for_everyone()
                dist.broadcast(epochs_no_improve_tensor, src=0)
                epochs_no_improve = epochs_no_improve_tensor.item()
            
            if epochs_no_improve >= args.patience:
                if accelerator.is_main_process:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            ae_model.train() # 학습 모드 복귀

    if accelerator.is_main_process:
        print("Training Completed.")

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
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads for cross-attention decoder")
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
    
    # Set seed AFTER accelerator initialization to ensure proper distributed seeding
    # Each process gets a different seed based on its rank
    seed = 42 + accelerator.process_index
    set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if accelerator.is_main_process:
        print(f"Running on {accelerator.num_processes} GPUs with Accelerate.")
        print(f"Main process seed: {seed}")
    
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
    
    # Ensure teacher model is in eval mode and synchronized across processes
    teacher_model.eval()
    # Synchronize all processes before proceeding
    accelerator.wait_for_everyone()
    
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
    
    # Check if num_heads divides hidden_dim
    if hidden_dim % args.num_heads != 0:
        if accelerator.is_main_process:
            print(f"Warning: hidden_dim ({hidden_dim}) not divisible by num_heads ({args.num_heads})")
            print(f"Adjusting num_heads to largest divisor <= {args.num_heads}")
        # Find largest divisor
        for nh in range(args.num_heads, 0, -1):
            if hidden_dim % nh == 0:
                args.num_heads = nh
                break
        if accelerator.is_main_process:
            print(f"Using num_heads = {args.num_heads}")

    # 4. Initialize AE with Cross-Attention Decoder
    ae_model = ConditionalAutoEncoderCrossAttn(
        input_dim=hidden_dim,
        latent_dim=args.latent_dim,
        teacher_embed=teacher_embed,
        num_heads=args.num_heads
    )
    
    if accelerator.is_main_process:
        print(f"AE Model initialized:")
        print(f"  Input Dim: {hidden_dim}")
        print(f"  Latent Dim: {args.latent_dim}")
        print(f"  Decoder: Cross-Attention (Z → Y_emb)")
        print(f"  Attention Heads: {args.num_heads}")
    
    # 5. Start Training
    train_autoencoder_distributed(args, ae_model, teacher_model, tokenizer, accelerator)

if __name__ == "__main__":
    main()
