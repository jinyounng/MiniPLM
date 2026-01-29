"""
Conditional AutoEncoder using pl_bolts AE with [x, y_emb] concatenation wrapper

핵심:
- pl_bolts의 AE를 그대로 사용
- 입력을 [x, y_emb]로 concatenate해서 넣는 wrapper
- encoder/decoder에 y 넣는 건 스위치로 토글 가능
"""

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

# Add parent directory to path for data_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from data_utils.indexed_dataset import MMapIndexedDataset
except ImportError:
    print("Warning: data_utils not found. Ensure the path is correct.")

try:
    from pl_bolts.models.autoencoders import AE
    PL_BOLTS_AVAILABLE = True
except ImportError:
    print("Warning: pl_bolts not found. Install with: pip install pytorch-lightning-bolts")
    PL_BOLTS_AVAILABLE = False
    # Fallback: simple AE structure
    class AE(nn.Module):
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, input_dim)
            )
        def forward(self, x):
            z = self.encoder(x)
            recon = self.decoder(z)
            return recon, z


# ========================================================================================
# Wrapper: pl_bolts AE with [x, y_emb] concatenation
# ========================================================================================

class ConditionalAEWrapper(nn.Module):
    """
    Wrapper around pl_bolts AE that:
    1. Concatenates [x, y_emb] as input
    2. Optionally adds y_emb to encoder/decoder (toggle via flags)
    
    Args:
        input_dim: dimension of x (hidden state)
        embedding_dim: dimension of y_emb (teacher embedding)
        latent_dim: latent dimension
        teacher_embed: teacher's embedding layer (for y_emb lookup)
        use_y_in_encoder: if True, encoder input = [x, y_emb], else just x
        use_y_in_decoder: if True, decoder input = [z, y_emb], else just z
    """
    def __init__(
        self,
        input_dim=4096,
        embedding_dim=None,
        latent_dim=25,
        teacher_embed=None,
        use_y_in_encoder=True,
        use_y_in_decoder=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim or (teacher_embed.embedding_dim if teacher_embed else input_dim)
        self.latent_dim = latent_dim
        self.use_y_in_encoder = use_y_in_encoder
        self.use_y_in_decoder = use_y_in_decoder
        
        # Store teacher embedding (buffer for DDP safety)
        if teacher_embed is not None:
            self.register_buffer('y_embed_weight', teacher_embed.weight.data.clone())
        else:
            self.y_embed_weight = None
        
        # Determine AE input/output dimensions
        if use_y_in_encoder:
            ae_input_dim = input_dim + self.embedding_dim
        else:
            ae_input_dim = input_dim
        
        if use_y_in_decoder:
            ae_decoder_input_dim = latent_dim + self.embedding_dim
        else:
            ae_decoder_input_dim = latent_dim
        
        # Create pl_bolts AE (or fallback simple AE)
        # Note: pl_bolts AE is designed for images, so we'll use a simple MLP-based AE
        # that mimics the structure but works with 1D vectors
        self.ae = self._create_ae(ae_input_dim, latent_dim, ae_decoder_input_dim, input_dim)
    
    def _create_ae(self, enc_input_dim, latent_dim, dec_input_dim, output_dim):
        """Create encoder-decoder structure compatible with pl_bolts style"""
        # Simple MLP-based AE (pl_bolts AE is for images, so we adapt)
        encoder = nn.Sequential(
            nn.Linear(enc_input_dim, enc_input_dim // 2),
            nn.LayerNorm(enc_input_dim // 2),
            nn.GELU(),
            nn.Linear(enc_input_dim // 2, latent_dim)
        )
        
        decoder = nn.Sequential(
            nn.Linear(dec_input_dim, output_dim // 2),
            nn.LayerNorm(output_dim // 2),
            nn.GELU(),
            nn.Linear(output_dim // 2, output_dim)
        )
        
        class SimpleAE(nn.Module):
            def __init__(self, encoder, decoder):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
            
            def forward(self, x):
                z = self.encoder(x)
                recon = self.decoder(z)
                return recon, z
        
        return SimpleAE(encoder, decoder)
    
    def _embed_y(self, y_token):
        """Get y embedding from buffer"""
        if self.y_embed_weight is None:
            raise ValueError("y_embed_weight not initialized. Provide teacher_embed in __init__")
        return F.embedding(y_token, self.y_embed_weight)
    
    def forward(self, x, y_token=None):
        """
        Args:
            x: [B, input_dim] hidden states
            y_token: [B] token indices for conditioning (optional if use_y_* flags are False)
        
        Returns:
            recon: [B, input_dim] reconstructed hidden states
            z: [B, latent_dim] latent code
        """
        # Get y embedding if needed
        y_emb = None
        if (self.use_y_in_encoder or self.use_y_in_decoder) and y_token is not None:
            y_emb = self._embed_y(y_token).float()  # [B, embedding_dim]
        
        # Encoder input
        if self.use_y_in_encoder and y_emb is not None:
            enc_input = torch.cat([x.float(), y_emb], dim=-1)  # [B, input_dim + embedding_dim]
        else:
            enc_input = x.float()  # [B, input_dim]
        
        # Encode
        z = self.ae.encoder(enc_input)  # [B, latent_dim]
        
        # Decoder input
        if self.use_y_in_decoder and y_emb is not None:
            dec_input = torch.cat([z, y_emb], dim=-1)  # [B, latent_dim + embedding_dim]
        else:
            dec_input = z  # [B, latent_dim]
        
        # Decode
        recon = self.ae.decoder(dec_input)  # [B, input_dim]
        
        return recon, z


# ========================================================================================
# Helper Functions (same as train_ae_onthefly.py)
# ========================================================================================

def get_lm_logits_from_hidden(model, hidden_states, force_float32=False):
    """Get logits from hidden states"""
    if hasattr(model, "module"):
        actual_model = model.module
    else:
        actual_model = model
    
    if hasattr(actual_model, "transformer") and hasattr(actual_model.transformer, "ln_f"):
        ln_f = actual_model.transformer.ln_f
        if force_float32:
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
        with torch.no_grad():
            lm_head_weight_f32 = lm_head.weight.float().detach()
        logits = F.linear(hidden_norm, lm_head_weight_f32, bias=None)
    else:
        head_dtype = lm_head.weight.dtype
        hidden_norm = hidden_norm.to(head_dtype)
        logits = lm_head(hidden_norm)
    return logits


# ========================================================================================
# Dataset & Collate (same as train_ae_onthefly.py)
# ========================================================================================

class TeacherPredictionDatasetOptimized(Dataset):
    def __init__(self, data_path, max_samples=None, max_length=512):
        self.max_length = max_length
        print(f"Loading data from {data_path}...")
        try:
            self.dataset = MMapIndexedDataset(data_path, skip_warmup=True)
            total_sequences = len(self.dataset)
        except Exception as e:
            print(f"Error loading dataset: {e}")
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
        return self.valid_indices[idx]


def collate_fn_optimized(batch_indices, dataset, teacher_model, tokenizer, device, max_length, accelerator=None):
    """Same as train_ae_onthefly.py"""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    batch_tokens = []
    for idx in batch_indices:
        if isinstance(dataset.dataset, list):
            tokens = np.array(dataset.dataset[idx], dtype=np.int64)
        else:
            tokens = dataset.dataset[idx].astype(np.int64)
        tokens = tokens[: max_length + 1]
        if len(tokens) > 1:
            batch_tokens.append(tokens)
    
    if not batch_tokens:
        return None

    batch_size = len(batch_tokens)
    max_batch_len = max(len(t) for t in batch_tokens)
    
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
    
    teacher_model.eval()
    with torch.no_grad():
        with torch.inference_mode():
            outputs = teacher_model(
                input_ids=input_tensor,
                attention_mask=mask_tensor,
                output_hidden_states=True,
                use_cache=False
            )
            last_hidden = outputs.hidden_states[-1]
            teacher_logits = outputs.logits
    
    bool_mask = mask_tensor.bool()
    valid_hidden = last_hidden[bool_mask]
    teacher_predictions = torch.argmax(teacher_logits, dim=-1)
    valid_y = teacher_predictions[bool_mask]
    
    return {
        'hidden': valid_hidden,
        'y_token': valid_y
    }


# ========================================================================================
# Evaluation
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
    
    if accelerator is not None:
        accelerator.wait_for_everyone()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating", disable=not is_main_process, leave=False)
        for batch in pbar:
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
                batch_samples = h.size(0)
                val_loss += loss.item() * batch_samples
                total_samples += batch_samples
                
                pbar.set_postfix({
                    'loss': loss.item(), 
                    'avg_loss': val_loss / total_samples if total_samples > 0 else 0.0
                })
    
    if accelerator is not None and accelerator.num_processes > 1:
        total_samples_tensor = torch.tensor([total_samples], device=accelerator.device, dtype=torch.float32)
        val_loss_tensor = torch.tensor([val_loss], device=accelerator.device, dtype=torch.float32)
        
        gathered_samples = accelerator.gather(total_samples_tensor)
        gathered_loss = accelerator.gather(val_loss_tensor)
        
        total_samples = int(gathered_samples.sum().item())
        val_loss = gathered_loss.sum().item()
        
        accelerator.wait_for_everyone()
    
    avg_val_loss = val_loss / total_samples if total_samples > 0 else float('inf')
    return avg_val_loss


# ========================================================================================
# Training
# ========================================================================================

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
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    epochs_no_improve_tensor = torch.tensor([0], device=accelerator.device, dtype=torch.int)
    
    if accelerator.is_main_process:
        print(f"Start Training on {accelerator.num_processes} GPUs")
        print("Evaluation will be performed at the end of each epoch to prevent DDP desynchronization.")
        print(f"AE Config: input_dim={args.input_dim}, latent_dim={args.latent_dim}")
        print(f"  use_y_in_encoder={args.use_y_in_encoder}, use_y_in_decoder={args.use_y_in_decoder}")
    
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
        
        with tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch+1}") as pbar:
            for batch_idx, batch in enumerate(pbar):
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
                pbar.set_postfix({
                    'loss': loss.item(),
                    'avg_loss': avg_train_loss
                })

        # Epoch evaluation
        if val_loader is not None:
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
                    save_path = os.path.join(args.output_dir, f"best_ae_plbolts_ld{args.latent_dim}_enc{int(args.use_y_in_encoder)}_dec{int(args.use_y_in_decoder)}.pt")
                    torch.save(unwrapped_model.state_dict(), save_path)
                    print(f"  Saved best model to {save_path}")
                else:
                    epochs_no_improve += 1
                
                epochs_no_improve_tensor[0] = epochs_no_improve
            
            if accelerator.num_processes > 1:
                accelerator.wait_for_everyone()
                dist.broadcast(epochs_no_improve_tensor, src=0)
                epochs_no_improve = epochs_no_improve_tensor.item()
            
            if epochs_no_improve >= args.patience:
                if accelerator.is_main_process:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            ae_model.train()

    if accelerator.is_main_process:
        print("Training Completed.")


# ========================================================================================
# Main
# ========================================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Conditional AE using pl_bolts wrapper")
    
    # Data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, default=None)
    parser.add_argument("--train_samples", type=int, default=None)
    parser.add_argument("--val_samples", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=1024)
    
    # Model
    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument("--input_dim", type=int, default=4096)
    parser.add_argument("--latent_dim", type=int, default=25)
    parser.add_argument("--use_y_in_encoder", action='store_true', default=True,
                        help="Use y_emb in encoder input [x, y_emb]")
    parser.add_argument("--use_y_in_decoder", action='store_true', default=True,
                        help="Use y_emb in decoder input [z, y_emb]")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=10)
    
    # Loss weights
    parser.add_argument("--alpha_mse", type=float, default=0.0)
    parser.add_argument("--alpha_cosine", type=float, default=0.0)
    parser.add_argument("--alpha_logit", type=float, default=1.0)
    parser.add_argument("--alpha_logit_mse", type=float, default=0.0)
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True)
    
    args = parser.parse_args()
    
    # Initialize Accelerator
    accelerator = Accelerator()
    
    seed = 42 + accelerator.process_index
    set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if accelerator.is_main_process:
        print(f"Running on {accelerator.num_processes} GPUs with Accelerate.")
    
    # Load Teacher
    if accelerator.is_main_process:
        print(f"Loading teacher model from {args.teacher_path}...")
    
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_path,
        torch_dtype=torch.bfloat16,
        device_map=None
    )
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    teacher_model.to(accelerator.device)
    accelerator.wait_for_everyone()
    
    # Get dimensions from teacher config
    if hasattr(teacher_model.config, "n_embd"):
        args.input_dim = teacher_model.config.n_embd
    elif hasattr(teacher_model.config, "hidden_size"):
        args.input_dim = teacher_model.config.hidden_size
    
    # Get teacher embedding
    teacher_embed = None
    if hasattr(teacher_model, "transformer") and hasattr(teacher_model.transformer, "wte"):
        teacher_embed = teacher_model.transformer.wte
    elif hasattr(teacher_model, "model") and hasattr(teacher_model.model, "embed_tokens"):
        teacher_embed = teacher_model.model.embed_tokens
    elif hasattr(teacher_model, "get_input_embeddings"):
        teacher_embed = teacher_model.get_input_embeddings()
    
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_path)
    
    # Initialize AE with wrapper
    ae_model = ConditionalAEWrapper(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        teacher_embed=teacher_embed,
        use_y_in_encoder=args.use_y_in_encoder,
        use_y_in_decoder=args.use_y_in_decoder,
    )
    
    if accelerator.is_main_process:
        print(f"AE initialized:")
        print(f"  Input Dim: {args.input_dim}")
        print(f"  Latent Dim: {args.latent_dim}")
        print(f"  Use Y in Encoder: {args.use_y_in_encoder}")
        print(f"  Use Y in Decoder: {args.use_y_in_decoder}")
    
    # Train
    train_autoencoder_distributed(args, ae_model, teacher_model, tokenizer, accelerator)


if __name__ == "__main__":
    main()
