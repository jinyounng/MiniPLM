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
from data_utils.indexed_dataset import MMapIndexedDataset

# Import encodec RVQ
try:
    from encodec.quantization.core_vq import ResidualVectorQuantization
except ImportError:
    print("Warning: encodec not found. Please install: pip install encodec")
    sys.exit(1)


# ========================================================================================
# Compressor Model with Embedding Conditioning
# ========================================================================================

class HiddenStateCompressorEmb(nn.Module):
    """
    Compression model with embedding conditioning:
    - Encoder: Y_emb + hidden → compressed
    - Decoder: Z + Y_emb → hidden 복원
    Uses encodec's ResidualVectorQuantization
    """
    def __init__(
        self,
        teacher_hidden_dim=4096,
        embedding_dim=None,  # Will be set from teacher model
        compressed_dim=1024,
        num_stages=25,
        num_codes=1024,
        vocab_size=32000,
        decay=0.99,
        kmeans_init=True,
        kmeans_iters=50,
        threshold_ema_dead_code=2,
        encoder_depth=3,
        decoder_depth=3,
    ):
        super().__init__()
        self.teacher_hidden_dim = teacher_hidden_dim
        self.embedding_dim = embedding_dim or teacher_hidden_dim  # Default to hidden_dim if not specified
        self.compressed_dim = compressed_dim
        
        # Encoder: (Y_emb + hidden) → compressed
        # Input: [B, embedding_dim + teacher_hidden_dim]
        enc_input_dim = self.embedding_dim + teacher_hidden_dim
        enc_dims = np.linspace(enc_input_dim, compressed_dim, encoder_depth + 1).astype(int).tolist()
        enc_layers = []
        for i in range(encoder_depth):
            enc_layers.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
            enc_layers.append(nn.LayerNorm(enc_dims[i + 1]))
            enc_layers.append(nn.GELU())
        self.encoder = nn.Sequential(*enc_layers)

        self.rvq = ResidualVectorQuantization(
            dim=compressed_dim,
            codebook_size=num_codes,
            num_quantizers=num_stages,
            decay=decay,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )

        # Decoder: (Z + Y_emb) → hidden 복원
        # Input: [B, compressed_dim + embedding_dim]
        dec_input_dim = compressed_dim + self.embedding_dim
        dec_dims = np.linspace(dec_input_dim, teacher_hidden_dim, decoder_depth + 1).astype(int).tolist()
        dec_layers = []
        for i in range(decoder_depth):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < decoder_depth - 1:
                dec_layers.append(nn.LayerNorm(dec_dims[i + 1]))
                dec_layers.append(nn.GELU())
        self.decoder = nn.Sequential(*dec_layers)
        
    def forward(self, teacher_hidden, token_embeddings, return_indices=False, n_q=None, compute_perplexity=False):
        """
        Args:
            teacher_hidden: [B, teacher_hidden_dim]
            token_embeddings: [B, embedding_dim] — Y_emb
            return_indices: whether to return quantization indices
            n_q: number of quantizers to use (None = use all)
            compute_perplexity: whether to compute perplexity (expensive, use only in eval or periodically)
            
        Returns:
            hidden_recon: [B, teacher_hidden_dim] — 복원된 hidden
            indices (optional): [num_stages, B] or [n_q, B]
            info: dict with perplexities, commit_losses, etc.
        """
        # Encoder: Y_emb + hidden
        enc_input = torch.cat([token_embeddings, teacher_hidden], dim=-1)  # [B, embedding_dim + teacher_hidden_dim]
        z_e = self.encoder(enc_input)  # [B, compressed_dim]
        
        # Reshape for encodec RVQ: expects [B, D, N] where N=1 for our case
        z_e_reshaped = z_e.unsqueeze(-1)  # [B, D, 1]
        
        # Quantize
        z_q, indices, commit_losses = self.rvq(z_e_reshaped, n_q=n_q)
        # z_q: [B, D, 1], indices: [num_stages, B] or [n_q, B], commit_losses: [num_stages] or [n_q]
        
        # Reshape back
        z_q = z_q.squeeze(-1)  # [B, D]
        
        # Decoder: Z + Y_emb
        dec_input = torch.cat([z_q, token_embeddings], dim=-1)  # [B, compressed_dim + embedding_dim]
        hidden_recon = self.decoder(dec_input)  # [B, teacher_hidden_dim]
        
        # Calculate perplexity only when requested (expensive operation)
        perplexities = []
        if compute_perplexity:
            for stage_indices in indices:
                # Count code usage
                code_counts = torch.bincount(stage_indices.flatten(), minlength=self.rvq.layers[0]._codebook.codebook_size)
                probs = code_counts.float() / code_counts.sum()
                probs = probs[probs > 0]  # Remove zeros
                perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
                perplexities.append(perplexity.item())
        
        info = {
            'perplexities': perplexities,
            'avg_perplexity': sum(perplexities) / len(perplexities) if perplexities else 0.0,
            'commit_losses': commit_losses
        }
        
        if return_indices:
            return hidden_recon, indices, info
        else:
            return hidden_recon, info
    
    def compress(self, teacher_hidden, token_embeddings, n_q=None):
        """
        Compress to indices only (for storage)
        Args:
            teacher_hidden: [B, teacher_hidden_dim]
            token_embeddings: [B, embedding_dim]
        Returns: [num_stages, B] or [n_q, B] int64 indices
        """
        enc_input = torch.cat([token_embeddings, teacher_hidden], dim=-1)
        z_e = self.encoder(enc_input)
        z_e_reshaped = z_e.unsqueeze(-1)  # [B, D, 1]
        indices = self.rvq.encode(z_e_reshaped, n_q=n_q)
        return indices
    
    def decompress(self, indices, token_embeddings):
        """
        Decompress from indices → hidden 복원
        Args:
            indices: [num_stages, B] or [n_q, B]
            token_embeddings: [B, embedding_dim] — Y_emb
        Returns:
            hidden_recon: [B, teacher_hidden_dim]
        """
        z_q = self.rvq.decode(indices)  # [B, D, 1]
        z_q = z_q.squeeze(-1)  # [B, D]
        dec_input = torch.cat([z_q, token_embeddings], dim=-1)  # [B, compressed_dim + embedding_dim]
        return self.decoder(dec_input)


# ========================================================================================
# Helper Functions
# ========================================================================================

def get_lm_logits_from_hidden(model, hidden_states, force_float32=False):
    """Get logits from hidden states"""
    # Handle DataParallel/DDP wrapper
    if hasattr(model, "module"):
        actual_model = model.module
    else:
        actual_model = model
    
    if hasattr(actual_model, "transformer") and hasattr(actual_model.transformer, "ln_f"):
        ln_f = actual_model.transformer.ln_f
        if force_float32:
            with torch.no_grad():
                hidden_states_f32 = hidden_states.float()
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
# Dataset & Collate
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
    2. Teacher Inference (배치 단위로 한 번만)
    3. Hidden states, logits, token embeddings 추출
    
    Note: Teacher inference in collate_fn requires num_workers=0 (CUDA in forked process issue).
    This can be a bottleneck. Consider pre-extracting teacher hidden states if data loading
    becomes the bottleneck.
    """
    batch_sequences = []
    for seq_idx in batch_indices:
        seq = dataset.dataset[seq_idx]
        if isinstance(seq, np.ndarray):
            seq = torch.from_numpy(seq.astype(np.int64))
        else:
            seq = torch.tensor(seq, dtype=torch.long)
        
        # Truncate or pad
        if len(seq) > max_length:
            seq = seq[:max_length]
        elif len(seq) < max_length:
            pad_len = max_length - len(seq)
            seq = torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)])
        
        batch_sequences.append(seq)
    
    if len(batch_sequences) == 0:
        return None
    
    # Stack to batch
    input_ids = torch.stack(batch_sequences).to(device)  # [B, L]
    attention_mask = (input_ids != 0).long()  # [B, L]
    
    # Teacher Inference
    teacher_model.eval()
    
    with torch.no_grad():
        with torch.inference_mode():
            outputs = teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False
            )
            last_hidden = outputs.hidden_states[-1]  # [B, L, H]
            teacher_logits = outputs.logits  # [B, L, vocab_size]
            
            # Get token embeddings (Y_emb)
            # Handle different model architectures
            if hasattr(teacher_model, "module"):
                actual_model = teacher_model.module
            else:
                actual_model = teacher_model
            
            if hasattr(actual_model, "transformer") and hasattr(actual_model.transformer, "wte"):
                # GPT-style
                token_embeddings_full = actual_model.transformer.wte(input_ids)  # [B, L, embedding_dim]
            elif hasattr(actual_model, "model") and hasattr(actual_model.model, "embed_tokens"):
                # Llama-style
                token_embeddings_full = actual_model.model.embed_tokens(input_ids)  # [B, L, embedding_dim]
            elif hasattr(actual_model, "embed_tokens"):
                token_embeddings_full = actual_model.embed_tokens(input_ids)  # [B, L, embedding_dim]
            else:
                # Fallback: use first hidden state (usually embedding output)
                token_embeddings_full = outputs.hidden_states[0]  # [B, L, H]
    
    # Extract last token hidden states, logits, and embeddings
    seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
    batch_size = input_ids.size(0)
    
    hidden_list = []
    logits_list = []
    token_emb_list = []
    for i in range(batch_size):
        last_idx = seq_lengths[i].item()
        hidden_list.append(last_hidden[i, last_idx])  # [H]
        logits_list.append(teacher_logits[i, last_idx])  # [vocab_size]
        token_emb_list.append(token_embeddings_full[i, last_idx])  # [embedding_dim]
    
    hidden_batch = torch.stack(hidden_list)  # [B, H]
    logits_batch = torch.stack(logits_list)  # [B, vocab_size]
    token_emb_batch = torch.stack(token_emb_list)  # [B, embedding_dim]
    
    return {
        'hidden': hidden_batch,
        'logits': logits_batch,
        'token_embeddings': token_emb_batch  # Y_emb
    }


# ========================================================================================
# Evaluation
# ========================================================================================

def evaluate_model(
    compressor, teacher_model, val_loader, accelerator=None
):
    """Evaluate compressor on validation set"""
    compressor.eval()
    val_loss = 0.0
    total_samples = 0
    total_perplexity = 0.0
    
    is_main_process = accelerator.is_main_process if accelerator is not None else True
    
    if accelerator is not None:
        accelerator.wait_for_everyone()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating", disable=not is_main_process, leave=False)
        for batch in pbar:
            if batch is None: continue
            
            h = batch['hidden']  # [B, H]
            teacher_logits = batch['logits']  # [B, vocab_size]
            token_emb = batch['token_embeddings']  # [B, embedding_dim]
            h_f32 = h.float()
            token_emb_f32 = token_emb.float()
            
            # Forward: hidden 복원 → teacher lm_head로 logits 얻어서 KL
            hidden_recon, info = compressor(h_f32, token_emb_f32, compute_perplexity=True)
            logits = get_lm_logits_from_hidden(teacher_model, hidden_recon)
            kl_loss = F.kl_div(
                F.log_softmax(logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction='batchmean'
            )
            
            if not torch.isnan(kl_loss) and not torch.isinf(kl_loss):
                batch_samples = h.size(0)
                val_loss += kl_loss.item() * batch_samples
                total_samples += batch_samples
                if info['avg_perplexity'] > 0:
                    total_perplexity += info['avg_perplexity'] * batch_samples
                
                pbar.set_postfix({
                    'loss': kl_loss.item(),
                    'avg_loss': val_loss / total_samples if total_samples > 0 else 0.0,
                    'perplexity': info['avg_perplexity'] if info['avg_perplexity'] > 0 else 0.0
                })
    
    # Distributed reduction
    if accelerator is not None and accelerator.num_processes > 1:
        total_samples_tensor = torch.tensor([total_samples], device=accelerator.device, dtype=torch.float32)
        val_loss_tensor = torch.tensor([val_loss], device=accelerator.device, dtype=torch.float32)
        perplexity_tensor = torch.tensor([total_perplexity], device=accelerator.device, dtype=torch.float32)
        
        gathered_samples = accelerator.gather(total_samples_tensor)
        gathered_loss = accelerator.gather(val_loss_tensor)
        gathered_perplexity = accelerator.gather(perplexity_tensor)
        
        total_samples = int(gathered_samples.sum().item())
        val_loss = gathered_loss.sum().item()
        total_perplexity = gathered_perplexity.sum().item()
        
        accelerator.wait_for_everyone()
    
    avg_val_loss = val_loss / total_samples if total_samples > 0 else float('inf')
    avg_perplexity = total_perplexity / total_samples if total_samples > 0 else 0.0
    
    return avg_val_loss, avg_perplexity


# ========================================================================================
# Training
# ========================================================================================

def train_compressor_distributed(
    args, compressor, teacher_model, tokenizer, accelerator
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

    # Optimizer: Only encoder and decoder (RVQ updates via EMA)
    optimizer = torch.optim.AdamW(
        list(compressor.encoder.parameters()) + 
        list(compressor.decoder.parameters()),
        lr=args.lr, weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.patience//2
    )

    # Prepare with Accelerator (wraps model in DDP)
    compressor, optimizer, train_loader, val_loader = accelerator.prepare(
        compressor, optimizer, train_loader, val_loader
    )
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    epochs_no_improve_tensor = torch.tensor([0], device=accelerator.device, dtype=torch.int)
    
    # Progressive training: start with fewer quantizers and gradually increase
    use_progressive_training = args.progressive_training
    if use_progressive_training:
        progressive_schedule = [
            (0.0, 1),
            (0.25, max(1, args.num_stages // 2)),
            (0.5, max(2, args.num_stages * 3 // 4)),
            (0.75, args.num_stages)
        ]
    else:
        progressive_schedule = None
    
    if accelerator.is_main_process:
        print(f"Start Training on {accelerator.num_processes} GPUs")
        print("Evaluation will be performed at the end of each epoch to prevent DDP desynchronization.")
        print(f"RVQ Config: {args.num_stages} stages, {args.num_codes} codes/stage, {args.compressed_dim} dim")
        print(f"Using encodec's ResidualVectorQuantization")
        print(f"Progressive Training: {use_progressive_training}")
        print(f"Dead Code Threshold: {args.threshold_ema_dead_code} (batch_size={args.batch_size})")
        print(f"Note: Perplexity computation disabled during training for efficiency")
    
    # Initial evaluation
    if val_loader is not None:
        if accelerator.is_main_process:
            print("\n" + "="*100)
            print("Initial Evaluation (Before Training)")
            print("="*100)
        initial_val_loss, initial_perplexity = evaluate_model(
            compressor, teacher_model, val_loader, accelerator=accelerator
        )
        if accelerator.is_main_process:
            print(f"Initial Val Loss: {initial_val_loss:.6f}, Perplexity: {initial_perplexity:.2f}")
        best_val_loss = initial_val_loss
        accelerator.wait_for_everyone()
    
    for epoch in range(args.epochs):
        compressor.train()
        train_loss = 0.0
        train_samples = 0
        total_commitment_loss = 0.0
        total_perplexity = 0.0
        
        # Determine n_q for progressive training
        n_q = None
        if use_progressive_training and progressive_schedule:
            epoch_progress = epoch / args.epochs
            for threshold, n_stages in progressive_schedule:
                if epoch_progress >= threshold:
                    n_q = n_stages
            n_q = n_q or args.num_stages
        
        compute_perplexity_period = args.perplexity_compute_period
        step_count = 0
        
        # Train Loop
        with tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch+1}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                if batch is None: continue
                
                h = batch['hidden']  # [B, H]
                teacher_logits = batch['logits']  # [B, vocab_size]
                token_emb = batch['token_embeddings']  # [B, embedding_dim]
                h_f32 = h.float()
                token_emb_f32 = token_emb.float()
                
                # Determine if we should compute perplexity this step
                should_compute_perplexity = (
                    compute_perplexity_period > 0 and 
                    step_count % compute_perplexity_period == 0
                )
                
                # Forward: hidden 복원 → teacher lm_head로 logits 얻어서 KL
                hidden_recon, info = compressor(h_f32, token_emb_f32, n_q=n_q, compute_perplexity=should_compute_perplexity)
                logits = get_lm_logits_from_hidden(teacher_model, hidden_recon)
                kl_loss = F.kl_div(
                    F.log_softmax(logits, dim=-1),
                    F.softmax(teacher_logits, dim=-1),
                    reduction='batchmean'
                )
                
                # Commitment loss from RVQ
                if len(info['commit_losses']) > 0:
                    stage_weights = torch.linspace(1.0, 0.5, len(info['commit_losses']), device=info['commit_losses'].device)
                    commit_loss = (info['commit_losses'] * stage_weights).mean()
                else:
                    commit_loss = torch.tensor(0.0, device=h_f32.device)
                
                # Total loss (logit KL + commitment)
                loss = kl_loss + args.commitment_weight * commit_loss
                
                step_count += 1
                
                # Backward
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    unwrapped_compressor = accelerator.unwrap_model(compressor)
                    accelerator.clip_grad_norm_(
                        list(unwrapped_compressor.encoder.parameters()) + 
                        list(unwrapped_compressor.decoder.parameters()),
                        1.0
                    )
                
                optimizer.step()
                optimizer.zero_grad()
                
                batch_size = h.size(0)
                train_loss += loss.item() * batch_size
                train_samples += batch_size
                total_commitment_loss += commit_loss.item() * batch_size
                if info['avg_perplexity'] > 0:
                    total_perplexity += info['avg_perplexity'] * batch_size
                
                avg_train_loss = train_loss / train_samples if train_samples > 0 else 0.0
                postfix_dict = {
                    'loss': loss.item(),
                    'kl': kl_loss.item(),
                    'commit': commit_loss.item(),
                    'avg_loss': avg_train_loss,
                }
                if n_q is not None:
                    postfix_dict['n_q'] = n_q
                if info['avg_perplexity'] > 0:
                    postfix_dict['perp'] = info['avg_perplexity']
                pbar.set_postfix(postfix_dict)

        # Epoch 끝난 후 평가
        if val_loader is not None:
            accelerator.wait_for_everyone()
            
            avg_val_loss, avg_perplexity = evaluate_model(
                compressor, teacher_model, val_loader, accelerator=accelerator
            )
            
            scheduler.step(avg_val_loss)
            
            if accelerator.is_main_process:
                avg_train_loss = train_loss / train_samples if train_samples > 0 else float('inf')
                avg_commitment = total_commitment_loss / train_samples if train_samples > 0 else 0.0
                avg_train_perplexity = total_perplexity / train_samples if train_samples > 0 else 0.0
                
                print(f"\nEpoch {epoch+1} Finished")
                print(f"  Train Loss: {avg_train_loss:.6f} (KL: {avg_train_loss - args.commitment_weight * avg_commitment:.6f}, Commit: {avg_commitment:.6f})")
                print(f"  Train Perplexity: {avg_train_perplexity:.2f}")
                print(f"  Val Loss: {avg_val_loss:.6f}, Val Perplexity: {avg_perplexity:.2f}")
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(compressor)
                    os.makedirs(args.output_dir, exist_ok=True)
                    # Include num_stages, num_codes, gamma(decay) in filename for experiment tracking
                    g_tag = f"g{str(args.decay).replace('.', 'p')}"
                    save_path = os.path.join(
                        args.output_dir,
                        f"best_rvq_emb_s{args.num_stages}_c{args.num_codes}_{g_tag}_d{args.compressed_dim}_enc{args.encoder_depth}_dec{args.decoder_depth}.pt"
                    )
                    torch.save(unwrapped_model.state_dict(), save_path)
                    print(f"  Saved best model to {save_path}")
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
            
            compressor.train()

    if accelerator.is_main_process:
        print("Training Completed.")


# ========================================================================================
# Main
# ========================================================================================

def main():
    parser = argparse.ArgumentParser(description="Train RVQ Compressor with Embedding Conditioning")
    
    # Data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, default=None)
    parser.add_argument("--train_samples", type=int, default=None)
    parser.add_argument("--val_samples", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=1024)
    
    # Model
    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument("--teacher_hidden_dim", type=int, default=4096)
    parser.add_argument("--embedding_dim", type=int, default=None, help="Token embedding dimension (default: same as teacher_hidden_dim)")
    parser.add_argument("--compressed_dim", type=int, default=1024)
    parser.add_argument("--num_stages", type=int, default=25)
    parser.add_argument("--num_codes", type=int, default=1024)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--decay", type=float, default=0.99, help="EMA decay rate for RVQ codebooks")
    parser.add_argument("--gamma", type=float, default=None, help="Alias for --decay (EMA decay rate)")
    parser.add_argument("--kmeans_init", action='store_true', default=True)
    parser.add_argument("--kmeans_iters", type=int, default=50)
    parser.add_argument("--threshold_ema_dead_code", type=int, default=10)
    parser.add_argument("--encoder_depth", type=int, default=3, help="Number of encoder blocks (Linear+LN+GELU)")
    parser.add_argument("--decoder_depth", type=int, default=3, help="Number of decoder blocks (last block is Linear only)")
    
    # Training options
    parser.add_argument("--progressive_training", action='store_true', 
                        help="Use progressive training: start with fewer quantizers, gradually increase")
    parser.add_argument("--perplexity_compute_period", type=int, default=0,
                        help="Compute perplexity every N steps during training (0 = never, only in eval)")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--commitment_weight", type=float, default=0.25)
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True)
    
    args = parser.parse_args()
    
    # Handle --gamma as alias for --decay
    if args.gamma is not None:
        args.decay = args.gamma
    
    # Initialize Accelerator
    accelerator = Accelerator()
    
    # Set seed
    seed = 42 + accelerator.process_index
    set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if accelerator.is_main_process:
        print(f"Running on {accelerator.num_processes} GPUs with Accelerate.")
        print(f"Main process seed: {seed}")
    
    # Load Teacher (Frozen)
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
    
    # Get vocab size from teacher if not specified
    if hasattr(teacher_model.config, 'vocab_size'):
        args.vocab_size = teacher_model.config.vocab_size
    
    # Get teacher hidden dim from config
    if hasattr(teacher_model.config, 'hidden_size'):
        args.teacher_hidden_dim = teacher_model.config.hidden_size
    elif hasattr(teacher_model.config, 'n_embd'):
        args.teacher_hidden_dim = teacher_model.config.n_embd
    
    # Get embedding dim from teacher model
    if args.embedding_dim is None:
        # Try to get from config
        if hasattr(teacher_model.config, 'hidden_size'):
            args.embedding_dim = teacher_model.config.hidden_size
        elif hasattr(teacher_model.config, 'n_embd'):
            args.embedding_dim = teacher_model.config.n_embd
        else:
            # Fallback: check actual embedding layer
            if hasattr(teacher_model, "module"):
                actual_model = teacher_model.module
            else:
                actual_model = teacher_model
            
            if hasattr(actual_model, "transformer") and hasattr(actual_model.transformer, "wte"):
                args.embedding_dim = actual_model.transformer.wte.weight.size(1)
            elif hasattr(actual_model, "model") and hasattr(actual_model.model, "embed_tokens"):
                args.embedding_dim = actual_model.model.embed_tokens.weight.size(1)
            else:
                args.embedding_dim = args.teacher_hidden_dim  # Default fallback
    
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_path)
    
    # Initialize Compressor
    compressor = HiddenStateCompressorEmb(
        teacher_hidden_dim=args.teacher_hidden_dim,
        embedding_dim=args.embedding_dim,
        compressed_dim=args.compressed_dim,
        num_stages=args.num_stages,
        num_codes=args.num_codes,
        vocab_size=args.vocab_size,
        decay=args.decay,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        threshold_ema_dead_code=args.threshold_ema_dead_code,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
    )
    
    if accelerator.is_main_process:
        print(f"Compressor initialized:")
        print(f"  Teacher Hidden Dim: {args.teacher_hidden_dim}")
        print(f"  Embedding Dim: {args.embedding_dim}")
        print(f"  Compressed Dim: {args.compressed_dim}")
        print(f"  Encoder Depth: {args.encoder_depth}")
        print(f"  Decoder Depth: {args.decoder_depth}")
        print(f"  RVQ Stages: {args.num_stages}")
        print(f"  Codes per Stage: {args.num_codes}")
        print(f"  Vocab Size: {args.vocab_size}")
        print(f"  Compression Ratio: {args.num_stages * np.log2(args.num_codes):.1f} bits per sample")
        print(f"  Using encodec's ResidualVectorQuantization")
        print(f"  Encoder input: [Y_emb + hidden] → compressed")
        print(f"  Decoder input: [Z + Y_emb] → hidden")
    
    # Train
    train_compressor_distributed(args, compressor, teacher_model, tokenizer, accelerator)


if __name__ == "__main__":
    main()
