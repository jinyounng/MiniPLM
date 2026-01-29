"""
Eval script: Codebook(RVQ) / AE 모델의 hidden → logits 복원 품질 평가

Metrics:
  - Logit MSE: MSE(teacher_logits, recon_logits)
  - Logit KL:  KL(recon_logits || teacher_logits)
  - Hidden MSE: MSE(teacher_hidden, recon_hidden)
  - Hidden Cosine: cosine_similarity(teacher_hidden, recon_hidden)

Usage (run from project root or from scripts/AE/train):
  # AE
  python scripts/AE/train/eval_ae_rvq_metrics.py --model_type ae --checkpoint_path .../best_ae_ld25.pt \
    --teacher_path .../qwen-7B --data_path .../data_0 --latent_dim 25

  # RVQ (hidden 복원 모델)
  python scripts/AE/train/eval_ae_rvq_metrics.py --model_type rvq --checkpoint_path .../best_rvq_*.pt \
    --teacher_path .../qwen-7B --data_path .../data_0 --num_stages 4 --num_codes 1024 --compressed_dim 1024
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys
import numpy as np
import json

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from data_utils.indexed_dataset import MMapIndexedDataset


def get_lm_logits_from_hidden(model, hidden_states, force_float32=True):
    """Teacher hidden -> logits (eval only, no grad)"""
    if hasattr(model, "module"):
        actual_model = model.module
    else:
        actual_model = model

    with torch.no_grad():
        if hasattr(actual_model, "transformer") and hasattr(actual_model.transformer, "ln_f"):
            ln_f = actual_model.transformer.ln_f
            if force_float32:
                h = hidden_states.float()
                w = ln_f.weight.float()
                b = ln_f.bias.float() if ln_f.bias is not None else None
                hidden_norm = F.layer_norm(h, (h.size(-1),), weight=w, bias=b, eps=ln_f.eps)
            else:
                hidden_norm = ln_f(hidden_states.to(ln_f.weight.dtype))
        else:
            hidden_norm = hidden_states.float() if force_float32 else hidden_states

        lm_head = actual_model.lm_head
        if force_float32:
            logits = F.linear(hidden_norm, lm_head.weight.float(), bias=None)
        else:
            logits = lm_head(hidden_norm.to(lm_head.weight.dtype))
    return logits


# ------------------------------------------------------------------------------
# Dataset & Collate
# ------------------------------------------------------------------------------

class EvalHiddenDataset(Dataset):
    def __init__(self, data_path, max_samples=2000, max_length=512):
        self.max_length = max_length
        try:
            self.dataset = MMapIndexedDataset(data_path, skip_warmup=True)
            total = len(self.dataset)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.dataset = [np.random.randint(0, 50257, (512,)) for _ in range(min(2000, max_samples or 2000))]
            total = len(self.dataset)
        self.valid_indices = range(min(max_samples or total, total))
        print(f"Eval dataset: {len(self.valid_indices)} sequences")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, i):
        return self.valid_indices[i]


def collate_hidden_and_y(batch_indices, dataset, teacher_model, device, max_length):
    """Returns hidden [N,H], y_token [N], teacher_logits [N,V] (flatten valid tokens)."""
    pad_id = 0
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

    B = len(batch_tokens)
    L = max(len(t) - 1 for t in batch_tokens)
    input_ids = np.full((B, L), pad_id, dtype=np.int64)
    attn = np.zeros((B, L), dtype=np.int64)
    for i, t in enumerate(batch_tokens):
        n = len(t) - 1
        input_ids[i, :n] = t[:n]
        attn[i, :n] = 1

    inp = torch.tensor(input_ids, device=device, dtype=torch.long)
    mask = torch.tensor(attn, device=device, dtype=torch.long)

    teacher_model.eval()
    with torch.no_grad():
        out = teacher_model(
            input_ids=inp,
            attention_mask=mask,
            output_hidden_states=True,
            use_cache=False,
        )
    last_hidden = out.hidden_states[-1]   # [B,L,H]
    teacher_logits = out.logits           # [B,L,V]
    pred_y = teacher_logits.argmax(dim=-1)

    m = mask.bool()
    hidden_flat = last_hidden[m]   # [N,H]
    y_flat = pred_y[m]              # [N]
    logits_flat = teacher_logits[m] # [N,V]

    return {
        "hidden": hidden_flat,
        "y_token": y_flat,
        "teacher_logits": logits_flat,
    }


# ------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------

def compute_metrics(teacher_hidden, recon_hidden, teacher_model, temperature=1.0):
    """
    teacher_hidden, recon_hidden: [N, H]
    Returns: dict with logit_mse, logit_kl, hidden_mse, hidden_cosine
    """
    with torch.no_grad():
        teacher_logits = get_lm_logits_from_hidden(teacher_model, teacher_hidden)
        recon_logits = get_lm_logits_from_hidden(teacher_model, recon_hidden)
    # Logit MSE
    logit_mse = F.mse_loss(recon_logits.float(), teacher_logits.float()).item()
    # Logit KL (recon || teacher)
    logit_kl = F.kl_div(
        F.log_softmax(recon_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ).item() * (temperature ** 2)
    # Hidden MSE
    hidden_mse = F.mse_loss(recon_hidden.float(), teacher_hidden.float()).item()
    # Hidden Cosine (per token, then mean). [N,H] -> similarity per row -> scalar
    cos = F.cosine_similarity(recon_hidden.float(), teacher_hidden.float(), dim=-1)
    hidden_cosine = cos.mean().item()
    return {
        "logit_mse": logit_mse,
        "logit_kl": logit_kl,
        "hidden_mse": hidden_mse,
        "hidden_cosine": hidden_cosine,
    }


# ------------------------------------------------------------------------------
# AE Model loading
# ------------------------------------------------------------------------------

def load_ae_model(checkpoint_path, teacher_model, latent_dim, device):
    from train_ae_onthefly import ConditionalAutoEncoder

    teacher_embed = None
    if hasattr(teacher_model, "transformer") and hasattr(teacher_model.transformer, "wte"):
        teacher_embed = teacher_model.transformer.wte
    elif hasattr(teacher_model, "model") and hasattr(teacher_model.model, "embed_tokens"):
        teacher_embed = teacher_model.model.embed_tokens
    elif hasattr(teacher_model, "get_input_embeddings"):
        teacher_embed = teacher_model.get_input_embeddings()

    if hasattr(teacher_model.config, "n_embd"):
        input_dim = teacher_model.config.n_embd
    elif hasattr(teacher_model.config, "hidden_size"):
        input_dim = teacher_model.config.hidden_size
    else:
        input_dim = 4096

    ae = ConditionalAutoEncoder(input_dim=input_dim, latent_dim=latent_dim, teacher_embed=teacher_embed)
    state = torch.load(checkpoint_path, map_location="cpu")
    # drop keys that may come from buffer (y_embed_weight) if not in state
    ae_state = {k: v for k, v in state.items() if k in ae.state_dict()}
    if not ae_state and "y_embed_weight" not in state:
        ae_state = state
    ae.load_state_dict(ae_state, strict=False)
    ae.to(device)
    ae.eval()
    return ae, input_dim


# ------------------------------------------------------------------------------
# RVQ Model loading
# ------------------------------------------------------------------------------

def parse_model_params_from_filename(filename):
    """Parse parameters from checkpoint filename
    Examples:
        best_rvq_emb_s25_c4096_g0p95_d1024_enc4_dec4.pt
        best_rvq_s4_c1024_d1024_enc3_dec3.pt
    Returns dict with parsed params or None if parsing fails
    """
    import re
    params = {}
    
    # Extract num_stages (s25 -> 25)
    m = re.search(r'_s(\d+)_', filename)
    if m:
        params['num_stages'] = int(m.group(1))
    
    # Extract num_codes (c4096 -> 4096)
    m = re.search(r'_c(\d+)_', filename)
    if m:
        params['num_codes'] = int(m.group(1))
    
    # Extract gamma/decay (g0p95 -> 0.95)
    m = re.search(r'_g(\d+)p(\d+)_', filename)
    if m:
        params['decay'] = float(f"{m.group(1)}.{m.group(2)}")
    
    # Extract compressed_dim (d1024 -> 1024)
    m = re.search(r'_d(\d+)_', filename)
    if m:
        params['compressed_dim'] = int(m.group(1))
    
    # Extract encoder_depth (enc4 -> 4)
    m = re.search(r'_enc(\d+)_', filename)
    if m:
        params['encoder_depth'] = int(m.group(1))
    
    # Extract decoder_depth (dec4 -> 4)
    m = re.search(r'_dec(\d+)', filename)
    if m:
        params['decoder_depth'] = int(m.group(1))
    
    # Check if it's rvq_emb model
    if 'rvq_emb' in filename or 'emb' in filename:
        params['is_emb'] = True
    
    return params if params else None


def load_rvq_model(checkpoint_path, teacher_hidden_dim, compressed_dim, num_stages, num_codes, device,
                   encoder_depth=3, decoder_depth=3, decay=0.99, is_emb=False):
    if is_emb:
        from train_RVQ_emb import HiddenStateCompressorEmb
        compressor = HiddenStateCompressorEmb(
            teacher_hidden_dim=teacher_hidden_dim,
            compressed_dim=compressed_dim,
            num_stages=num_stages,
            num_codes=num_codes,
            vocab_size=32000,
            decay=decay,
            kmeans_init=True,
            kmeans_iters=50,
            threshold_ema_dead_code=2,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
        )
    else:
        from train_RVQ import HiddenStateCompressor
        compressor = HiddenStateCompressor(
            teacher_hidden_dim=teacher_hidden_dim,
            compressed_dim=compressed_dim,
            num_stages=num_stages,
            num_codes=num_codes,
            vocab_size=32000,
            decay=decay,
            kmeans_init=True,
            kmeans_iters=50,
            threshold_ema_dead_code=2,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
        )
    
    state = torch.load(checkpoint_path, map_location="cpu")
    compressor.load_state_dict(state, strict=True)
    compressor.to(device)
    compressor.eval()
    return compressor


# ------------------------------------------------------------------------------
# Main Eval Loop
# ------------------------------------------------------------------------------

def run_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Teacher
    print(f"Loading teacher from {args.teacher_path} ...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    teacher.eval()
    teacher.to(device)

    if hasattr(teacher.config, "n_embd"):
        teacher_hidden_dim = teacher.config.n_embd
    elif hasattr(teacher.config, "hidden_size"):
        teacher_hidden_dim = teacher.config.hidden_size
    else:
        teacher_hidden_dim = 4096

    # Dataset
    dataset = EvalHiddenDataset(
        args.data_path,
        max_samples=args.max_samples,
        max_length=args.max_length,
    )
    def collate_fn(batch):
        return collate_hidden_and_y(batch, dataset, teacher, device, args.max_length)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Model
    # Try to parse parameters from filename if not provided
    filename = os.path.basename(args.checkpoint_path)
    parsed_params = parse_model_params_from_filename(filename)
    
    if args.model_type == "ae":
        model, input_dim = load_ae_model(args.checkpoint_path, teacher, args.latent_dim, device)
        use_y = True
        is_emb = False
    else:
        # Check if it's rvq_emb model (from filename or explicit flag)
        is_emb = parsed_params.get('is_emb', False) if parsed_params else False
        if hasattr(args, 'is_emb') and args.is_emb:
            is_emb = True
        
        # Use parsed params if available, otherwise use args
        num_stages = parsed_params.get('num_stages', args.num_stages) if parsed_params else args.num_stages
        num_codes = parsed_params.get('num_codes', args.num_codes) if parsed_params else args.num_codes
        compressed_dim = parsed_params.get('compressed_dim', args.compressed_dim) if parsed_params else args.compressed_dim
        encoder_depth = parsed_params.get('encoder_depth', getattr(args, 'encoder_depth', 3)) if parsed_params else getattr(args, 'encoder_depth', 3)
        decoder_depth = parsed_params.get('decoder_depth', getattr(args, 'decoder_depth', 3)) if parsed_params else getattr(args, 'decoder_depth', 3)
        decay = parsed_params.get('decay', getattr(args, 'decay', 0.99)) if parsed_params else getattr(args, 'decay', 0.99)
        
        if parsed_params:
            print(f"Parsed params from filename: stages={num_stages}, codes={num_codes}, dim={compressed_dim}, "
                  f"enc={encoder_depth}, dec={decoder_depth}, decay={decay}, is_emb={is_emb}", flush=True)
        
        model = load_rvq_model(
            args.checkpoint_path,
            teacher_hidden_dim=teacher_hidden_dim,
            compressed_dim=compressed_dim,
            num_stages=num_stages,
            num_codes=num_codes,
            device=device,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            decay=decay,
            is_emb=is_emb,
        )
        use_y = is_emb  # rvq_emb needs y_token (token embeddings)

    # Accumulators
    n_total = 0
    sum_logit_mse = 0.0
    sum_logit_kl = 0.0
    sum_hidden_mse = 0.0
    sum_hidden_cosine = 0.0

    model.eval()
    for batch in tqdm(loader, desc="Eval", leave=False, file=sys.stderr):
        if batch is None:
            continue
        hidden = batch["hidden"].float()
        teacher_logits_ref = batch.get("teacher_logits")
        y_token = batch.get("y_token")

        with torch.no_grad():
            if args.model_type == "ae":
                recon_hidden, _ = model(hidden, y_token=y_token)
            elif is_emb:
                # rvq_emb needs token embeddings (y_token -> embedding lookup)
                if y_token is None:
                    y_token = torch.zeros(hidden.size(0), dtype=torch.long, device=device)
                # Get token embeddings from teacher
                if hasattr(teacher, "transformer") and hasattr(teacher.transformer, "wte"):
                    token_emb = teacher.transformer.wte(y_token)
                elif hasattr(teacher, "model") and hasattr(teacher.model, "embed_tokens"):
                    token_emb = teacher.model.embed_tokens(y_token)
                else:
                    token_emb = teacher.get_input_embeddings()(y_token)
                recon_hidden, _ = model(hidden, token_embeddings=token_emb.float())
            else:
                recon_hidden, _ = model(hidden)

        n = hidden.size(0)
        m = compute_metrics(hidden, recon_hidden, teacher)
        n_total += n
        sum_logit_mse += m["logit_mse"] * n
        sum_logit_kl += m["logit_kl"] * n
        sum_hidden_mse += m["hidden_mse"] * n
        sum_hidden_cosine += m["hidden_cosine"] * n

    if n_total == 0:
        print("No samples evaluated.", flush=True)
        return

    # Calculate averages
    avg_logit_mse = sum_logit_mse / n_total
    avg_logit_kl = sum_logit_kl / n_total
    avg_hidden_mse = sum_hidden_mse / n_total
    avg_hidden_cosine = sum_hidden_cosine / n_total

    # Prepare results dict
    results = {
        "logit_mse": avg_logit_mse,
        "logit_kl": avg_logit_kl,
        "hidden_mse": avg_hidden_mse,
        "hidden_cosine": avg_hidden_cosine,
        "num_samples": n_total,
        "checkpoint_path": args.checkpoint_path,
        "model_type": args.model_type,
    }

    # Save results to file
    # Extract folder name (e.g., "logit_only") and model name from checkpoint path
    ckpt_basename = os.path.basename(args.checkpoint_path)
    model_name = os.path.splitext(ckpt_basename)[0]  # Remove .pt extension
    
    # Find folder name (e.g., "logit_only")
    # Example: .../checkpoints/AE/logit_only/layernorm/best_ae_ld25.pt
    # We want "logit_only" (the folder under checkpoints/AE/)
    path_parts = os.path.normpath(args.checkpoint_path).split(os.sep)
    folder_name = "unknown"
    if "checkpoints" in path_parts and "AE" in path_parts:
        ckpt_idx = path_parts.index("checkpoints")
        ae_idx = path_parts.index("AE", ckpt_idx)
        if ae_idx + 1 < len(path_parts):
            folder_name = path_parts[ae_idx + 1]  # checkpoints/AE/{folder_name}/...
    
    # Save to recon_results/{folder_name}/{model_name}.json
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))  # project root
    result_dir = os.path.join(base_path, "recon_results", folder_name)
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"{model_name}.json")
    
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # tqdm이 stdout을 덮어쓰지 않도록 새 줄 + stderr에 결과 출력
    sys.stderr.write("\n")
    sys.stderr.flush()
    print("\n" + "=" * 60, flush=True)
    print("Eval metrics (hidden -> logits 복원, hidden 복원)", flush=True)
    print("=" * 60, flush=True)
    print(f"  Logit MSE:        {avg_logit_mse:.6f}", flush=True)
    print(f"  Logit KL div:     {avg_logit_kl:.6f}", flush=True)
    print(f"  Hidden MSE:       {avg_hidden_mse:.6f}", flush=True)
    print(f"  Hidden Cosine:    {avg_hidden_cosine:.6f}", flush=True)
    print("=" * 60, flush=True)
    print(f"  # samples: {n_total}", flush=True)
    print(f"\n  Results saved to: {result_file}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Eval AE / RVQ: logit MSE/KL, hidden MSE/cosine")
    p.add_argument("--model_type", type=str, choices=["ae", "rvq"], required=True)
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--teacher_path", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)

    # AE
    p.add_argument("--latent_dim", type=int, default=25)

    # RVQ
    p.add_argument("--num_stages", type=int, default=4)
    p.add_argument("--num_codes", type=int, default=1024)
    p.add_argument("--compressed_dim", type=int, default=1024)
    p.add_argument("--encoder_depth", type=int, default=3)
    p.add_argument("--decoder_depth", type=int, default=3)

    args = p.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
