"""
1-stage KD: Feature matching between student hidden state and AE-reconstructed teacher hidden.

- Teacher last hidden → AE(teacher_hidden, y_token) → recon_hidden (target)
- Student hidden state is trained to match recon_hidden (MSE / cosine)
- Training uses only 5% of the dataset (configurable).
- Teacher and AE are frozen; only student (and optional projection) is trained.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
import sys
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from data_utils.indexed_dataset import MMapIndexedDataset
except ImportError:
    print("Warning: data_utils not found.")
    MMapIndexedDataset = None

# Import AE from train script
sys.path.insert(0, os.path.join(_script_dir, "..", "train"))
from train_ae_onthefly import ConditionalAutoEncoder


# ------------------------------------------------------------------------------
# Dataset: 5% of data (or configurable fraction)
# ------------------------------------------------------------------------------

class KDSequenceDataset(Dataset):
    """Sequence indices only. Actual loading + teacher inference in collate."""
    def __init__(self, data_path, max_samples=None, max_length=512, data_fraction=0.05):
        self.max_length = max_length
        if MMapIndexedDataset is None:
            raise RuntimeError("MMapIndexedDataset not available")
        self._raw = MMapIndexedDataset(data_path, skip_warmup=True)
        total = len(self._raw)
        # Use only data_fraction (e.g. 5%) of training data
        if max_samples is not None:
            cap = min(max_samples, total)
        else:
            cap = max(1, int(total * data_fraction))
        self.valid_indices = list(range(cap))
        print(f"KD Dataset: {len(self.valid_indices)} sequences (data_fraction={data_fraction:.2%}, total={total})")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        return self.valid_indices[idx]


def collate_kd_batch(batch_indices, dataset, teacher_model, tokenizer, device, max_length):
    """Build batch: input_ids, attention_mask for student; teacher hidden + y for AE target."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batch_tokens = []
    raw = dataset._raw
    for idx in batch_indices:
        tokens = raw[idx].astype(np.int64)[: max_length + 1]
        if len(tokens) > 1:
            batch_tokens.append(tokens)

    if not batch_tokens:
        return None

    B = len(batch_tokens)
    L = max(len(t) - 1 for t in batch_tokens)
    input_ids = np.full((B, L), pad_id, dtype=np.int64)
    attention_mask = np.zeros((B, L), dtype=np.int64)
    for i, t in enumerate(batch_tokens):
        n = len(t) - 1
        input_ids[i, :n] = t[:n]
        attention_mask[i, :n] = 1

    input_tensor = torch.tensor(input_ids, device=device, dtype=torch.long)
    mask_tensor = torch.tensor(attention_mask, device=device, dtype=torch.long)

    teacher_model.eval()
    with torch.no_grad():
        outputs = teacher_model(
            input_ids=input_tensor,
            attention_mask=mask_tensor,
            output_hidden_states=True,
            use_cache=False,
        )
    last_hidden = outputs.hidden_states[-1]   # [B, L, H_teacher]
    teacher_logits = outputs.logits
    y_token = teacher_logits.argmax(dim=-1)  # [B, L]

    m = mask_tensor.bool()
    teacher_hidden_flat = last_hidden[m]    # [N, H_teacher]
    y_token_flat = y_token[m]               # [N]

    return {
        "input_ids": input_tensor,
        "attention_mask": mask_tensor,
        "teacher_hidden": teacher_hidden_flat,
        "y_token": y_token_flat,
    }


# ------------------------------------------------------------------------------
# Projection (student_dim -> teacher_dim) when dimensions differ
# ------------------------------------------------------------------------------

class HiddenProjection(nn.Module):
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim)

    def forward(self, x):
        return self.proj(x)


# ------------------------------------------------------------------------------
# Training step and loop
# ------------------------------------------------------------------------------

def train_one_epoch(
    student_model,
    ae_model,
    teacher_model,
    train_loader,
    optimizer,
    accelerator,
    projection_module,
    loss_type="mse",
):
    student_model.train()
    ae_model.eval()
    teacher_model.eval()
    for p in ae_model.parameters():
        p.requires_grad = False
    for p in teacher_model.parameters():
        p.requires_grad = False

    total_loss = 0.0
    total_tokens = 0
    is_main = accelerator.is_main_process

    pbar = tqdm(train_loader, disable=not is_main, desc="1stage FM")
    for batch in pbar:
        if batch is None:
            continue
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        teacher_hidden = batch["teacher_hidden"].float()
        y_token = batch["y_token"]

        with torch.no_grad():
            recon_hidden, _ = ae_model(teacher_hidden, y_token=y_token)  # [N, H_teacher]
            recon_hidden = recon_hidden.to(torch.bfloat16)  # match student dtype for loss/backward

        student_outputs = student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        student_last = student_outputs.hidden_states[-1]  # [B, L, H_student]
        m = attention_mask.bool()
        student_hidden_flat = student_last[m]  # [N, H_student]

        if projection_module is not None:
            student_hidden_flat = projection_module(student_hidden_flat)  # [N, H_teacher]

        if loss_type == "mse":
            loss = F.mse_loss(student_hidden_flat, recon_hidden)
        else:
            loss = (1 - F.cosine_similarity(student_hidden_flat, recon_hidden, dim=-1).mean())

        n = student_hidden_flat.size(0)
        total_loss += loss.item() * n
        total_tokens += n

        optimizer.zero_grad()
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            torch.nn.utils.clip_grad_norm_(
                list(student_model.parameters()) + (list(projection_module.parameters()) if projection_module is not None else []),
                1.0,
            )
        optimizer.step()
        pbar.set_postfix({"loss": loss.item(), "avg": total_loss / max(1, total_tokens)})
    return total_loss / max(1, total_tokens)


def main():
    parser = argparse.ArgumentParser(description="1-stage KD: student hidden ~ AE(teacher hidden)")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument("--student_path", type=str, required=True)
    parser.add_argument("--ae_checkpoint", type=str, required=True, help="AE checkpoint (e.g. best_ae_ld25.pt)")
    parser.add_argument("--latent_dim", type=int, default=25)
    parser.add_argument("--data_fraction", type=float, default=0.05, help="Use 5%% of data")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap total samples (overrides data_fraction if set)")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output_dir", type=str, default="./ckpt_1stage_fm")
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "cosine"])
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--from_scratch", action="store_true", help="Initialize student from config only (pretraining; no pretrained weights)")
    args = parser.parse_args()

    accelerator = Accelerator()
    set_seed(42 + accelerator.process_index)

    device = accelerator.device
    tokenizer_path = args.tokenizer_path or args.teacher_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Teacher (frozen)
    if accelerator.is_main_process:
        print("Loading teacher...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_path,
        torch_dtype=torch.bfloat16,
    )
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model.to(device)

    # Teacher hidden dim & embed for AE
    if hasattr(teacher_model.config, "n_embd"):
        teacher_hidden_dim = teacher_model.config.n_embd
    elif hasattr(teacher_model.config, "hidden_size"):
        teacher_hidden_dim = teacher_model.config.hidden_size
    else:
        teacher_hidden_dim = 4096
    if hasattr(teacher_model, "transformer") and hasattr(teacher_model.transformer, "wte"):
        teacher_embed = teacher_model.transformer.wte
    else:
        teacher_embed = teacher_model.get_input_embeddings()

    # AE (frozen, from checkpoint)
    if accelerator.is_main_process:
        print("Loading AE...")
    ae_model = ConditionalAutoEncoder(
        input_dim=teacher_hidden_dim,
        latent_dim=args.latent_dim,
        teacher_embed=teacher_embed,
    )
    ckpt = torch.load(args.ae_checkpoint, map_location="cpu")
    ae_model.load_state_dict(ckpt, strict=False)
    ae_model.to(device)
    ae_model.eval()
    for p in ae_model.parameters():
        p.requires_grad = False

    # Student (trainable): from scratch (config only) or from checkpoint
    if accelerator.is_main_process:
        print("Loading student (from_scratch={})...".format(args.from_scratch))
    if args.from_scratch:
        config = AutoConfig.from_pretrained(args.student_path)
        student_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    else:
        student_model = AutoModelForCausalLM.from_pretrained(
            args.student_path,
            torch_dtype=torch.bfloat16,
        )
    if hasattr(student_model.config, "n_embd"):
        student_hidden_dim = student_model.config.n_embd
    elif hasattr(student_model.config, "hidden_size"):
        student_hidden_dim = student_model.config.hidden_size
    else:
        student_hidden_dim = 768
    student_model.to(device)

    projection_module = None
    if student_hidden_dim != teacher_hidden_dim:
        projection_module = HiddenProjection(student_hidden_dim, teacher_hidden_dim).to(device=device, dtype=torch.bfloat16)
        if accelerator.is_main_process:
            print(f"Using projection: {student_hidden_dim} -> {teacher_hidden_dim}")

    # Dataset: 5% (or max_samples)
    dataset = KDSequenceDataset(
        args.data_path,
        max_samples=args.max_samples,
        max_length=args.max_length,
        data_fraction=args.data_fraction,
    )

    def collate_fn(batch):
        return collate_kd_batch(
            batch, dataset, teacher_model, tokenizer, device, args.max_length
        )

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    params = list(student_model.parameters())
    if projection_module is not None:
        params += list(projection_module.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    to_prepare = [student_model, optimizer, train_loader]
    if projection_module is not None:
        to_prepare.append(projection_module)
    prepared = accelerator.prepare(*to_prepare)
    student_model = prepared[0]
    optimizer = prepared[1]
    train_loader = prepared[2]
    if projection_module is not None:
        projection_module = prepared[3]

    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(
            student_model, ae_model, teacher_model,
            train_loader, optimizer, accelerator,
            projection_module, args.loss_type,
        )
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} avg loss: {avg_loss:.6f}")
            unwrapped = accelerator.unwrap_model(student_model)
            unwrapped.save_pretrained(os.path.join(args.output_dir, f"epoch_{epoch+1}"))
            if projection_module is not None:
                torch.save(projection_module.state_dict(), os.path.join(args.output_dir, f"proj_epoch_{epoch+1}.pt"))

    if accelerator.is_main_process:
        print("1-stage feature matching done. Saved to", args.output_dir)


if __name__ == "__main__":
    main()
