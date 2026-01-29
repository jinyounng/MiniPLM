"""
Save AE latent (z) for 5% of data — same forwarding as 1stage feature matching.

- Same dataset (5%), same collate: teacher forward → teacher_hidden, y_token
- AE(teacher_hidden, y_token) → take z only (latent), save to disk
- Saves per-batch chunks: latent_z and y_token, plus meta.json
"""

import os
import argparse
import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from data_utils.indexed_dataset import MMapIndexedDataset
except ImportError:
    MMapIndexedDataset = None

sys.path.insert(0, os.path.join(_script_dir, "..", "train"))
from train_ae_onthefly import ConditionalAutoEncoder


# ------------------------------------------------------------------------------
# Same dataset & collate as 1stage (teacher forward only)
# ------------------------------------------------------------------------------

class KDSequenceDataset(Dataset):
    def __init__(self, data_path, max_samples=None, max_length=512, data_fraction=0.05):
        self.max_length = max_length
        if MMapIndexedDataset is None:
            raise RuntimeError("MMapIndexedDataset not available")
        self._raw = MMapIndexedDataset(data_path, skip_warmup=True)
        total = len(self._raw)
        if max_samples is not None:
            cap = min(max_samples, total)
        else:
            cap = max(1, int(total * data_fraction))
        self.valid_indices = list(range(cap))
        print(f"Save latent dataset: {len(self.valid_indices)} sequences (data_fraction={data_fraction:.2%}, total={total})")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        return self.valid_indices[idx]


def collate_kd_batch(batch_indices, dataset, teacher_model, tokenizer, device, max_length):
    """Same as 1stage: teacher forward → teacher_hidden_flat, y_token_flat."""
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
    last_hidden = outputs.hidden_states[-1]
    teacher_logits = outputs.logits
    y_token = teacher_logits.argmax(dim=-1)

    m = mask_tensor.bool()
    teacher_hidden_flat = last_hidden[m]
    y_token_flat = y_token[m]

    return {
        "teacher_hidden": teacher_hidden_flat,
        "y_token": y_token_flat,
    }


def main():
    parser = argparse.ArgumentParser(description="Save AE latent (z) for 5% data — same forward as 1stage")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument("--ae_checkpoint", type=str, required=True)
    parser.add_argument("--latent_dim", type=int, default=40)
    parser.add_argument("--data_fraction", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save latent chunks and meta")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_path = args.tokenizer_path or args.teacher_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Teacher (frozen)
    print("Loading teacher...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_path,
        torch_dtype=torch.bfloat16,
    )
    teacher_model.eval()
    teacher_model.to(device)

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

    # AE (frozen)
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

    # Dataset & loader — same as 1stage
    dataset = KDSequenceDataset(
        args.data_path,
        max_samples=args.max_samples,
        max_length=args.max_length,
        data_fraction=args.data_fraction,
    )

    def collate_fn(batch):
        return collate_kd_batch(batch, dataset, teacher_model, tokenizer, device, args.max_length)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    total_tokens = 0
    chunk_list = []

    print("Saving AE latent (same forward as 1stage)...")
    for chunk_idx, batch in enumerate(tqdm(loader, desc="Save latent")):
        if batch is None:
            continue
        teacher_hidden = batch["teacher_hidden"].float()
        y_token = batch["y_token"]

        with torch.no_grad():
            _, z = ae_model(teacher_hidden, y_token=y_token)  # [N, latent_dim]

        z_np = z.cpu().numpy().astype(np.float32)
        y_np = y_token.cpu().numpy()

        chunk_path = os.path.join(args.save_dir, f"latent_chunk_{chunk_idx:06d}.npz")
        np.savez_compressed(chunk_path, z=z_np, y_token=y_np)
        chunk_list.append({"path": os.path.basename(chunk_path), "n": int(z_np.shape[0])})
        total_tokens += z_np.shape[0]

    meta = {
        "num_chunks": len(chunk_list),
        "latent_dim": args.latent_dim,
        "total_tokens": total_tokens,
        "data_fraction": args.data_fraction,
        "max_length": args.max_length,
        "chunks": chunk_list,
    }
    with open(os.path.join(args.save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Saved {total_tokens} tokens in {len(chunk_list)} chunks under {args.save_dir}")
    print(f"  meta.json: num_chunks={meta['num_chunks']}, latent_dim={meta['latent_dim']}, total_tokens={meta['total_tokens']}")


if __name__ == "__main__":
    main()
