"""
1-stage KD from pre-saved latent: Teacher forward 없이 저장된 z, y_token만 사용.

- 저장된 latent (results/AE/latent_5pct/rank_*/latent_chunk_*.npz) 로드
- input_ids는 동일 dataset/rank/world_size/batch_size로 생성 (save 시와 동일 순서)
- AE decoder만 사용: z, y_token → recon_hidden (target)
- Student forward → student_hidden, loss = mse(student_hidden, recon_hidden)

사용 조건: save_ae_latent_5pct.py와 동일한 data_path, data_fraction, max_length, batch_size, rank/world_size 사용.
"""

import os
import argparse
import json
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
import sys
from accelerate import Accelerator
from accelerate.utils import set_seed

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
# Dataset: save_ae_latent_5pct와 동일 (rank/world_size sharding)
# ------------------------------------------------------------------------------

class KDSequenceDataset(Dataset):
    """save_ae_latent_5pct와 동일: rank/world_size로 shard."""
    def __init__(self, data_path, max_samples=None, max_length=512, data_fraction=0.05, rank=0, world_size=1):
        self.max_length = max_length
        if MMapIndexedDataset is None:
            raise RuntimeError("MMapIndexedDataset not available")
        self._raw = MMapIndexedDataset(data_path, skip_warmup=True)
        total = len(self._raw)
        if max_samples is not None:
            cap = min(max_samples, total)
        else:
            cap = max(1, int(total * data_fraction))
        full_indices = list(range(cap))
        self.valid_indices = full_indices[rank::world_size]
        if world_size > 1:
            print(f"From-latent dataset rank {rank}/{world_size}: {len(self.valid_indices)} sequences")
        else:
            print(f"From-latent dataset: {len(self.valid_indices)} sequences (data_fraction={data_fraction:.2%})")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        return self.valid_indices[idx]


def collate_input_only(batch_indices, dataset, tokenizer, device, max_length):
    """Teacher 없이 input_ids, attention_mask만 생성 (save 시와 동일 padding)."""
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
    # Create tensors on CPU first, then move to device to avoid CUDA peer memory access errors
    # Use non_blocking=False to ensure proper synchronization
    input_tensor = torch.from_numpy(input_ids).to(device, non_blocking=False)
    mask_tensor = torch.from_numpy(attention_mask).to(device, non_blocking=False)
    return {"input_ids": input_tensor, "attention_mask": mask_tensor}


def ae_decode(ae_model, z, y_token):
    """z, y_token → recon_hidden (decoder만 사용)."""
    cond = ae_model._embed_y(y_token).float()
    z_f = z.float()
    dec_input = torch.cat([z_f, cond], dim=-1)
    return ae_model.decoder(dec_input)


# ------------------------------------------------------------------------------
# Projection (1stage와 동일)
# ------------------------------------------------------------------------------

class HiddenProjection(nn.Module):
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim)

    def forward(self, x):
        return self.proj(x)


# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------

def train_one_epoch(
    student_model,
    ae_model,
    train_loader,
    chunk_paths,
    optimizer,
    accelerator,
    projection_module,
    loss_type,
    device,
    max_length,
    tokenizer,
    dataset,
):
    student_model.train()
    ae_model.eval()
    for p in ae_model.parameters():
        p.requires_grad = False

    total_loss = 0.0
    total_tokens = 0
    is_main = accelerator.is_main_process

    # chunk를 여러 배치로 나눠서 사용할 수 있도록 처리
    # chunk_idx: 현재 사용 중인 chunk 인덱스
    # chunk_offset: 현재 chunk 내에서 사용한 토큰 수
    chunk_idx = 0
    chunk_offset = 0
    current_chunk = None
    current_chunk_z = None
    current_chunk_y = None

    for batch_idx, batch_indices in enumerate(tqdm(train_loader, disable=not is_main, desc="1stage from latent")):
        if chunk_idx >= len(chunk_paths):
            break
        # input_ids, attention_mask (teacher 없음)
        batch = collate_input_only(batch_indices, dataset, tokenizer, device, max_length)
        if batch is None:
            continue
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        # 현재 배치의 토큰 수 계산
        num_tokens = attention_mask.sum().item()

        # 필요한 chunk 로드 또는 재사용
        while current_chunk is None or chunk_offset + num_tokens > len(current_chunk_z):
            if current_chunk is not None and chunk_offset < len(current_chunk_z):
                # 현재 chunk에 남은 데이터가 있지만 부족한 경우, 다음 chunk로
                chunk_idx += 1
                chunk_offset = 0
            if chunk_idx >= len(chunk_paths):
                break
            # 새 chunk 로드
            current_chunk = np.load(chunk_paths[chunk_idx])
            current_chunk_z = current_chunk["z"]  # (N, latent_dim) or (N,)
            current_chunk_y = current_chunk["y_token"]  # (N,)
            if len(current_chunk_z.shape) == 1:
                current_chunk_z = current_chunk_z[:, None]  # (N, 1)로 변환

        if chunk_idx >= len(chunk_paths):
            break

        # 현재 배치에 해당하는 latent 추출
        z_np = current_chunk_z[chunk_offset:chunk_offset + num_tokens]
        y_np = current_chunk_y[chunk_offset:chunk_offset + num_tokens]
        chunk_offset += num_tokens
        
        # z_np shape 처리: (N,) -> (N, 1) 또는 (N, latent_dim) 유지
        if len(z_np.shape) == 1:
            z_np = z_np[:, None]

        # Create tensors on CPU first, then move to device to avoid CUDA peer memory access errors
        # Use non_blocking=False to ensure proper synchronization
        z = torch.from_numpy(z_np).to(device, dtype=torch.float32, non_blocking=False)
        y_token = torch.from_numpy(y_np).to(device, dtype=torch.long, non_blocking=False)

        # AE decoder: z, y_token → recon_hidden
        with torch.no_grad():
            recon_hidden = ae_decode(ae_model, z, y_token)  # [N, H_teacher]

        # Student forward
        student_outputs = student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        student_last = student_outputs.hidden_states[-1]
        m = attention_mask.bool()
        student_hidden_flat = student_last[m]

        if projection_module is not None:
            student_hidden_flat = projection_module(student_hidden_flat)

        student_f32 = student_hidden_flat.float()
        recon_f32 = recon_hidden.float()
        if loss_type == "mse":
            loss = F.mse_loss(student_f32, recon_f32)
        else:
            loss = (1 - F.cosine_similarity(student_f32, recon_f32, dim=-1)).mean()

        n = student_hidden_flat.size(0)
        total_loss += loss.item() * n
        total_tokens += n

        optimizer.zero_grad()
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            params = list(student_model.parameters())
            if projection_module is not None:
                params += list(projection_module.parameters())
            torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

    return total_loss / max(1, total_tokens)


def main():
    parser = argparse.ArgumentParser(description="1-stage KD from pre-saved latent (decoder only)")
    parser.add_argument("--latent_dir", type=str, required=True,
                        help="저장된 latent 디렉토리 (예: results/AE/latent_5pct)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="save 시와 동일한 data_path")
    parser.add_argument("--teacher_path", type=str, required=True,
                        help="AE의 y_embed용 (teacher embedding만 사용, forward 안 함)")
    parser.add_argument("--student_path", type=str, required=True)
    parser.add_argument("--ae_checkpoint", type=str, required=True)
    parser.add_argument("--latent_dim", type=int, default=40)
    parser.add_argument("--data_fraction", type=float, default=0.05,
                        help="save 시와 동일해야 함")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="save 시와 동일해야 함")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--output_dir", type=str, default="./ckpt_1stage_from_latent")
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "cosine"])
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--from_scratch", action="store_true")
    args = parser.parse_args()

    accelerator = Accelerator()
    set_seed(42 + accelerator.process_index)
    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    
    # Ensure each process uses the correct CUDA device
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    # meta.json에서 latent_dim 등 확인
    meta_path = os.path.join(args.latent_dir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        args.latent_dim = meta.get("latent_dim", args.latent_dim)
        if accelerator.is_main_process:
            print(f"Loaded meta: latent_dim={args.latent_dim}, total_tokens={meta.get('total_tokens')}")

    tokenizer_path = args.tokenizer_path or args.teacher_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Teacher: embedding만 AE에 넘기기 위해 로드 (forward는 안 함)
    if accelerator.is_main_process:
        print("Loading teacher (embedding only for AE)...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_path,
        torch_dtype=torch.bfloat16,
    )
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
    del teacher_model  # forward 안 하므로 메모리 해제

    # AE (decoder만 사용)
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

    # Student
    if accelerator.is_main_process:
        print("Loading student...")
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

    # Dataset: save와 동일 (rank, world_size)
    dataset = KDSequenceDataset(
        args.data_path,
        max_samples=None,
        max_length=args.max_length,
        data_fraction=args.data_fraction,
        rank=rank,
        world_size=world_size,
    )

    # Chunk 경로 (이 rank 것만, 정렬)
    rank_dir = os.path.join(args.latent_dir, f"rank_{rank}")
    if not os.path.isdir(rank_dir):
        rank_dir = args.latent_dir  # single GPU일 때 rank_0 없을 수 있음
    chunk_paths = sorted(glob.glob(os.path.join(rank_dir, "latent_chunk_*.npz")))
    if not chunk_paths:
        raise FileNotFoundError(f"No latent chunks in {rank_dir}")

    # DataLoader: indices만 반환, collate에서 input_ids 생성 (teacher 없음)
    def collate_fn(batch):
        return batch  # list of indices

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # save 시와 동일 순서 유지
        num_workers=0,
        collate_fn=collate_fn,
    )
    # dataloader는 prepare하지 않음 (이미 rank별 shard 되어 있음)
    train_loader = train_loader

    params = list(student_model.parameters())
    if projection_module is not None:
        params += list(projection_module.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    to_prepare = [student_model, optimizer]
    if projection_module is not None:
        to_prepare.append(projection_module)
    prepared = accelerator.prepare(*to_prepare)
    student_model = prepared[0]
    optimizer = prepared[1]
    if projection_module is not None:
        projection_module = prepared[2]

    if accelerator.is_main_process:
        print(f"Batches (this rank): {len(train_loader)}, Chunks: {len(chunk_paths)}")

    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(
            student_model, ae_model, train_loader, chunk_paths,
            optimizer, accelerator, projection_module, args.loss_type,
            device, args.max_length, tokenizer, dataset,
        )
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} avg loss: {avg_loss:.6f}")
            unwrapped = accelerator.unwrap_model(student_model)
            unwrapped.save_pretrained(os.path.join(args.output_dir, f"epoch_{epoch+1}"))
            if projection_module is not None:
                torch.save(projection_module.state_dict(), os.path.join(args.output_dir, f"proj_epoch_{epoch+1}.pt"))

    if accelerator.is_main_process:
        print("1-stage from latent done. Saved to", args.output_dir)


if __name__ == "__main__":
    main()
