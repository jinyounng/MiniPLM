"""
Teacher Logit K Analysis (Distributed)
======================================

Teacher의 각 토큰 스텝에서 softmax 확률을 정렬하여,
누적확률 임계값(0.99, 0.999)을 달성하는 최소 K를 분석합니다.

고정 K Coverage Rate (추가):
  K ∈ {1000, 2000, 5000, 10000}에 대해, 각 step에서 top-K 누적확률이 0.99 이상인 비율을 구함.
  • compute_coverage_rate_stats(logits, tau=1.0): (T, V) logits 입력 시 단일 JSON 요약 반환.
  • run_analysis() 실행 시 coverage_rate_0.99.json 자동 저장 (total_steps, tau, per-K: coverage_rate_0.99, mean_mass, p1/p5/p10_mass, fail_rate).

사용법:
    # Multi-GPU 실행 (8 GPUs)
    torchrun --nproc_per_node=8 logit_k_analysis.py \
        --model-path /path/to/teacher/model \
        --data-dir /path/to/bin/data \
        --output-dir ./analysis_results \
        --max-samples 100000 \
        --batch-size 16 \
        --num-workers 4 \
        --distributed

    # Single GPU 실행
    python logit_k_analysis.py \
        --model-path /path/to/teacher/model \
        --data-dir /path/to/bin/data \
        --output-dir ./analysis_results \
        --max-samples 10000 \
        --batch-size 8
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================================
# Distributed Utils
# ============================================================================

def is_main_process():
    """Check if current process is main (rank 0)"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank():
    """Get current process rank"""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """Get total number of processes"""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        return True
    return False


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def print_rank0(msg):
    """Print only from rank 0"""
    if is_main_process():
        print(msg)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class StepStats:
    """각 토큰 스텝의 통계"""
    k_99: int           # 누적확률 >= 0.99가 되는 최소 K
    k_999: int          # 누적확률 >= 0.999가 되는 최소 K
    top1_prob: float    # Top-1 확률
    entropy: float      # Shannon Entropy H(p)
    

@dataclass
class AggregateStats:
    """전체 데이터에 대한 집계 통계"""
    total_steps: int
    
    # K_0.99 분포
    k99_mean: float
    k99_median: float
    k99_std: float
    k99_min: int
    k99_max: int
    k99_percentiles: Dict[int, float]
    
    # K_0.999 분포
    k999_mean: float
    k999_median: float
    k999_std: float
    k999_min: int
    k999_max: int
    k999_percentiles: Dict[int, float]
    
    # Top-1 확률 분포
    top1_mean: float
    top1_median: float
    top1_std: float
    
    # Entropy 분포
    entropy_mean: float
    entropy_median: float
    entropy_std: float
    
    # 요약 문장용 지표
    k99_at_95pct: int
    k999_at_95pct: int
    k999_at_99pct: int


# ============================================================================
# Core Analysis Functions
# ============================================================================

# 고정 K에 대한 Coverage Rate 분석용 상수
COVERAGE_K_VALUES = (1000, 2000, 5000, 10000)
COVERAGE_THRESHOLD = 0.99


def compute_k_for_threshold(probs_sorted: np.ndarray, threshold: float) -> int:
    """내림차순 정렬된 확률에서 누적합이 threshold 이상이 되는 최소 K를 찾습니다."""
    cumsum = np.cumsum(probs_sorted)
    k_idx = np.searchsorted(cumsum, threshold, side='left')
    return int(k_idx + 1)


def compute_entropy(probs: np.ndarray, eps: float = 1e-10) -> float:
    """Shannon Entropy 계산: H(p) = -sum(p * log(p))"""
    probs_safe = np.clip(probs, eps, 1.0)
    return float(-np.sum(probs * np.log(probs_safe)))


# ============================================================================
# Fixed-K Coverage Rate Analysis
# ============================================================================
#
# CoverageRate(K) = (1/T) * sum_t I_t(K), where I_t(K) = 1 if sum of top-K probs >= 0.99 else 0.
# 입력: teacher step별 logits z_t (T, V), 옵션 temperature tau (기본 1).
# 출력: total_steps, tau, per-K: coverage_rate_0.99, mean_mass, p1/p5/p10_mass, fail_rate.
#


def compute_coverage_rate_stats(
    logits: np.ndarray,
    tau: float = 1.0,
    k_values: Tuple[int, ...] = COVERAGE_K_VALUES,
    threshold: float = COVERAGE_THRESHOLD,
) -> Dict:
    """
    고정 K들에 대해 Coverage Rate 및 보조 통계를 계산합니다.

    입력:
        logits: (T, V) step별 teacher logits (full vocab).
        tau: temperature (기본 1).
        k_values: 고정 K 리스트 (기본 (1000, 2000, 5000, 10000)).
        threshold: 누적확률 기준 (기본 0.99).

    반환:
        단일 JSON 요약 dict: total_steps, tau, per-K coverage_rate_0.99, mean_mass,
        p1_mass, p5_mass, p10_mass, fail_rate.
    """
    if logits.ndim != 2:
        raise ValueError("logits must be (T, V)")
    T, V = logits.shape

    # p_t = softmax(z_t / tau)
    if tau != 1.0:
        logits = logits / tau
    logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # 내림차순 정렬 후 누적합: S_t(K) = sum of top-K probs
    probs_sorted = np.sort(probs, axis=-1)[:, ::-1]  # (T, V)
    cumsum = np.cumsum(probs_sorted, axis=-1)  # (T, V)

    out = {
        "total_steps": int(T),
        "tau": float(tau),
        "threshold": float(threshold),
        "k_values": list(k_values),
        "per_k": {},
    }

    for K in k_values:
        if K > V:
            # vocab보다 큰 K는 상한으로
            k_idx = V - 1
        else:
            k_idx = K - 1  # 0-indexed: cumsum[:, k_idx] = sum of first (k_idx+1) = top K
        S_t = np.asarray(cumsum[:, k_idx], dtype=np.float64).ravel()  # (T,)
        I_t = (S_t >= threshold).astype(np.float64)

        coverage_rate = float(np.mean(I_t))
        mean_mass = float(np.mean(S_t))
        p1 = float(np.percentile(S_t, 1))
        p5 = float(np.percentile(S_t, 5))
        p10 = float(np.percentile(S_t, 10))
        fail_rate = 1.0 - coverage_rate

        out["per_k"][str(K)] = {
            "coverage_rate_0.99": coverage_rate,
            "mean_mass": mean_mass,
            "p1_mass": p1,
            "p5_mass": p5,
            "p10_mass": p10,
            "fail_rate": fail_rate,
        }

    return out


def aggregate_coverage_from_arrays(
    mass_by_k: Dict[int, np.ndarray],
    tau: float = 1.0,
    threshold: float = COVERAGE_THRESHOLD,
) -> Dict:
    """
    이미 수집된 per-step top-K 누적질량 배열들로부터 Coverage Rate 요약을 만듭니다.

    mass_by_k: {K: array of shape (T,) with S_t(K) for each step t}
    """
    total_steps = None
    per_k = {}
    for K, S_t in mass_by_k.items():
        S_t = np.asarray(S_t, dtype=np.float64).ravel()
        if total_steps is None:
            total_steps = len(S_t)
        I_t = (S_t >= threshold).astype(np.float64)
        coverage_rate = float(np.mean(I_t))
        mean_mass = float(np.mean(S_t))
        p1 = float(np.percentile(S_t, 1))
        p5 = float(np.percentile(S_t, 5))
        p10 = float(np.percentile(S_t, 10))
        per_k[str(K)] = {
            "coverage_rate_0.99": coverage_rate,
            "mean_mass": mean_mass,
            "p1_mass": p1,
            "p5_mass": p5,
            "p10_mass": p10,
            "fail_rate": 1.0 - coverage_rate,
        }
    return {
        "total_steps": total_steps or 0,
        "tau": float(tau),
        "threshold": float(threshold),
        "k_values": list(mass_by_k.keys()),
        "per_k": per_k,
    }


def analyze_single_step(logits: np.ndarray, temperature: float = 1.0) -> StepStats:
    """단일 토큰 스텝의 logits를 분석합니다."""
    if temperature != 1.0:
        logits = logits / temperature
    
    logits_shifted = logits - np.max(logits)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / np.sum(exp_logits)
    
    probs_sorted = np.sort(probs)[::-1]
    
    k_99 = compute_k_for_threshold(probs_sorted, 0.99)
    k_999 = compute_k_for_threshold(probs_sorted, 0.999)
    top1_prob = float(probs_sorted[0])
    entropy = compute_entropy(probs)
    
    return StepStats(k_99=k_99, k_999=k_999, top1_prob=top1_prob, entropy=entropy)


def aggregate_stats(step_stats: List[StepStats]) -> AggregateStats:
    """개별 스텝 통계를 집계합니다."""
    k99_values = np.array([s.k_99 for s in step_stats])
    k999_values = np.array([s.k_999 for s in step_stats])
    top1_values = np.array([s.top1_prob for s in step_stats])
    entropy_values = np.array([s.entropy for s in step_stats])
    
    percentile_keys = [50, 90, 95, 99]
    
    return AggregateStats(
        total_steps=len(step_stats),
        k99_mean=float(np.mean(k99_values)),
        k99_median=float(np.median(k99_values)),
        k99_std=float(np.std(k99_values)),
        k99_min=int(np.min(k99_values)),
        k99_max=int(np.max(k99_values)),
        k99_percentiles={p: float(np.percentile(k99_values, p)) for p in percentile_keys},
        k999_mean=float(np.mean(k999_values)),
        k999_median=float(np.median(k999_values)),
        k999_std=float(np.std(k999_values)),
        k999_min=int(np.min(k999_values)),
        k999_max=int(np.max(k999_values)),
        k999_percentiles={p: float(np.percentile(k999_values, p)) for p in percentile_keys},
        top1_mean=float(np.mean(top1_values)),
        top1_median=float(np.median(top1_values)),
        top1_std=float(np.std(top1_values)),
        entropy_mean=float(np.mean(entropy_values)),
        entropy_median=float(np.median(entropy_values)),
        entropy_std=float(np.std(entropy_values)),
        k99_at_95pct=int(np.percentile(k99_values, 95)),
        k999_at_95pct=int(np.percentile(k999_values, 95)),
        k999_at_99pct=int(np.percentile(k999_values, 99)),
    )


# ============================================================================
# Parallel Data Loading
# ============================================================================

class BinDataset(Dataset):
    """PyTorch Dataset wrapper for MMapIndexedDataset"""
    def __init__(self, data_dir: str, max_samples: Optional[int] = None):
        from data_utils.indexed_dataset import MMapIndexedDataset
        from glob import glob
        
        idx_files = sorted(glob(os.path.join(data_dir, "data_*.idx")))
        
        if not idx_files:
            raise FileNotFoundError(f"No .idx files found in {data_dir}")
        
        self.datasets = []
        self.cumulative_sizes = [0]
        total_size = 0
        
        for idx_file in idx_files:
            prefix = idx_file.replace('.idx', '')
            dataset = MMapIndexedDataset(prefix, skip_warmup=True)
            self.datasets.append(dataset)
            total_size += len(dataset)
            self.cumulative_sizes.append(total_size)
            
            if max_samples is not None and total_size >= max_samples:
                break
        
        self.total_size = min(total_size, max_samples) if max_samples else total_size
        print_rank0(f"Loaded {len(self.datasets)} shards, total {self.total_size} samples")
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        if idx >= self.total_size:
            raise IndexError(f"Index {idx} out of range")
        
        for i, (start, end) in enumerate(zip(self.cumulative_sizes[:-1], self.cumulative_sizes[1:])):
            if start <= idx < end:
                local_idx = idx - start
                data = self.datasets[i][local_idx]
                return torch.from_numpy(np.array(data, dtype=np.int64))
        
        raise IndexError(f"Index {idx} out of range")


def collate_fn(batch, max_length=1024):
    """Dynamic padding collate function"""
    batch = [b for b in batch if b is not None and len(b) > 0]
    
    if not batch:
        return None, None
    
    batch = [b[:max_length] for b in batch]
    max_len = max(len(b) for b in batch)
    
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i, tokens in enumerate(batch):
        seq_len = len(tokens)
        input_ids[i, :seq_len] = tokens
        attention_mask[i, :seq_len] = 1
    
    return input_ids, attention_mask


def create_collate_fn(max_length: int):
    """Create collate function with specified max_length"""
    def _collate(batch):
        return collate_fn(batch, max_length)
    return _collate


# ============================================================================
# Model Loading & Inference
# ============================================================================

def load_model_and_tokenizer(model_path: str, device: str = "cuda", dtype: str = "bf16", distributed: bool = False):
    """Teacher 모델과 토크나이저를 로드합니다."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print_rank0(f"Loading model from {model_path}...")
    
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16 if dtype == "fp16" else torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if distributed:
        # 분산 환경에서는 각 GPU에 모델 로드
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map={'': local_rank},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    else:
        # Single GPU에서는 device_map="auto" 사용
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    
    model.eval()
    
    print_rank0(f"Model loaded. Vocab size: {model.config.vocab_size}")
    return model, tokenizer


def process_batch_logits(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float = 1.0,
    use_amp: bool = True,
    return_coverage: bool = False,
    coverage_k_values: Tuple[int, ...] = COVERAGE_K_VALUES,
):
    """배치의 logits를 벡터화하여 한 번에 분석 (진짜 배치 처리).

    return_coverage=True이면 (step_stats, mass_by_k) 반환.
    mass_by_k: {K: np.ndarray of S_t(K) for each valid step in this batch}.
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with torch.no_grad():
        if use_amp and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device)
                )
                logits = outputs.logits.float()
        else:
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device)
            )
            logits = outputs.logits.float()
        
        # GPU에서 바로 처리
        batch_size, seq_len, vocab_size = logits.shape
        
        # Temperature scaling (GPU에서)
        if temperature != 1.0:
            logits = logits / temperature
        
        # Softmax (GPU에서, 메모리 효율적으로)
        probs = torch.softmax(logits, dim=-1)  # [batch, seq, vocab]
        
        # Entropy 계산 (GPU에서)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # [batch, seq]
        
        # Top-1 확률 추출
        top1_probs, _ = probs.max(dim=-1)  # [batch, seq]
        
        # 내림차순 정렬 (GPU에서)
        probs_sorted, _ = torch.sort(probs, dim=-1, descending=True)  # [batch, seq, vocab]
        
        # 누적 합 계산 (GPU에서)
        cumsum_probs = torch.cumsum(probs_sorted, dim=-1)  # [batch, seq, vocab]
        
        # K_0.99, K_0.999 계산 (GPU에서)
        k99 = (cumsum_probs >= 0.99).int().argmax(dim=-1) + 1  # [batch, seq]
        k999 = (cumsum_probs >= 0.999).int().argmax(dim=-1) + 1  # [batch, seq]
        
        # 고정 K 누적질량 (Coverage 분석용): S_t(K) = cumsum_probs[:, :, k_idx]
        mass_by_k_batch: Optional[Dict[int, np.ndarray]] = None
        if return_coverage:
            mass_by_k_batch = {}
            for K in coverage_k_values:
                k_idx = min(K - 1, vocab_size - 1)
                S_t = cumsum_probs[:, :, k_idx].cpu().numpy()  # [batch, seq]
                mass_by_k_batch[K] = S_t
        
        # CPU로 이동 (한 번만)
        attention_mask_cpu = attention_mask.cpu()
        k99_cpu = k99.cpu().numpy()
        k999_cpu = k999.cpu().numpy()
        top1_probs_cpu = top1_probs.cpu().numpy()
        entropy_cpu = entropy.cpu().numpy()
        
        del outputs, logits, probs, log_probs, probs_sorted, cumsum_probs, k99, k999, top1_probs, entropy
        torch.cuda.empty_cache()
    
    # 결과를 StepStats 리스트로 변환 (CPU에서); coverage는 valid step만 수집
    step_stats = []
    mass_by_k_valid: Optional[Dict[int, List[float]]] = {}
    if return_coverage and mass_by_k_batch is not None:
        for K in coverage_k_values:
            mass_by_k_valid[K] = []
    for b in range(batch_size):
        valid_len = int(attention_mask_cpu[b].sum())
        for t in range(valid_len):
            step_stats.append(StepStats(
                k_99=int(k99_cpu[b, t]),
                k_999=int(k999_cpu[b, t]),
                top1_prob=float(top1_probs_cpu[b, t]),
                entropy=float(entropy_cpu[b, t])
            ))
            if return_coverage and mass_by_k_batch is not None:
                for K in coverage_k_values:
                    mass_by_k_valid[K].append(float(mass_by_k_batch[K][b, t]))
    
    if return_coverage and mass_by_k_valid is not None:
        mass_by_k_out = {K: np.array(mass_by_k_valid[K], dtype=np.float64) for K in coverage_k_values}
        return step_stats, mass_by_k_out
    return step_stats


# ============================================================================
# Distributed Gather
# ============================================================================

def gather_stats_distributed(local_stats: List[StepStats]) -> List[StepStats]:
    """Gather stats from all ranks to rank 0"""
    if not dist.is_initialized():
        return local_stats
    
    world_size = get_world_size()
    rank = get_rank()
    
    # Convert to numpy arrays for gathering
    local_k99 = np.array([s.k_99 for s in local_stats], dtype=np.int32)
    local_k999 = np.array([s.k_999 for s in local_stats], dtype=np.int32)
    local_top1 = np.array([s.top1_prob for s in local_stats], dtype=np.float32)
    local_entropy = np.array([s.entropy for s in local_stats], dtype=np.float32)
    
    # Gather sizes first
    local_size = torch.tensor([len(local_stats)], dtype=torch.long, device='cuda')
    all_sizes = [torch.zeros(1, dtype=torch.long, device='cuda') for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    all_sizes = [int(s.item()) for s in all_sizes]
    
    if rank == 0:
        # Prepare receive buffers
        all_k99 = [np.zeros(size, dtype=np.int32) for size in all_sizes]
        all_k999 = [np.zeros(size, dtype=np.int32) for size in all_sizes]
        all_top1 = [np.zeros(size, dtype=np.float32) for size in all_sizes]
        all_entropy = [np.zeros(size, dtype=np.float32) for size in all_sizes]
    else:
        all_k99 = all_k999 = all_top1 = all_entropy = None
    
    # Gather using gloo backend for CPU tensors or convert to GPU
    # Convert to torch tensors on GPU for gathering
    local_k99_t = torch.from_numpy(local_k99).cuda()
    local_k999_t = torch.from_numpy(local_k999).cuda()
    local_top1_t = torch.from_numpy(local_top1).cuda()
    local_entropy_t = torch.from_numpy(local_entropy).cuda()
    
    # Pad to max size for gather
    max_size = max(all_sizes)
    
    def pad_tensor(t, max_size):
        if len(t) < max_size:
            padded = torch.zeros(max_size, dtype=t.dtype, device=t.device)
            padded[:len(t)] = t
            return padded
        return t
    
    local_k99_t = pad_tensor(local_k99_t, max_size)
    local_k999_t = pad_tensor(local_k999_t, max_size)
    local_top1_t = pad_tensor(local_top1_t, max_size)
    local_entropy_t = pad_tensor(local_entropy_t, max_size)
    
    gathered_k99 = [torch.zeros(max_size, dtype=local_k99_t.dtype, device='cuda') for _ in range(world_size)]
    gathered_k999 = [torch.zeros(max_size, dtype=local_k999_t.dtype, device='cuda') for _ in range(world_size)]
    gathered_top1 = [torch.zeros(max_size, dtype=local_top1_t.dtype, device='cuda') for _ in range(world_size)]
    gathered_entropy = [torch.zeros(max_size, dtype=local_entropy_t.dtype, device='cuda') for _ in range(world_size)]
    
    dist.all_gather(gathered_k99, local_k99_t)
    dist.all_gather(gathered_k999, local_k999_t)
    dist.all_gather(gathered_top1, local_top1_t)
    dist.all_gather(gathered_entropy, local_entropy_t)
    
    if rank == 0:
        # Reconstruct StepStats
        all_step_stats = []
        for i in range(world_size):
            size = all_sizes[i]
            k99_arr = gathered_k99[i][:size].cpu().numpy()
            k999_arr = gathered_k999[i][:size].cpu().numpy()
            top1_arr = gathered_top1[i][:size].cpu().numpy()
            entropy_arr = gathered_entropy[i][:size].cpu().numpy()
            
            for j in range(size):
                all_step_stats.append(StepStats(
                    k_99=int(k99_arr[j]),
                    k_999=int(k999_arr[j]),
                    top1_prob=float(top1_arr[j]),
                    entropy=float(entropy_arr[j])
                ))
        
        return all_step_stats
    
    return []


def gather_coverage_mass_distributed(
    local_mass_by_k: Dict[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    """Gather per-step top-K mass arrays from all ranks to rank 0 (for coverage stats)."""
    if not dist.is_initialized() or not local_mass_by_k:
        return local_mass_by_k

    world_size = get_world_size()
    rank = get_rank()
    k_values = sorted(local_mass_by_k.keys())
    local_size = len(local_mass_by_k[k_values[0]])

    local_size_t = torch.tensor([local_size], dtype=torch.long, device='cuda')
    all_sizes = [torch.zeros(1, dtype=torch.long, device='cuda') for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size_t)
    all_sizes = [int(s.item()) for s in all_sizes]
    max_size = max(all_sizes)

    def pad_and_gather(arr: np.ndarray):
        t = torch.from_numpy(arr.astype(np.float32)).cuda()
        if len(t) < max_size:
            padded = torch.zeros(max_size, dtype=t.dtype, device='cuda')
            padded[:len(t)] = t
            t = padded
        gathered = [torch.zeros(max_size, dtype=t.dtype, device='cuda') for _ in range(world_size)]
        dist.all_gather(gathered, t)
        if rank == 0:
            return np.concatenate([gathered[i][:all_sizes[i]].cpu().numpy().astype(np.float64) for i in range(world_size)])
        return None

    if rank == 0:
        return {K: pad_and_gather(local_mass_by_k[K]) for K in k_values}
    for K in k_values:
        pad_and_gather(local_mass_by_k[K])
    return {}


# ============================================================================
# Reporting Functions
# ============================================================================

def generate_summary_report(stats: AggregateStats, extra_info: Optional[Dict] = None) -> str:
    """사람이 읽을 수 있는 요약 리포트를 생성합니다."""
    lines = []
    lines.append("=" * 70)
    lines.append("Teacher Logit K Analysis Report")
    lines.append("=" * 70)
    lines.append("")
    
    if extra_info:
        lines.append("[ Metadata ]")
        for k, v in extra_info.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    
    lines.append(f"[ 분석 대상: {stats.total_steps:,} 토큰 스텝 ]")
    lines.append("")
    
    lines.append("-" * 50)
    lines.append("K_{0.99} 분포 (누적확률 99%에 필요한 최소 토큰 수)")
    lines.append("-" * 50)
    lines.append(f"  평균 (mean):    {stats.k99_mean:.1f}")
    lines.append(f"  중앙값 (median): {stats.k99_median:.1f}")
    lines.append(f"  표준편차 (std): {stats.k99_std:.1f}")
    lines.append(f"  최소/최대:      {stats.k99_min} / {stats.k99_max}")
    lines.append(f"  백분위수:")
    for p, v in stats.k99_percentiles.items():
        lines.append(f"    {p}%ile: {v:.0f}")
    lines.append("")
    
    lines.append("-" * 50)
    lines.append("K_{0.999} 분포 (누적확률 99.9%에 필요한 최소 토큰 수)")
    lines.append("-" * 50)
    lines.append(f"  평균 (mean):    {stats.k999_mean:.1f}")
    lines.append(f"  중앙값 (median): {stats.k999_median:.1f}")
    lines.append(f"  표준편차 (std): {stats.k999_std:.1f}")
    lines.append(f"  최소/최대:      {stats.k999_min} / {stats.k999_max}")
    lines.append(f"  백분위수:")
    for p, v in stats.k999_percentiles.items():
        lines.append(f"    {p}%ile: {v:.0f}")
    lines.append("")
    
    lines.append("-" * 50)
    lines.append("Top-1 확률 분포")
    lines.append("-" * 50)
    lines.append(f"  평균: {stats.top1_mean:.4f}")
    lines.append(f"  중앙값: {stats.top1_median:.4f}")
    lines.append(f"  표준편차: {stats.top1_std:.4f}")
    lines.append("")
    
    lines.append("-" * 50)
    lines.append("Entropy 분포 (nats)")
    lines.append("-" * 50)
    lines.append(f"  평균: {stats.entropy_mean:.4f}")
    lines.append(f"  중앙값: {stats.entropy_median:.4f}")
    lines.append(f"  표준편차: {stats.entropy_std:.4f}")
    lines.append("")
    
    lines.append("=" * 70)
    lines.append("핵심 요약 (Key Findings)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"• 상위 99% 확률 질량을 위해 필요한 K의 중앙값은 {stats.k99_median:.0f}개입니다.")
    lines.append(f"• 상위 99.9% 확률 질량을 위해 필요한 K의 중앙값은 {stats.k999_median:.0f}개입니다.")
    lines.append("")
    lines.append(f"• 전체 스텝의 95%에서 K_{{0.99}} ≤ {stats.k99_at_95pct}")
    lines.append(f"• 전체 스텝의 95%에서 K_{{0.999}} ≤ {stats.k999_at_95pct}")
    lines.append(f"• 전체 스텝의 99%에서 K_{{0.999}} ≤ {stats.k999_at_99pct}")
    lines.append("")
    lines.append(f"• Top-1 토큰의 평균 확률: {stats.top1_mean:.2%}")
    lines.append(f"• 평균 Entropy: {stats.entropy_mean:.2f} nats")
    lines.append("")
    
    return "\n".join(lines)


def plot_distributions(step_stats: List[StepStats], output_dir: str, prefix: str = ""):
    """K 분포 히스토그램 및 CDF를 그립니다."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed. Skipping plots.")
        return
    
    k99_values = np.array([s.k_99 for s in step_stats])
    k999_values = np.array([s.k_999 for s in step_stats])
    top1_values = np.array([s.top1_prob for s in step_stats])
    entropy_values = np.array([s.entropy for s in step_stats])
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    ax = axes[0, 0]
    ax.hist(k99_values, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('K_{0.99}')
    ax.set_ylabel('Frequency')
    ax.set_title('K_{0.99} Distribution (Histogram)')
    ax.axvline(np.median(k99_values), color='r', linestyle='--', label=f'Median: {np.median(k99_values):.0f}')
    ax.legend()
    
    ax = axes[0, 1]
    sorted_k99 = np.sort(k99_values)
    cdf = np.arange(1, len(sorted_k99) + 1) / len(sorted_k99)
    ax.plot(sorted_k99, cdf)
    ax.set_xlabel('K_{0.99}')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('K_{0.99} CDF')
    ax.axhline(0.95, color='r', linestyle='--', alpha=0.5, label='95%')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    ax.hist(k999_values, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel('K_{0.999}')
    ax.set_ylabel('Frequency')
    ax.set_title('K_{0.999} Distribution (Histogram)')
    ax.axvline(np.median(k999_values), color='r', linestyle='--', label=f'Median: {np.median(k999_values):.0f}')
    ax.legend()
    
    ax = axes[1, 0]
    sorted_k999 = np.sort(k999_values)
    cdf = np.arange(1, len(sorted_k999) + 1) / len(sorted_k999)
    ax.plot(sorted_k999, cdf, color='orange')
    ax.set_xlabel('K_{0.999}')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('K_{0.999} CDF')
    ax.axhline(0.95, color='r', linestyle='--', alpha=0.5, label='95%')
    ax.axhline(0.99, color='g', linestyle='--', alpha=0.5, label='99%')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.hist(top1_values, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel('Top-1 Probability')
    ax.set_ylabel('Frequency')
    ax.set_title('Top-1 Probability Distribution')
    ax.axvline(np.median(top1_values), color='r', linestyle='--', label=f'Median: {np.median(top1_values):.3f}')
    ax.legend()
    
    ax = axes[1, 2]
    if len(k999_values) > 10000:
        idx = np.random.choice(len(k999_values), 10000, replace=False)
        ax.scatter(entropy_values[idx], k999_values[idx], alpha=0.1, s=1)
    else:
        ax.scatter(entropy_values, k999_values, alpha=0.1, s=1)
    ax.set_xlabel('Entropy (nats)')
    ax.set_ylabel('K_{0.999}')
    ax.set_title('K_{0.999} vs Entropy')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}k_distribution.png'), dpi=150)
    plt.close()
    
    # Log scale plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.hist(k99_values, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('K_{0.99}')
    ax.set_ylabel('Frequency (log scale)')
    ax.set_title('K_{0.99} Distribution (Log Scale)')
    ax.set_yscale('log')
    
    ax = axes[1]
    ax.hist(k999_values, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel('K_{0.999}')
    ax.set_ylabel('Frequency (log scale)')
    ax.set_title('K_{0.999} Distribution (Log Scale)')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}k_distribution_logscale.png'), dpi=150)
    plt.close()


def save_raw_stats(step_stats: List[StepStats], output_dir: str, prefix: str = ""):
    """원시 통계를 numpy 파일로 저장합니다."""
    k99_values = np.array([s.k_99 for s in step_stats])
    k999_values = np.array([s.k_999 for s in step_stats])
    top1_values = np.array([s.top1_prob for s in step_stats])
    entropy_values = np.array([s.entropy for s in step_stats])
    
    np.savez(
        os.path.join(output_dir, f'{prefix}raw_stats.npz'),
        k99=k99_values,
        k999=k999_values,
        top1_prob=top1_values,
        entropy=entropy_values
    )


# ============================================================================
# Main Analysis
# ============================================================================

def run_analysis(
    model_path: str,
    data_dir: str,
    output_dir: str,
    max_samples: int = 10000,
    batch_size: int = 8,
    max_length: int = 1024,
    temperature: float = 1.0,
    device: str = "cuda",
    dtype: str = "bf16",
    num_workers: int = 4,
    use_amp: bool = True,
    prefetch_factor: int = 2,
    distributed: bool = False
):
    """Teacher 모델로 teacher forcing하여 logits를 분석합니다."""
    
    # Setup distributed if requested
    if distributed:
        setup_distributed()
    
    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
    
    # Sync before proceeding
    if distributed:
        dist.barrier()
    
    # Dataset & DataLoader
    print_rank0(f"Loading data from {data_dir}...")
    dataset = BinDataset(data_dir, max_samples=max_samples)
    
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=False)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=create_collate_fn(max_length),
            pin_memory=True,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=create_collate_fn(max_length),
            pin_memory=True if device == 'cuda' else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False
        )
    
    # 모델 로드
    model, tokenizer = load_model_and_tokenizer(model_path, device=device, dtype=dtype, distributed=distributed)
    
    # 분석 (고정 K coverage 수집 포함)
    local_step_stats = []
    local_mass_by_k: Dict[int, List[np.ndarray]] = {K: [] for K in COVERAGE_K_VALUES}

    pbar = tqdm(dataloader, desc=f"[Rank {get_rank()}] Analyzing", disable=not is_main_process())
    for batch_idx, (input_ids, attention_mask) in enumerate(pbar):
        if input_ids is None:
            continue

        try:
            result = process_batch_logits(
                model, input_ids, attention_mask, temperature, use_amp,
                return_coverage=True,
                coverage_k_values=COVERAGE_K_VALUES,
            )
            batch_stats, mass_by_k_batch = result
            local_step_stats.extend(batch_stats)
            for K in COVERAGE_K_VALUES:
                local_mass_by_k[K].append(mass_by_k_batch[K])

            if is_main_process() and (batch_idx + 1) % 50 == 0:
                temp_agg = aggregate_stats(local_step_stats)
                pbar.set_postfix({
                    'steps': f'{len(local_step_stats):,}',
                    'K99_med': f'{temp_agg.k99_median:.0f}',
                    'K999_med': f'{temp_agg.k999_median:.0f}'
                })

            del batch_stats, mass_by_k_batch
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n[Rank {get_rank()}] Error at batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print_rank0(f"Local steps analyzed: {len(local_step_stats):,}")

    # Gather stats from all ranks
    if distributed:
        dist.barrier()
        all_step_stats = gather_stats_distributed(local_step_stats)
    else:
        all_step_stats = local_step_stats

    # Concatenate local coverage mass per K (single GPU) or gather (distributed)
    if distributed:
        mass_by_k_concat = {
            K: np.concatenate(local_mass_by_k[K], axis=0) if local_mass_by_k[K] else np.array([], dtype=np.float64)
            for K in COVERAGE_K_VALUES
        }
        all_mass_by_k = gather_coverage_mass_distributed(mass_by_k_concat)
    else:
        all_mass_by_k = {
            K: np.concatenate(local_mass_by_k[K], axis=0) if local_mass_by_k[K] else np.array([], dtype=np.float64)
            for K in COVERAGE_K_VALUES
        }

    # Only rank 0 saves results
    if is_main_process():
        if not all_step_stats:
            raise RuntimeError("No data was processed successfully")

        print(f"\nTotal steps analyzed: {len(all_step_stats):,}")

        agg_stats = aggregate_stats(all_step_stats)

        extra_info = {
            'model_path': model_path,
            'data_dir': data_dir,
            'max_samples': max_samples,
            'temperature': temperature,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'use_amp': use_amp,
            'distributed': distributed,
            'world_size': get_world_size(),
        }

        report = generate_summary_report(agg_stats, extra_info)
        report_path = os.path.join(output_dir, 'analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\n{report}")

        json_path = os.path.join(output_dir, 'aggregate_stats.json')
        with open(json_path, 'w') as f:
            json.dump(asdict(agg_stats), f, indent=2)

        # 고정 K Coverage Rate 요약 저장
        if all_mass_by_k and any(len(arr) > 0 for arr in all_mass_by_k.values()):
            coverage_summary = aggregate_coverage_from_arrays(all_mass_by_k, tau=temperature)
            coverage_path = os.path.join(output_dir, 'coverage_rate_0.99.json')
            with open(coverage_path, 'w') as f:
                json.dump(coverage_summary, f, indent=2)
            print_rank0(f"Coverage rate summary saved to {coverage_path}")

        plot_distributions(all_step_stats, output_dir)
        save_raw_stats(all_step_stats, output_dir)
    
    # Cleanup
    if distributed:
        cleanup_distributed()
    
    return all_step_stats if is_main_process() else []


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Teacher Logit K Analysis (Distributed)')
    
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    
    parser.add_argument('--max-samples', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default='bf16', choices=['bf16', 'fp16', 'fp32'])
    
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--distributed', action='store_true', help='Enable distributed multi-GPU processing')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print_rank0("=" * 70)
    print_rank0("Teacher Logit K Analysis (Distributed)")
    print_rank0("=" * 70)
    print_rank0(f"Model: {args.model_path}")
    print_rank0(f"Data: {args.data_dir}")
    print_rank0(f"Output: {args.output_dir}")
    print_rank0(f"Max samples: {args.max_samples}")
    print_rank0(f"Batch size: {args.batch_size}")
    print_rank0(f"Num workers: {args.num_workers}")
    print_rank0(f"AMP: {not args.no_amp}")
    print_rank0(f"Distributed: {args.distributed}")
    print_rank0("=" * 70)
    
    run_analysis(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        temperature=args.temperature,
        device=args.device,
        dtype=args.dtype,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        prefetch_factor=args.prefetch_factor,
        distributed=args.distributed
    )
    
    print_rank0("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
