"""
Sparse Knowledge Distillation Loss Functions

Sparse teacher logits에서 확률을 복원하고 KL divergence를 계산합니다.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional


def reconstruct_teacher_probs_from_sparse_batch(
    sparse_logits: Dict,
    vocab_size: int,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    배치 단위로 sparse logits에서 teacher 확률 복원 (GPU 효율적)
    
    Args:
        sparse_logits: Dataset collate에서 온 dict
            - token_ids: [batch, seq_len, K] int32
            - values: [batch, seq_len, K] float32 (probs or counts)
            - seq_lens: [batch] int32
            - valid_mask: [batch] bool
            - method: 'topk' or 'random'
            - (random only) lengths: [batch, seq_len] int16
            - (random only) num_samples: int
            - (topk only) k: int
        vocab_size: 전체 vocabulary 크기
        device: 연산 디바이스
    
    Returns:
        teacher_probs: [batch, seq_len, vocab_size] float32
                       valid_mask=False인 샘플은 0으로 채워짐
    """
    batch_size = sparse_logits['token_ids'].shape[0]
    seq_len = sparse_logits['token_ids'].shape[1]
    
    # GPU로 이동
    token_ids = sparse_logits['token_ids'].to(device, dtype=torch.long)  # [batch, seq_len, K]
    values = sparse_logits['values'].to(device, dtype=torch.float32)    # [batch, seq_len, K]
    valid_mask = sparse_logits['valid_mask'].to(device, dtype=torch.bool)  # [batch]
    seq_lens = sparse_logits['seq_lens'].to(device, dtype=torch.long)   # [batch]
    
    # 확률 텐서 초기화
    teacher_probs = torch.zeros(batch_size, seq_len, vocab_size, device=device, dtype=torch.float32)
    
    # Invalid token_ids (-1) 처리: scatter 전에 0으로 치환
    valid_token_mask = token_ids >= 0  # [batch, seq_len, K]
    token_ids_safe = torch.where(valid_token_mask, token_ids, torch.zeros_like(token_ids))  # -1 -> 0
    
    if sparse_logits['method'] == 'random':
        # Random Sampling: counts / num_samples
        num_samples = float(sparse_logits['num_samples'])
        prob_values = values / num_samples  # [batch, seq_len, K]
        
        # Invalid 위치는 확률값 0으로 (scatter 시 무시됨)
        prob_values = prob_values * valid_token_mask.float()
        
        # Scatter로 확률 할당 (token_ids_safe 사용, -1은 이미 0으로 치환됨)
        teacher_probs.scatter_(-1, token_ids_safe, prob_values)
        
    else:  # topk
        # Top-K: 확률값 그대로 사용
        # Invalid 위치는 확률값 0으로
        prob_values = values * valid_token_mask.float()
        
        # Scatter로 확률 할당
        teacher_probs.scatter_(-1, token_ids_safe, prob_values)
    
    # Valid mask 적용: valid_mask=False인 샘플은 모두 0으로
    teacher_probs = teacher_probs * valid_mask.unsqueeze(1).unsqueeze(2).float()
    
    # Sequence length 마스킹: seq_len 이후는 0으로 (vectorized)
    # [batch, seq_len] -> 각 position이 해당 샘플의 seq_len보다 작은지 체크
    position_indices = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
    seq_lens_expanded = seq_lens.unsqueeze(1)  # [batch, 1]
    length_mask = (position_indices < seq_lens_expanded).unsqueeze(2).float()  # [batch, seq_len, 1]
    teacher_probs = teacher_probs * length_mask
    
    return teacher_probs


def compute_sparse_kd_loss(
    student_logits: torch.Tensor,
    sparse_teacher_logits: Dict,
    loss_mask: torch.Tensor,
    vocab_size: int,
    device: str = 'cuda',
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Sparse KD Loss 계산 (KL divergence)
    
    Args:
        student_logits: [batch, seq_len, vocab_size] Student 모델 logits
        sparse_teacher_logits: Dataset에서 온 sparse logits dict
        loss_mask: [batch, seq_len] Loss 계산할 position 마스크
        vocab_size: Vocabulary 크기
        device: 연산 디바이스
        temperature: Temperature scaling (기본 1.0)
    
    Returns:
        kd_loss: [batch] 각 샘플의 KD loss
    """
    # Teacher 확률 복원
    teacher_probs = reconstruct_teacher_probs_from_sparse_batch(
        sparse_teacher_logits, vocab_size, device
    )  # [batch, seq_len, vocab_size]
    
    # Temperature scaling: Student와 Teacher 둘 다 적용
    if temperature != 1.0:
        # Teacher: log(probs) / T -> softmax
        teacher_logprobs_raw = torch.where(
            teacher_probs > 0,
            torch.log(teacher_probs),
            torch.full_like(teacher_probs, -1e8)  # log(0) = -inf 대신 큰 음수
        )
        teacher_probs = F.softmax(teacher_probs_raw / temperature, dim=-1)
    
    # Student log probabilities
    student_logprobs = F.log_softmax(student_logits / temperature, dim=-1, dtype=torch.float32)
    
    # KL divergence: D_KL(teacher || student) = sum(teacher_probs * log(teacher_probs / student_probs))
    # = sum(teacher_probs * log(teacher_probs)) - sum(teacher_probs * log(student_probs))
    # Gradient 계산에는 두 번째 항만 필요 (첫 번째는 teacher에 대한 상수)
    # Loss 절댓값은 다를 수 있지만, gradient는 동일
    
    # inf/nan 체크
    inf_mask = torch.isinf(student_logprobs) | torch.isnan(student_logprobs)
    student_logprobs = torch.masked_fill(student_logprobs, inf_mask, 0.0)
    
    # Cross-entropy: -sum(teacher_probs * log(student_probs))
    # (KL divergence의 두 번째 항, gradient는 동일)
    # Numerical stability: teacher_probs가 0인 경우 0 * log(...) = 0으로 자동 처리됨
    prod = teacher_probs * student_logprobs  # [batch, seq_len, vocab_size]
    prod = torch.masked_fill(prod, inf_mask, 0.0)
    
    # Sum over vocab dimension
    kl_terms = torch.sum(prod, dim=-1)  # [batch, seq_len]
    
    # Apply loss mask and average over sequence length
    masked_kl = kl_terms * loss_mask  # [batch, seq_len]
    kd_loss = -torch.sum(masked_kl, dim=-1) / (torch.sum(loss_mask, dim=-1) + 1e-8)  # [batch]
    
    return kd_loss


def compute_sparse_kd_entropy(
    sparse_teacher_logits: Dict,
    loss_mask: torch.Tensor,
    vocab_size: int,
    device: str = 'cuda',
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Teacher entropy 계산 (디버깅/모니터링용)
    
    Args:
        sparse_teacher_logits: Dataset에서 온 sparse logits dict
        loss_mask: [batch, seq_len] Loss 계산할 position 마스크
        vocab_size: Vocabulary 크기
        device: 연산 디바이스
        temperature: Temperature scaling
    
    Returns:
        entropy: [batch] 각 샘플의 teacher entropy
    """
    # Teacher 확률 복원
    teacher_probs = reconstruct_teacher_probs_from_sparse_batch(
        sparse_teacher_logits, vocab_size, device
    )  # [batch, seq_len, vocab_size]
    
    # Temperature scaling
    if temperature != 1.0:
        teacher_logprobs_raw = torch.where(
            teacher_probs > 0,
            torch.log(teacher_probs),
            torch.full_like(teacher_probs, -1e8)
        )
        teacher_probs = F.softmax(teacher_logprobs_raw / temperature, dim=-1)
    
    # Entropy: -sum(p * log(p))
    # Numerical stability: 0 * log(0) = 0으로 처리
    teacher_logprobs = torch.where(
        teacher_probs > 0,
        torch.log(teacher_probs),
        torch.zeros_like(teacher_probs)
    )
    entropy_terms = -teacher_probs * teacher_logprobs  # [batch, seq_len, vocab_size]
    entropy = torch.sum(entropy_terms, dim=-1)  # [batch, seq_len]
    
    # Apply loss mask and average
    masked_entropy = entropy * loss_mask  # [batch, seq_len]
    avg_entropy = torch.sum(masked_entropy, dim=-1) / (torch.sum(loss_mask, dim=-1) + 1e-8)  # [batch]
    
    return avg_entropy

