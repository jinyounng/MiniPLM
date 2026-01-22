"""
Sparse Knowledge Distillation Loss Functions

Sparse teacher logits에서 KL divergence를 계산합니다.
★ 최적화: Full vocab reconstruction 없이 sparse token들에 대해서만 loss 계산
★ Top-K normalization 적용
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional
import warnings

# Temperature scaling 경고를 한 번만 출력하기 위한 플래그
_temperature_warning_shown = False


def compute_sparse_kd_loss_efficient(
    student_logits: torch.Tensor,
    sparse_teacher_logits: Dict,
    loss_mask: torch.Tensor,
    device: str = 'cuda',
    temperature: float = 1.0
) -> torch.Tensor:
    """
    ★ 효율적인 Sparse KD Loss 계산 (Full vocab reconstruction 없음)
    
    Sparse token들에 대해서만 loss 계산하여 메모리/속도 최적화.
    Cross-entropy loss: -sum(teacher_probs * log(student_probs)) on sparse tokens only
    
    Args:
        student_logits: [batch, seq_len, vocab_size] Student 모델 logits
        sparse_teacher_logits: Dataset에서 온 sparse logits dict
            - token_ids: [batch, seq_len, K]
            - values: [batch, seq_len, K] (probs or counts)
            - method: 'topk' or 'random'
        loss_mask: [batch, seq_len] Loss 계산할 position 마스크
        device: 연산 디바이스
        temperature: Temperature scaling (기본 1.0)
    
    Returns:
        kd_loss: [batch] 각 샘플의 KD loss
    """
    # GPU로 이동
    token_ids = sparse_teacher_logits['token_ids'].to(device, dtype=torch.long)  # [batch, seq_len, K]
    values = sparse_teacher_logits['values'].to(device, dtype=torch.float32)    # [batch, seq_len, K]
    valid_mask = sparse_teacher_logits['valid_mask'].to(device, dtype=torch.bool)  # [batch]
    seq_lens = sparse_teacher_logits.get('seq_lens', None)  # [batch] 실제 시퀀스 길이
    
    batch_size, seq_len, K = token_ids.shape
    
    # loss_mask와 student_logits의 실제 길이 확인
    student_seq_len = student_logits.shape[1]
    loss_mask_seq_len = loss_mask.shape[1] if loss_mask.dim() > 1 else loss_mask.shape[0]
    
    # 시퀀스 길이 불일치 경고
    if seq_len != student_seq_len or seq_len != loss_mask_seq_len:
        method_name = sparse_teacher_logits.get('method', 'unknown')
        warnings.warn(
            f"⚠️ Sequence length mismatch detected ({method_name} KD): "
            f"cached_teacher_logits={seq_len}, student_logits={student_seq_len}, "
            f"loss_mask={loss_mask_seq_len}. Using min={min(seq_len, student_seq_len, loss_mask_seq_len)}. "
            f"This may indicate a data loading or batch processing issue.",
            UserWarning
        )
    
    # 길이 맞추기: 최소값 사용
    actual_seq_len = min(seq_len, student_seq_len, loss_mask_seq_len)
    
    # 실제 길이에 맞춰서 데이터 정렬
    token_ids_aligned = token_ids[:, :actual_seq_len, :]  # [batch, actual_seq_len, K]
    values_aligned = values[:, :actual_seq_len, :]  # [batch, actual_seq_len, K]
    valid_token_mask_aligned = token_ids_aligned >= 0  # [batch, actual_seq_len, K]
    token_ids_safe_aligned = torch.where(valid_token_mask_aligned, token_ids_aligned, torch.zeros_like(token_ids_aligned))
    
    # Teacher 확률 계산 (sparse 형태 유지, 실제 길이에 맞춤)
    if sparse_teacher_logits['method'] == 'random':
        # Random Sampling: counts / num_samples -> 이미 unbiased estimator
        # ★ Normalize 하지 않음! (합이 이미 1)
        num_samples = float(sparse_teacher_logits['num_samples'])
        teacher_probs_normalized = values_aligned / num_samples  # [batch, actual_seq_len, K]
        teacher_probs_normalized = teacher_probs_normalized * valid_token_mask_aligned.float()
        
    else:  # topk
        # Top-K: normalize 필요 (합이 1이 아닐 수 있음, tail 확률 버림)
        teacher_probs_sparse = values_aligned.clone()  # [batch, actual_seq_len, K]
        teacher_probs_sparse = teacher_probs_sparse * valid_token_mask_aligned.float()
        
        # ★ Top-K만 Normalization: 합이 1이 되도록
        prob_sum = teacher_probs_sparse.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [batch, actual_seq_len, 1]
        teacher_probs_normalized = teacher_probs_sparse / prob_sum  # [batch, actual_seq_len, K]
    
    # Temperature scaling on teacher (optional)
    if temperature != 1.0:
        global _temperature_warning_shown
        if not _temperature_warning_shown:
            warnings.warn(
                f"⚠️ Sparse KD with temperature={temperature}: "
                f"Teacher softmax is applied over K tokens only (not full vocab). "
                f"This may differ from standard temperature scaling. "
                f"Consider using temperature=1.0 for sparse KD.",
                UserWarning
            )
            _temperature_warning_shown = True
        
        # log(probs) / T -> softmax over K dimension
        teacher_log = torch.where(
            teacher_probs_normalized > 0,
            torch.log(teacher_probs_normalized),
            torch.full_like(teacher_probs_normalized, -1e8)
        )
        teacher_probs_normalized = F.softmax(teacher_log / temperature, dim=-1)
    
    # ★ Student log probabilities: 전체 vocab에 대해 log_softmax 후 sparse token만 gather
    # 중요: log_softmax는 전체 vocab에 대해 계산해야 올바른 확률 분포!
    # student_logits: [batch, seq_len, vocab_size] -> 실제 길이에 맞춤
    student_logits_aligned = student_logits[:, :actual_seq_len, :]  # [batch, actual_seq_len, vocab_size]
    
    student_logprobs_full = F.log_softmax(student_logits_aligned / temperature, dim=-1, dtype=torch.float32)  # [batch, actual_seq_len, vocab_size]
    
    # Sparse token들만 gather (메모리: vocab_size → K)
    student_logprobs_sparse = torch.gather(student_logprobs_full, dim=-1, index=token_ids_safe_aligned)  # [batch, actual_seq_len, K]
    
    # student_logprobs_full 메모리 즉시 해제 (vocab_size 차원)
    del student_logprobs_full
    
    # Cross-entropy: -sum(teacher_probs * log(student_probs)) over K dimension
    # inf/nan 체크 (이미 actual_seq_len에 맞춰짐)
    inf_mask = torch.isinf(student_logprobs_sparse) | torch.isnan(student_logprobs_sparse)
    student_logprobs_sparse = torch.masked_fill(student_logprobs_sparse, inf_mask, 0.0)
    
    prod = teacher_probs_normalized * student_logprobs_sparse  # [batch, actual_seq_len, K]
    prod = torch.masked_fill(prod, inf_mask, 0.0)
    prod = prod * valid_token_mask_aligned.float()  # Invalid 위치 제거
    
    # Sum over K dimension
    kl_terms = torch.sum(prod, dim=-1)  # [batch, seq_len]
    
    # 길이 맞추기: 실제 길이만큼만 사용
    kl_terms = kl_terms[:, :actual_seq_len]  # [batch, actual_seq_len]
    
    # loss_mask도 실제 길이에 맞춤
    if loss_mask.dim() == 2:
        loss_mask_aligned = loss_mask[:, :actual_seq_len]  # [batch, actual_seq_len]
    else:
        # loss_mask가 1D인 경우 (배치 전체에 동일한 마스크)
        loss_mask_aligned = loss_mask[:actual_seq_len].unsqueeze(0).expand(batch_size, -1)  # [batch, actual_seq_len]
    
    # Apply loss mask and valid_mask
    valid_mask_expanded = valid_mask.unsqueeze(1).float()  # [batch, 1]
    masked_kl = kl_terms * loss_mask_aligned * valid_mask_expanded  # [batch, actual_seq_len]
    
    # 분모: valid_mask가 True인 경우만 loss_mask 합 계산
    # valid_mask가 False인 샘플은 분모도 0이 되어 kd_loss = 0
    denominator = torch.sum(loss_mask_aligned * valid_mask_expanded, dim=-1)  # [batch]
    kd_loss = -torch.sum(masked_kl, dim=-1) / (denominator + 1e-8)  # [batch]
    
    # valid_mask가 False인 샘플은 kd_loss = 0으로 명시적 설정
    kd_loss = kd_loss * valid_mask.float()  # [batch]
    
    return kd_loss


def reconstruct_teacher_probs_from_sparse_batch(
    sparse_logits: Dict,
    vocab_size: int,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    배치 단위로 sparse logits에서 teacher 확률 복원 (GPU 효율적)
    ★ 주의: 메모리 비효율적 (vocab_size 차원 생성)
    ★ 가능하면 compute_sparse_kd_loss_efficient() 사용 권장
    
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
        # Top-K: 확률값 사용
        prob_values = values * valid_token_mask.float()
        
        # ★ Top-K Normalization: 합이 1이 되도록 normalize
        prob_sum = prob_values.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        prob_values = prob_values / prob_sum
        
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
    ★ 효율적인 버전 사용: compute_sparse_kd_loss_efficient()
    
    Args:
        student_logits: [batch, seq_len, vocab_size] Student 모델 logits
        sparse_teacher_logits: Dataset에서 온 sparse logits dict
        loss_mask: [batch, seq_len] Loss 계산할 position 마스크
        vocab_size: Vocabulary 크기 (사용 안 함, 호환성 위해 유지)
        device: 연산 디바이스
        temperature: Temperature scaling (기본 1.0)
    
    Returns:
        kd_loss: [batch] 각 샘플의 KD loss
    """
    # ★ 효율적인 버전 사용 (vocab_size 불필요)
    return compute_sparse_kd_loss_efficient(
        student_logits=student_logits,
        sparse_teacher_logits=sparse_teacher_logits,
        loss_mask=loss_mask,
        device=device,
        temperature=temperature
    )


def compute_sparse_kd_entropy_efficient(
    sparse_teacher_logits: Dict,
    loss_mask: torch.Tensor,
    device: str = 'cuda',
    temperature: float = 1.0
) -> torch.Tensor:
    """
    ★ 효율적인 Teacher entropy 계산 (sparse token들만 사용)
    
    Args:
        sparse_teacher_logits: Dataset에서 온 sparse logits dict
        loss_mask: [batch, seq_len] Loss 계산할 position 마스크
        device: 연산 디바이스
        temperature: Temperature scaling
    
    Returns:
        entropy: [batch] 각 샘플의 teacher entropy
    """
    # GPU로 이동
    token_ids = sparse_teacher_logits['token_ids'].to(device, dtype=torch.long)  # [batch, seq_len, K]
    values = sparse_teacher_logits['values'].to(device, dtype=torch.float32)    # [batch, seq_len, K]
    valid_mask = sparse_teacher_logits['valid_mask'].to(device, dtype=torch.bool)  # [batch]
    
    batch_size, seq_len, K = token_ids.shape
    
    # loss_mask의 실제 길이 확인
    loss_mask_seq_len = loss_mask.shape[1] if loss_mask.dim() > 1 else loss_mask.shape[0]
    
    # 시퀀스 길이 불일치 경고
    if seq_len != loss_mask_seq_len:
        method_name = sparse_teacher_logits.get('method', 'unknown')
        warnings.warn(
            f"⚠️ Sequence length mismatch detected in entropy calculation ({method_name} KD): "
            f"cached_teacher_logits={seq_len}, loss_mask={loss_mask_seq_len}. "
            f"Using min={min(seq_len, loss_mask_seq_len)}. "
            f"This may indicate a data loading or batch processing issue.",
            UserWarning
        )
    
    # 길이 맞추기: 최소값 사용
    actual_seq_len = min(seq_len, loss_mask_seq_len)
    
    # 실제 길이에 맞춰서 데이터 정렬
    token_ids_aligned = token_ids[:, :actual_seq_len, :]  # [batch, actual_seq_len, K]
    values_aligned = values[:, :actual_seq_len, :]  # [batch, actual_seq_len, K]
    
    # Invalid token_ids (-1) 처리
    valid_token_mask_aligned = token_ids_aligned >= 0  # [batch, actual_seq_len, K]
    
    # Teacher 확률 계산 (sparse 형태 유지, 실제 길이에 맞춤)
    if sparse_teacher_logits['method'] == 'random':
        # Random Sampling: counts / num_samples -> 이미 unbiased estimator
        # ★ Normalize 하지 않음! (합이 이미 1)
        num_samples = float(sparse_teacher_logits['num_samples'])
        teacher_probs_normalized = values_aligned / num_samples  # [batch, actual_seq_len, K]
        teacher_probs_normalized = teacher_probs_normalized * valid_token_mask_aligned.float()
        
    else:  # topk
        # Top-K: normalize 필요 (합이 1이 아닐 수 있음)
        teacher_probs_sparse = values_aligned.clone()  # [batch, actual_seq_len, K]
        teacher_probs_sparse = teacher_probs_sparse * valid_token_mask_aligned.float()
        
        # ★ Top-K만 Normalization
        prob_sum = teacher_probs_sparse.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [batch, actual_seq_len, 1]
        teacher_probs_normalized = teacher_probs_sparse / prob_sum  # [batch, actual_seq_len, K]
    
    # Temperature scaling
    if temperature != 1.0:
        global _temperature_warning_shown
        if not _temperature_warning_shown:
            warnings.warn(
                f"⚠️ Sparse KD entropy with temperature={temperature}: "
                f"Softmax is applied over K tokens only (not full vocab). "
                f"Consider using temperature=1.0 for sparse KD.",
                UserWarning
            )
            _temperature_warning_shown = True
        
        teacher_log = torch.where(
            teacher_probs_normalized > 0,
            torch.log(teacher_probs_normalized),
            torch.full_like(teacher_probs_normalized, -1e8)
        )
        teacher_probs_normalized = F.softmax(teacher_log / temperature, dim=-1)
    
    # Entropy: -sum(p * log(p)) over K dimension
    teacher_logprobs = torch.where(
        teacher_probs_normalized > 0,
        torch.log(teacher_probs_normalized),
        torch.zeros_like(teacher_probs_normalized)
    )
    entropy_terms = -teacher_probs_normalized * teacher_logprobs  # [batch, actual_seq_len, K]
    entropy_terms = entropy_terms * valid_token_mask_aligned.float()  # Invalid 위치 제거
    entropy = torch.sum(entropy_terms, dim=-1)  # [batch, actual_seq_len]
    
    # loss_mask도 실제 길이에 맞춤
    if loss_mask.dim() == 2:
        loss_mask_aligned = loss_mask[:, :actual_seq_len]  # [batch, actual_seq_len]
    else:
        # loss_mask가 1D인 경우 (배치 전체에 동일한 마스크)
        loss_mask_aligned = loss_mask[:actual_seq_len].unsqueeze(0).expand(batch_size, -1)  # [batch, actual_seq_len]
    
    # Apply loss mask and valid_mask
    masked_entropy = entropy * loss_mask_aligned * valid_mask.unsqueeze(1).float()  # [batch, actual_seq_len]
    avg_entropy = torch.sum(masked_entropy, dim=-1) / (torch.sum(loss_mask_aligned, dim=-1) + 1e-8)  # [batch]
    
    return avg_entropy


def compute_sparse_kd_entropy(
    sparse_teacher_logits: Dict,
    loss_mask: torch.Tensor,
    vocab_size: int,
    device: str = 'cuda',
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Teacher entropy 계산 (디버깅/모니터링용)
    ★ 효율적인 버전 사용: compute_sparse_kd_entropy_efficient()
    
    Args:
        sparse_teacher_logits: Dataset에서 온 sparse logits dict
        loss_mask: [batch, seq_len] Loss 계산할 position 마스크
        vocab_size: Vocabulary 크기 (사용 안 함, 호환성 위해 유지)
        device: 연산 디바이스
        temperature: Temperature scaling
    
    Returns:
        entropy: [batch] 각 샘플의 teacher entropy
    """
    return compute_sparse_kd_entropy_efficient(
        sparse_teacher_logits=sparse_teacher_logits,
        loss_mask=loss_mask,
        device=device,
        temperature=temperature
    )

