"""
Sparse Sampling for Knowledge Distillation

Two methods:
1. SparseLogitSampler (Random Sampling): Unbiased estimator via multinomial sampling
2. TopKSampler: Top-K tokens with their probabilities
"""

import torch
import numpy as np
from typing import Dict, Tuple


class SparseLogitSampler:
    """
    Random Sampling 방식: teacher 확률로 N번 샘플링
    
    장점: Unbiased estimator (기대값이 정확)
    원리: counts/num_samples → 확률 추정
    """
    
    def __init__(self, num_samples: int = 50):
        """
        Args:
            num_samples: 각 position에서 샘플링할 횟수 (N)
                        N이 클수록 정확하지만 저장 공간 증가
        """
        self.num_samples = num_samples
    
    def sample(self, probs: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Teacher 확률 분포에서 sparse representation 생성
        
        Args:
            probs: [seq_len, vocab_size] teacher 확률 (softmax 후)
        
        Returns:
            {
                'token_ids': [seq_len, max_unique] - 샘플링된 토큰 ID
                'counts': [seq_len, max_unique] - 각 토큰의 발생 횟수
                'num_samples': int - 총 샘플 수 (확률 복원용)
                'lengths': [seq_len] - 각 position의 실제 unique 토큰 수
            }
        """
        seq_len = probs.shape[0]
        all_token_ids = []
        all_counts = []
        
        for pos in range(seq_len):
            # N번 샘플링 (중복 허용 - replacement=True)
            samples = torch.multinomial(probs[pos], self.num_samples, replacement=True)
            
            # 고유 토큰 + 발생 횟수 계산
            unique_ids, counts = torch.unique(samples, return_counts=True)
            all_token_ids.append(unique_ids.cpu().numpy())
            all_counts.append(counts.cpu().numpy())
        
        # Padding (배치로 묶을 때 필요)
        max_unique = max(len(ids) for ids in all_token_ids)
        lengths = np.array([len(ids) for ids in all_token_ids], dtype=np.int16)
        
        # -1로 패딩 (유효하지 않은 토큰 표시)
        token_ids_padded = np.full((seq_len, max_unique), -1, dtype=np.int32)
        counts_padded = np.zeros((seq_len, max_unique), dtype=np.uint8)
        
        for i, (ids, cnts) in enumerate(zip(all_token_ids, all_counts)):
            token_ids_padded[i, :len(ids)] = ids
            counts_padded[i, :len(cnts)] = cnts
        
        return {
            'token_ids': token_ids_padded,  # [seq_len, max_unique]
            'counts': counts_padded,         # [seq_len, max_unique]
            'num_samples': np.int16(self.num_samples),
            'lengths': lengths               # [seq_len]
        }
    
    def sample_batch(self, probs: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        배치 단위 샘플링 (더 효율적)
        
        Args:
            probs: [batch_size, seq_len, vocab_size]
        
        Returns:
            List of sparse representations for each sequence
        """
        batch_size = probs.shape[0]
        results = []
        for b in range(batch_size):
            results.append(self.sample(probs[b]))
        return results


class TopKSampler:
    """
    Top-K 방식: 상위 K개 토큰과 확률값 직접 저장
    
    장점: 구현 간단, 확률값 정확
    단점: Biased (tail 확률 무시, 합이 1 안됨)
    """
    
    def __init__(self, k: int = 50):
        """
        Args:
            k: 저장할 상위 토큰 개수
        """
        self.k = k
    
    def sample(self, probs: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        상위 K개 토큰과 확률 추출
        
        Args:
            probs: [seq_len, vocab_size] teacher 확률
        
        Returns:
            {
                'token_ids': [seq_len, K] - 상위 K개 토큰 ID
                'probs': [seq_len, K] - 해당 확률값 (float16)
                'k': int
            }
        """
        topk_probs, topk_ids = probs.topk(self.k, dim=-1)  # [seq_len, K]
        
        return {
            'token_ids': topk_ids.cpu().numpy().astype(np.int32),
            'probs': topk_probs.cpu().numpy().astype(np.float16),
            'k': np.int16(self.k)
        }
    
    def sample_batch(self, probs: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        배치 단위 Top-K 추출 (벡터화로 더 효율적)
        
        Args:
            probs: [batch_size, seq_len, vocab_size]
        
        Returns:
            {
                'token_ids': [batch_size, seq_len, K]
                'probs': [batch_size, seq_len, K]
                'k': int
            }
        """
        topk_probs, topk_ids = probs.topk(self.k, dim=-1)
        
        return {
            'token_ids': topk_ids.cpu().numpy().astype(np.int32),
            'probs': topk_probs.cpu().numpy().astype(np.float16),
            'k': np.int16(self.k)
        }


def reconstruct_probs_from_sparse(sparse_data: Dict, vocab_size: int) -> torch.Tensor:
    """
    Sparse representation에서 확률 복원 (학습 시 사용)
    
    Args:
        sparse_data: sample() 메서드의 출력
        vocab_size: 전체 vocabulary 크기
    
    Returns:
        probs: [seq_len, vocab_size] 복원된 확률 (0으로 채워진 sparse)
    """
    token_ids = sparse_data['token_ids']  # [seq_len, K]
    seq_len = token_ids.shape[0]
    
    # 확률 텐서 초기화
    probs = torch.zeros(seq_len, vocab_size)
    
    if 'counts' in sparse_data:
        # Random Sampling 방식
        counts = sparse_data['counts']
        num_samples = sparse_data['num_samples']
        
        for pos in range(seq_len):
            valid_mask = token_ids[pos] >= 0
            valid_ids = token_ids[pos][valid_mask]
            valid_counts = counts[pos][valid_mask]
            probs[pos, valid_ids] = torch.tensor(valid_counts / num_samples, dtype=torch.float32)
    else:
        # Top-K 방식
        prob_values = sparse_data['probs']  # [seq_len, K]
        
        for pos in range(seq_len):
            probs[pos, token_ids[pos]] = torch.tensor(prob_values[pos], dtype=torch.float32)
    
    return probs


def reconstruct_probs_batch(sparse_data: Dict, vocab_size: int, device: str = 'cuda') -> torch.Tensor:
    """
    배치 단위로 sparse representation에서 확률 복원 (GPU 연산)
    
    Args:
        sparse_data: sample_batch()의 출력
        vocab_size: vocabulary 크기
        device: 연산 디바이스
    
    Returns:
        probs: [batch_size, seq_len, vocab_size]
    """
    token_ids = torch.tensor(sparse_data['token_ids'], device=device, dtype=torch.long)
    
    if 'counts' in sparse_data:
        # Random Sampling
        counts = torch.tensor(sparse_data['counts'], device=device, dtype=torch.float32)
        num_samples = float(sparse_data['num_samples'])
        prob_values = counts / num_samples
    else:
        # Top-K
        prob_values = torch.tensor(sparse_data['probs'], device=device, dtype=torch.float32)
    
    batch_size, seq_len, k = token_ids.shape
    probs = torch.zeros(batch_size, seq_len, vocab_size, device=device)
    
    # scatter로 효율적 할당
    probs.scatter_(-1, token_ids, prob_values)
    
    return probs

