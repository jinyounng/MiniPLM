"""
Sparse KD Dataset: LMDataset with cached teacher logits support

기존 텍스트 데이터와 함께 캐싱된 sparse teacher logits를 로드합니다.
"""

import os
import json
import numpy as np
import torch
from glob import glob
from typing import Dict, Optional, Tuple
from collections import OrderedDict

from utils import print_rank
from .lm_datasets import LMDataset


class SparseKDLMDataset(LMDataset):
    """
    LMDataset with cached sparse teacher logits support
    
    Usage:
        dataset = SparseKDLMDataset(
            args, tokenizer, split="data", 
            data_path="/path/to/data",
            cached_logits_dir="/path/to/cached_logits"
        )
    """
    
    def __init__(
        self, 
        args, 
        tokenizer, 
        split, 
        data_path=None, 
        num=-1, 
        ada_max_length=False,
        cached_logits_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            cached_logits_dir: 캐싱된 logits 디렉토리 경로
                              (예: /data/jykim/DB/miniplm_refined_corpus_logits_topk)
        """
        # 먼저 부모 클래스 초기화 (텍스트 데이터 로드)
        super().__init__(args, tokenizer, split, data_path, num, ada_max_length, **kwargs)
        
        # Cached logits 로드
        self.cached_logits_dir = cached_logits_dir
        self.sparse_logits_loaded = False
        # LRU cache: 최대 10개 shard만 메모리에 유지
        self.sparse_logits_cache = OrderedDict()  # shard_id -> npz data
        self.max_cache_size = 10
        
        if cached_logits_dir is not None:
            self._load_cached_logits_metadata()
        else:
            print_rank("⚠️ No cached_logits_dir provided, sparse KD will not work")
    
    def _load_cached_logits_metadata(self):
        """Cached logits 메타데이터 로드 및 인덱스 매핑 생성"""
        metadata_path = os.path.join(self.cached_logits_dir, 'metadata.json')
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}\n"
                f"Please run cache_teacher_logits.py first!"
            )
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.logits_metadata = json.load(f)
        
        self.logits_method = self.logits_metadata['method']  # 'topk', 'random', or 'both'
        self.logits_vocab_size = self.logits_metadata['vocab_size']
        self.shard_sizes = self.logits_metadata['shard_sizes']
        self.shard_offsets = self.logits_metadata['shard_offsets']
        
        # 'both'일 때 사용할 방법 결정 (args에서 받거나 기본값)
        if self.logits_method == 'both':
            # args에서 kd_method 확인 (없으면 기본값 'topk')
            self.use_method = getattr(self.args, 'kd_method', 'topk')
            if self.use_method not in ['topk', 'sparse']:
                print_rank(f"   ⚠️ Invalid kd_method={self.use_method}, defaulting to 'topk'")
                self.use_method = 'topk'
            print_rank(f"   ✅ Method is 'both', using '{self.use_method}' for training")
        else:
            self.use_method = self.logits_method  # 'topk' or 'random'
        
        print_rank(f"✅ Loaded cached logits metadata:")
        print_rank(f"   Cached method: {self.logits_method}")
        print_rank(f"   Using method: {self.use_method}")
        print_rank(f"   Vocab size: {self.logits_vocab_size}")
        print_rank(f"   Total shards: {len(self.shard_sizes)}")
        print_rank(f"   Total sequences: {self.logits_metadata['total_sequences']:,}")
        
        # Find all shard npz files (data_*.npz 패턴)
        npz_files = sorted(glob(os.path.join(self.cached_logits_dir, 'data_*.npz')))
        
        if not npz_files:
            raise FileNotFoundError(
                f"No cached logits files found in {self.cached_logits_dir}\n"
                f"Expected pattern: data_*.npz"
            )
        
        # Extract shard IDs
        def get_shard_id(path):
            basename = os.path.basename(path)
            name_no_ext = basename.rsplit('.', 1)[0]  # "data_5"
            return int(name_no_ext.split('_')[-1])    # 5
        
        self.shard_paths = {get_shard_id(p): p for p in npz_files}
        print_rank(f"   Found {len(self.shard_paths)} cached logits shards")
        
        # Build global_idx -> (shard_id, local_idx) mapping
        # (lazy loading을 위해 메타데이터만 저장)
        self.sparse_logits_loaded = True
    
    def _get_shard_and_local_idx(self, global_idx: int) -> Tuple[int, int]:
        """
        Global index로부터 shard_id와 local_idx 찾기 (Binary search)
        
        Returns:
            (shard_id, local_idx)
        """
        # Binary search로 올바른 shard 찾기
        # offsets = [0, 1000, 2000, 3000]일 때:
        # - global_idx=500 → shard_id=0 (offset[0] <= 500 < offset[1])
        # - global_idx=1500 → shard_id=1 (offset[1] <= 1500 < offset[2])
        # - global_idx=3500 → shard_id=3 (offset[3] <= 3500 < offset[3] + shard_sizes[3])
        
        left, right = 0, len(self.shard_offsets) - 1
        shard_id = 0
        
        while left <= right:
            mid = (left + right) // 2
            offset = self.shard_offsets[mid]
            
            # 다음 shard의 시작점 계산
            if mid + 1 < len(self.shard_offsets):
                next_offset = self.shard_offsets[mid + 1]
            else:
                # 마지막 shard: offset + shard_size까지
                next_offset = offset + self.shard_sizes[mid]
            
            if offset <= global_idx < next_offset:
                shard_id = mid
                break
            elif global_idx < offset:
                right = mid - 1
            else:  # global_idx >= next_offset
                left = mid + 1
                shard_id = mid + 1
        
        # 마지막 shard 범위 체크 (안전장치)
        if shard_id >= len(self.shard_offsets):
            shard_id = len(self.shard_offsets) - 1
        
        local_idx = global_idx - self.shard_offsets[shard_id]
        return shard_id, local_idx
    
    def _load_shard_logits(self, shard_id: int) -> Dict:
        """
        특정 shard의 logits 로드 (LRU cache with lazy loading)
        
        최대 max_cache_size개만 메모리에 유지하고, 초과 시 가장 오래된 것 제거
        """
        # Cache hit: 최근 사용으로 이동
        if shard_id in self.sparse_logits_cache:
            self.sparse_logits_cache.move_to_end(shard_id)
            return self.sparse_logits_cache[shard_id]
        
        # Cache miss: 새로 로드
        if shard_id not in self.shard_paths:
            raise KeyError(f"Shard {shard_id} not found in cached logits")
        
        npz_path = self.shard_paths[shard_id]
        shard_data = np.load(npz_path, allow_pickle=True)
        
        # LRU cache: 최대 크기 초과 시 가장 오래된 것 제거
        if len(self.sparse_logits_cache) >= self.max_cache_size:
            oldest_shard_id, oldest_data = self.sparse_logits_cache.popitem(last=False)
            del oldest_data  # 메모리 해제
            print_rank(f"   Evicted shard {oldest_shard_id} from cache (max_size={self.max_cache_size})")
        
        self.sparse_logits_cache[shard_id] = shard_data
        self.sparse_logits_cache.move_to_end(shard_id)  # 최근 사용으로 표시
        print_rank(f"   Loaded logits shard {shard_id} from {os.path.basename(npz_path)} (cache_size={len(self.sparse_logits_cache)})")
        
        return shard_data
    
    def __getitem__(self, index: int):
        """
        Returns:
            (index, data, sparse_logits_dict) or (index, data, None)
        """
        # 부모 클래스에서 기본 데이터 가져오기
        result = super().__getitem__(index)
        
        if result is None:
            return None
        
        idx, data = result
        
        # Cached logits 로드
        sparse_logits = None
        if self.sparse_logits_loaded:
            try:
                shard_id, local_idx = self._get_shard_and_local_idx(idx)
                shard_data = self._load_shard_logits(shard_id)
                
                # Bounds check
                if local_idx < 0 or local_idx >= len(shard_data['seq_lens']):
                    raise IndexError(
                        f"local_idx={local_idx} out of range [0, {len(shard_data['seq_lens'])}) "
                        f"for shard_id={shard_id}, global_idx={idx}"
                    )
                
                # 해당 시퀀스의 sparse logits 추출
                if self.logits_method == 'both':
                    # 'both' 방식: 선택한 방법에 따라 데이터 추출
                    if self.use_method == 'topk':
                        sparse_logits = {
                            'token_ids': shard_data['topk_token_ids'][local_idx],  # [seq_len, K]
                            'values': shard_data['topk_probs'][local_idx],          # [seq_len, K]
                            'seq_len': int(shard_data['seq_lens'][local_idx]),
                            'method': 'topk',
                            'k': int(shard_data['topk_k']),
                        }
                    else:  # sparse
                        sparse_logits = {
                            'token_ids': shard_data['sparse_token_ids'][local_idx],  # [seq_len, K]
                            'values': shard_data['sparse_counts'][local_idx],         # [seq_len, K]
                            'seq_len': int(shard_data['seq_lens'][local_idx]),
                            'method': 'random',
                            'lengths': shard_data['sparse_lengths'][local_idx],       # [seq_len]
                            'num_samples': int(shard_data['sparse_num_samples']),
                        }
                else:
                    # 기존 방식 (topk 또는 random)
                    sparse_logits = {
                        'token_ids': shard_data['token_ids'][local_idx],  # [seq_len, K]
                        'values': shard_data['values'][local_idx],        # [seq_len, K]
                        'seq_len': int(shard_data['seq_lens'][local_idx]),
                        'method': str(shard_data['method']),
                    }
                    
                    # Random sampling의 경우 lengths 추가
                    if self.logits_method == 'random' and 'lengths' in shard_data:
                        sparse_logits['lengths'] = shard_data['lengths'][local_idx]  # [seq_len]
                        sparse_logits['num_samples'] = int(shard_data['num_samples'])
                    elif self.logits_method == 'topk':
                        sparse_logits['k'] = int(shard_data['k'])
                
            except (KeyError, IndexError, ValueError) as e:
                print_rank(
                    f"⚠️ Failed to load sparse logits: global_idx={idx}, "
                    f"shard_id={shard_id if 'shard_id' in locals() else 'unknown'}, "
                    f"local_idx={local_idx if 'local_idx' in locals() else 'unknown'}, "
                    f"error={type(e).__name__}: {e}"
                )
                sparse_logits = None
        
        return idx, data, sparse_logits
    
    def collate(self, samples):
        """
        Collate function with sparse logits support
        
        Returns:
            model_batch, no_model_batch
            no_model_batch에 'sparse_logits' 키 추가
        """
        if samples[0] is None:
            return None, None
        
        # samples는 (idx, data, sparse_logits) 3개 튜플
        # 부모 클래스 collate는 (idx, data) 2개 튜플 기대
        base_samples = []
        sparse_logits_list = []
        
        for sample in samples:
            if len(sample) == 3:
                idx, data, sparse_logits = sample
                base_samples.append((idx, data))
                sparse_logits_list.append(sparse_logits)
            elif len(sample) == 2:
                # Fallback: 부모 클래스 형식 (sparse_logits 없음)
                idx, data = sample
                base_samples.append((idx, data))
                sparse_logits_list.append(None)
            else:
                raise ValueError(f"Unexpected sample format: {sample}")
        
        # 기존 collate 호출 (sparse_logits 제외)
        model_batch, no_model_batch = super().collate(base_samples)
        
        if model_batch is None:
            return None, None
        
        # Sparse logits batching (이미 위에서 추출됨)
        valid_sparse_logits = [sl for sl in sparse_logits_list if sl is not None]
        
        if len(valid_sparse_logits) > 0:
            # 배치 내 최대 시퀀스 길이
            max_seq_len = max(sl['seq_len'] for sl in valid_sparse_logits)
            max_seq_len = min(max_seq_len, self.max_length)
            
            # 배치 내 최대 K (token_ids의 두 번째 차원)
            max_k = max(sl['token_ids'].shape[1] for sl in valid_sparse_logits)
            
            bs = len(samples)
            batch_sparse_logits = {
                'token_ids': np.full((bs, max_seq_len, max_k), -1, dtype=np.int32),
                'values': np.zeros((bs, max_seq_len, max_k), dtype=np.float32),
                'seq_lens': np.array([sl['seq_len'] if sl is not None else 0 for sl in sparse_logits_list], dtype=np.int32),
                'method': valid_sparse_logits[0]['method'],
                'valid_mask': np.array([sl is not None for sl in sparse_logits_list], dtype=bool),
            }
            
            # Random sampling의 경우
            if self.logits_method == 'random':
                batch_sparse_logits['lengths'] = np.zeros((bs, max_seq_len), dtype=np.int16)
                batch_sparse_logits['num_samples'] = valid_sparse_logits[0]['num_samples']
            else:
                batch_sparse_logits['k'] = valid_sparse_logits[0]['k']
            
            # 각 샘플의 sparse logits를 배치에 맞게 패딩
            for i, sparse_logits in enumerate(sparse_logits_list):
                if sparse_logits is None:
                    continue
                
                seq_len = min(sparse_logits['seq_len'], max_seq_len)
                k = sparse_logits['token_ids'].shape[1]
                
                # Truncate and pad
                batch_sparse_logits['token_ids'][i, :seq_len, :k] = sparse_logits['token_ids'][:seq_len, :k]
                batch_sparse_logits['values'][i, :seq_len, :k] = sparse_logits['values'][:seq_len, :k]
                
                if self.logits_method == 'random' and 'lengths' in sparse_logits:
                    batch_sparse_logits['lengths'][i, :seq_len] = sparse_logits['lengths'][:seq_len]
            
            # Convert to torch tensors
            no_model_batch['sparse_logits'] = {
                'token_ids': torch.from_numpy(batch_sparse_logits['token_ids']),
                'values': torch.from_numpy(batch_sparse_logits['values']),
                'seq_lens': torch.from_numpy(batch_sparse_logits['seq_lens']),
                'valid_mask': torch.from_numpy(batch_sparse_logits['valid_mask']),
                'method': batch_sparse_logits['method'],
            }
            
            if self.logits_method == 'random':
                no_model_batch['sparse_logits']['lengths'] = torch.from_numpy(batch_sparse_logits['lengths'])
                no_model_batch['sparse_logits']['num_samples'] = batch_sparse_logits['num_samples']
            else:
                no_model_batch['sparse_logits']['k'] = batch_sparse_logits['k']
        else:
            # No sparse logits available
            no_model_batch['sparse_logits'] = None
        
        return model_batch, no_model_batch

