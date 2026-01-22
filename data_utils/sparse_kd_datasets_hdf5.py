"""
Sparse KD Dataset with HDF5 (flat + offsets + token_offsets)

구조:
    /topk/
        token_ids_flat, probs_flat, seq_offsets, seq_lens, k
    /sparse/
        token_ids_flat, counts_flat, seq_offsets (토큰 idx), token_offsets (element), num_samples
    /meta/
        seq_lens

TopK 읽기 O(1):
    s = topk/seq_lens[i]
    K = topk.attrs['k']
    start, end = seq_offsets[i], seq_offsets[i+1]
    x = flat[start:end].reshape(s, K)

Sparse 읽기 O(1) per token:
    tok_idx = seq_offsets[i] + t
    elem_start, elem_end = token_offsets[tok_idx], token_offsets[tok_idx+1]
    x = flat[elem_start:elem_end]
"""

import os
import json
import numpy as np
import h5py
import torch
from glob import glob
from typing import Dict, Optional, Tuple
from collections import OrderedDict

from utils import print_rank
from .lm_datasets import LMDataset


class SparseKDLMDatasetHDF5(LMDataset):
    """HDF5 (v3: token_offsets) 기반 Dataset"""
    
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
        super().__init__(args, tokenizer, split, data_path, num, ada_max_length, **kwargs)
        
        self.cached_logits_dir = cached_logits_dir
        # LRU cache: 최대 1개 shard만 메모리에 유지 (HDF5 파일 핸들)
        # HDF5는 메모리 맵을 사용하므로 실제 데이터는 디스크에서 읽지만,
        # 파일 핸들을 제한하여 메모리 사용량을 관리
        self.hdf5_files = OrderedDict()  # shard_id -> h5py.File (LRU)
        self.max_cache_size = 1  # 최대 1개 shard 파일만 열어둠
        self.sparse_logits_loaded = False
        
        if cached_logits_dir is not None:
            self._load_hdf5_metadata()
    
    def _load_hdf5_metadata(self):
        metadata_path = os.path.join(self.cached_logits_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.logits_metadata = json.load(f)
        
        self.logits_method = self.logits_metadata['method']
        self.shard_sizes = self.logits_metadata['shard_sizes']
        self.shard_offsets = self.logits_metadata['shard_offsets']
        
        if self.logits_method == 'both':
            self.use_method = getattr(self.args, 'kd_method', 'topk')
            if self.use_method not in ['topk', 'sparse']:
                self.use_method = 'topk'
            print_rank(f"   ✅ Using '{self.use_method}'")
        else:
            self.use_method = self.logits_method
        
        # data_*.h5 또는 shard_*.h5 패턴 지원
        h5_files = sorted(glob(os.path.join(self.cached_logits_dir, 'data_*.h5')))
        file_pattern = 'data_'
        if not h5_files:
            h5_files = sorted(glob(os.path.join(self.cached_logits_dir, 'shard_*.h5')))
            file_pattern = 'shard_'
        if not h5_files:
            raise FileNotFoundError(f"No HDF5 in {self.cached_logits_dir}")
        
        self.shard_paths = {
            int(os.path.basename(p).replace(file_pattern, '').replace('.h5', '')): p 
            for p in h5_files
        }
        
        print_rank(f"✅ HDF5 v3 loaded: {len(self.shard_paths)} shards")
        self.sparse_logits_loaded = True
    
    def _get_shard_and_local_idx(self, global_idx: int) -> Tuple[int, int]:
        left, right = 0, len(self.shard_offsets) - 1
        shard_id = 0
        
        while left <= right:
            mid = (left + right) // 2
            offset = self.shard_offsets[mid]
            next_offset = (self.shard_offsets[mid + 1] 
                          if mid + 1 < len(self.shard_offsets) 
                          else offset + self.shard_sizes[mid])
            
            if offset <= global_idx < next_offset:
                shard_id = mid
                break
            elif global_idx < offset:
                right = mid - 1
            else:
                left = mid + 1
                shard_id = mid + 1
        
        if shard_id >= len(self.shard_offsets):
            shard_id = len(self.shard_offsets) - 1
        
        return shard_id, global_idx - self.shard_offsets[shard_id]
    
    def _get_hdf5_file(self, shard_id: int) -> h5py.File:
        # LRU 캐시: 이미 열려있으면 맨 뒤로 이동 (최근 사용)
        if shard_id in self.hdf5_files:
            self.hdf5_files.move_to_end(shard_id)
            return self.hdf5_files[shard_id]
        
        # 캐시가 가득 찼으면 가장 오래된 파일 닫기
        if len(self.hdf5_files) >= self.max_cache_size:
            oldest_shard_id, oldest_file = self.hdf5_files.popitem(last=False)
            try:
                oldest_file.close()
            except:
                pass
        
        # 새 파일 열기
        new_file = h5py.File(self.shard_paths[shard_id], 'r')
        self.hdf5_files[shard_id] = new_file
        return new_file
    
    def _read_sparse_logits(self, shard_id: int, local_idx: int) -> Dict:
        f = self._get_hdf5_file(shard_id)
        
        if self.use_method == 'topk':
            # TopK: O(1)
            s = int(f['topk/seq_lens'][local_idx])
            K = int(f['topk'].attrs['k'])
            
            offsets = f['topk/seq_offsets']
            start, end = int(offsets[local_idx]), int(offsets[local_idx + 1])
            
            if s == 0 or start == end:
                return {
                    'token_ids': np.zeros((0, K), dtype=np.int32),
                    'values': np.zeros((0, K), dtype=np.float32),
                    'seq_len': 0,
                    'method': 'topk',
                    'k': K,
                }
            
            ids = f['topk/token_ids_flat'][start:end]
            probs = f['topk/probs_flat'][start:end]
            
            return {
                'token_ids': ids.reshape(s, K),
                'values': probs.reshape(s, K).astype(np.float32),
                'seq_len': s,
                'method': 'topk',
                'k': K,
            }
        
        else:  # sparse
            # Sparse: O(1) per token via token_offsets
            seq_offsets = f['sparse/seq_offsets']
            token_offsets = f['sparse/token_offsets']
            num_samples = int(f['sparse'].attrs['num_samples'])
            
            # 시퀀스의 토큰 범위
            tok_start = int(seq_offsets[local_idx])
            tok_end = int(seq_offsets[local_idx + 1])
            seq_len = tok_end - tok_start
            
            if seq_len == 0:
                return {
                    'token_ids': np.zeros((0, 1), dtype=np.int32),
                    'values': np.zeros((0, 1), dtype=np.float32),
                    'seq_len': 0,
                    'method': 'random',
                    'lengths': np.array([], dtype=np.int16),
                    'num_samples': num_samples,
                }
            
            # 토큰별 element 범위
            tok_offs = token_offsets[tok_start:tok_end + 1]
            elem_start = int(tok_offs[0])
            elem_end = int(tok_offs[-1])
            
            # 전체 읽기
            ids_all = f['sparse/token_ids_flat'][elem_start:elem_end]
            counts_all = f['sparse/counts_flat'][elem_start:elem_end]
            
            # 토큰별 lengths
            lengths = np.diff(tok_offs).astype(np.int16)
            max_k = int(np.max(lengths)) if len(lengths) > 0 else 1
            
            # 패딩
            token_ids = np.full((seq_len, max_k), -1, dtype=np.int32)
            values = np.zeros((seq_len, max_k), dtype=np.float32)
            
            pos = 0
            for t in range(seq_len):
                k_t = int(lengths[t])
                if k_t > 0:
                    token_ids[t, :k_t] = ids_all[pos:pos + k_t]
                    values[t, :k_t] = counts_all[pos:pos + k_t]
                    pos += k_t
            
            return {
                'token_ids': token_ids,
                'values': values,
                'seq_len': seq_len,
                'method': 'random',
                'lengths': lengths,
                'num_samples': num_samples,
            }
    
    def __getitem__(self, index: int):
        result = super().__getitem__(index)
        if result is None:
            return None
        
        idx, data = result
        
        sparse_logits = None
        if self.sparse_logits_loaded:
            try:
                shard_id, local_idx = self._get_shard_and_local_idx(idx)
                sparse_logits = self._read_sparse_logits(shard_id, local_idx)
            except Exception as e:
                print_rank(f"⚠️ idx={idx}: {e}")
        
        return idx, data, sparse_logits
    
    def collate(self, samples):
        if samples[0] is None:
            return None, None
        
        base_samples = []
        sparse_list = []
        
        for s in samples:
            if len(s) == 3:
                idx, data, sp = s
                base_samples.append((idx, data))
                sparse_list.append(sp)
            else:
                base_samples.append((s[0], s[1]))
                sparse_list.append(None)
        
        model_batch, no_model_batch = super().collate(base_samples)
        if model_batch is None:
            return None, None
        
        valid = [s for s in sparse_list if s is not None]
        
        if valid:
            # model_batch의 실제 길이 확인 (텍스트 데이터는 max_length로 패딩됨)
            # sparse_logits도 이 길이에 맞춰서 패딩/트렁케이션해야 함
            model_seq_len = model_batch['input_ids'].shape[1] if 'input_ids' in model_batch else self.max_length
            
            # model_batch 길이에 맞춤 (sparse_logits가 짧으면 패딩, 길면 자름)
            max_seq = model_seq_len
            max_k = max(s['token_ids'].shape[1] for s in valid) if valid else 1
            bs = len(samples)
            method = valid[0]['method']
            
            batch = {
                'token_ids': np.full((bs, max_seq, max_k), -1, dtype=np.int32),
                'values': np.zeros((bs, max_seq, max_k), dtype=np.float32),
                'seq_lens': np.array([s['seq_len'] if s else 0 for s in sparse_list], dtype=np.int32),
                'valid_mask': np.array([s is not None for s in sparse_list], dtype=bool),
            }
            
            if method == 'random':
                batch['lengths'] = np.zeros((bs, max_seq), dtype=np.int16)
                batch['num_samples'] = valid[0]['num_samples']
            else:
                batch['k'] = valid[0]['k']
            
            for i, sp in enumerate(sparse_list):
                if sp is None:
                    continue
                sl = min(sp['seq_len'], max_seq)
                k = sp['token_ids'].shape[1]
                batch['token_ids'][i, :sl, :k] = sp['token_ids'][:sl]
                batch['values'][i, :sl, :k] = sp['values'][:sl]
                if method == 'random' and 'lengths' in sp:
                    batch['lengths'][i, :sl] = sp['lengths'][:sl]
            
            no_model_batch['sparse_logits'] = {
                'token_ids': torch.from_numpy(batch['token_ids']),
                'values': torch.from_numpy(batch['values']),
                'seq_lens': torch.from_numpy(batch['seq_lens']),
                'valid_mask': torch.from_numpy(batch['valid_mask']),
                'method': method,
            }
            if method == 'random':
                no_model_batch['sparse_logits']['lengths'] = torch.from_numpy(batch['lengths'])
                no_model_batch['sparse_logits']['num_samples'] = batch['num_samples']
            else:
                no_model_batch['sparse_logits']['k'] = batch['k']
        else:
            no_model_batch['sparse_logits'] = None
        
        return model_batch, no_model_batch
    
    def __del__(self):
        for f in self.hdf5_files.values():
            try:
                f.close()
            except:
                pass
