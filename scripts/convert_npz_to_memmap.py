#!/usr/bin/env python3
"""
NPZ (object array) → Memmap 변환 스크립트

기존 npz 파일의 object array를 고정 크기 memmap 파일로 변환합니다.
이를 통해 여러 프로세스가 메모리를 공유할 수 있습니다.

사용법:
    python convert_npz_to_memmap.py \
        --input-dir /path/to/logits_both \
        --output-dir /path/to/logits_both_memmap \
        --max-seq-len 1024 \
        --max-k 100

메모리 요구량: shard당 ~50GB (변환 중)
예상 시간: shard당 10-30분
"""

import os
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import json
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True,
                        help='기존 npz 디렉토리')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='memmap 출력 디렉토리')
    parser.add_argument('--max-seq-len', type=int, default=1024,
                        help='최대 시퀀스 길이 (패딩)')
    parser.add_argument('--max-k', type=int, default=100,
                        help='최대 K (topk/sparse)')
    parser.add_argument('--start-shard', type=int, default=0)
    parser.add_argument('--end-shard', type=int, default=None)
    return parser.parse_args()


def convert_shard(npz_path: str, output_dir: str, max_seq_len: int, max_k: int):
    """
    하나의 npz shard를 memmap 파일들로 변환
    
    출력 파일 구조:
        shard_X/
            topk_token_ids.npy   # [num_seq, max_seq_len, max_k], int32
            topk_probs.npy       # [num_seq, max_seq_len, max_k], float16
            sparse_token_ids.npy # [num_seq, max_seq_len, max_k], int32
            sparse_counts.npy    # [num_seq, max_seq_len, max_k], int16
            sparse_lengths.npy   # [num_seq, max_seq_len], int16
            seq_lens.npy         # [num_seq], int32
            metadata.json
    """
    # 파일명에서 shard_id 추출
    basename = os.path.basename(npz_path)
    shard_id = int(basename.replace('data_', '').replace('.npz', ''))
    
    shard_output_dir = os.path.join(output_dir, f'shard_{shard_id}')
    os.makedirs(shard_output_dir, exist_ok=True)
    
    print(f"\n📂 Converting shard {shard_id}: {npz_path}")
    
    # NPZ 로드
    data = np.load(npz_path, allow_pickle=True)
    
    method = str(data['method'])
    num_seq = len(data['seq_lens'])
    seq_lens = data['seq_lens']
    
    print(f"   Method: {method}, Sequences: {num_seq:,}")
    
    # Memmap 파일 생성 및 데이터 복사
    if method == 'both':
        topk_k = int(data['topk_k'])
        sparse_num_samples = int(data['sparse_num_samples'])
        
        # 배열 shape 결정
        actual_max_k = min(max(topk_k, sparse_num_samples), max_k)
        
        # Memmap 생성
        topk_token_ids_mm = np.memmap(
            os.path.join(shard_output_dir, 'topk_token_ids.npy'),
            dtype=np.int32, mode='w+', shape=(num_seq, max_seq_len, actual_max_k)
        )
        topk_probs_mm = np.memmap(
            os.path.join(shard_output_dir, 'topk_probs.npy'),
            dtype=np.float16, mode='w+', shape=(num_seq, max_seq_len, actual_max_k)
        )
        sparse_token_ids_mm = np.memmap(
            os.path.join(shard_output_dir, 'sparse_token_ids.npy'),
            dtype=np.int32, mode='w+', shape=(num_seq, max_seq_len, actual_max_k)
        )
        sparse_counts_mm = np.memmap(
            os.path.join(shard_output_dir, 'sparse_counts.npy'),
            dtype=np.int16, mode='w+', shape=(num_seq, max_seq_len, actual_max_k)
        )
        sparse_lengths_mm = np.memmap(
            os.path.join(shard_output_dir, 'sparse_lengths.npy'),
            dtype=np.int16, mode='w+', shape=(num_seq, max_seq_len)
        )
        seq_lens_mm = np.memmap(
            os.path.join(shard_output_dir, 'seq_lens.npy'),
            dtype=np.int32, mode='w+', shape=(num_seq,)
        )
        
        # 초기화 (-1로 패딩)
        topk_token_ids_mm[:] = -1
        topk_probs_mm[:] = 0
        sparse_token_ids_mm[:] = -1
        sparse_counts_mm[:] = 0
        sparse_lengths_mm[:] = 0
        
        # 데이터 복사
        topk_token_ids_obj = data['topk_token_ids']
        topk_probs_obj = data['topk_probs']
        sparse_token_ids_obj = data['sparse_token_ids']
        sparse_counts_obj = data['sparse_counts']
        sparse_lengths_obj = data['sparse_lengths']
        
        for i in tqdm(range(num_seq), desc=f"Shard {shard_id}", leave=False):
            seq_len = min(int(seq_lens[i]), max_seq_len)
            seq_lens_mm[i] = seq_len
            
            # TopK
            tk_ids = topk_token_ids_obj[i]
            tk_probs = topk_probs_obj[i]
            k_topk = min(tk_ids.shape[1] if tk_ids.ndim > 1 else len(tk_ids), actual_max_k)
            s_topk = min(tk_ids.shape[0] if tk_ids.ndim > 1 else 1, seq_len)
            
            if tk_ids.ndim > 1:
                topk_token_ids_mm[i, :s_topk, :k_topk] = tk_ids[:s_topk, :k_topk]
                topk_probs_mm[i, :s_topk, :k_topk] = tk_probs[:s_topk, :k_topk].astype(np.float16)
            
            # Sparse
            sp_ids = sparse_token_ids_obj[i]
            sp_counts = sparse_counts_obj[i]
            sp_lens = sparse_lengths_obj[i]
            k_sparse = min(sp_ids.shape[1] if sp_ids.ndim > 1 else len(sp_ids), actual_max_k)
            s_sparse = min(sp_ids.shape[0] if sp_ids.ndim > 1 else 1, seq_len)
            
            if sp_ids.ndim > 1:
                sparse_token_ids_mm[i, :s_sparse, :k_sparse] = sp_ids[:s_sparse, :k_sparse]
                sparse_counts_mm[i, :s_sparse, :k_sparse] = sp_counts[:s_sparse, :k_sparse].astype(np.int16)
            sparse_lengths_mm[i, :s_sparse] = sp_lens[:s_sparse].astype(np.int16)
        
        # Flush
        topk_token_ids_mm.flush()
        topk_probs_mm.flush()
        sparse_token_ids_mm.flush()
        sparse_counts_mm.flush()
        sparse_lengths_mm.flush()
        seq_lens_mm.flush()
        
        del topk_token_ids_mm, topk_probs_mm, sparse_token_ids_mm, sparse_counts_mm, sparse_lengths_mm, seq_lens_mm
        
        # Metadata 저장
        metadata = {
            'method': 'both',
            'num_sequences': num_seq,
            'max_seq_len': max_seq_len,
            'max_k': actual_max_k,
            'topk_k': topk_k,
            'sparse_num_samples': sparse_num_samples,
            'shard_id': shard_id,
        }
    else:
        raise NotImplementedError(f"Method '{method}' not yet supported for conversion")
    
    # Metadata 저장
    with open(os.path.join(shard_output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 메모리 정리
    del data
    
    return num_seq


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 입력 npz 파일 찾기
    npz_files = sorted(glob(os.path.join(args.input_dir, 'data_*.npz')))
    
    if not npz_files:
        print(f"❌ No npz files found in {args.input_dir}")
        return
    
    print(f"🔍 Found {len(npz_files)} npz files")
    
    # Shard ID 추출 및 필터링
    def get_shard_id(path):
        return int(os.path.basename(path).replace('data_', '').replace('.npz', ''))
    
    shard_files = [(get_shard_id(p), p) for p in npz_files]
    shard_files.sort(key=lambda x: x[0])
    
    if args.end_shard is not None:
        shard_files = [(sid, p) for sid, p in shard_files if args.start_shard <= sid < args.end_shard]
    else:
        shard_files = [(sid, p) for sid, p in shard_files if sid >= args.start_shard]
    
    print(f"📦 Converting shards {args.start_shard} to {args.end_shard or 'end'}")
    print(f"   Max seq len: {args.max_seq_len}, Max K: {args.max_k}")
    
    # 기존 metadata 복사
    input_metadata_path = os.path.join(args.input_dir, 'metadata.json')
    if os.path.exists(input_metadata_path):
        shutil.copy(input_metadata_path, os.path.join(args.output_dir, 'metadata.json'))
        print(f"   Copied metadata.json")
    
    # 변환 시작
    total_sequences = 0
    for shard_id, npz_path in shard_files:
        num_seq = convert_shard(npz_path, args.output_dir, args.max_seq_len, args.max_k)
        total_sequences += num_seq
    
    print(f"\n✅ Conversion complete!")
    print(f"   Total sequences: {total_sequences:,}")
    print(f"   Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
