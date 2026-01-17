"""
병합 최적화 테스트 스크립트

배열 직접 합치기 방식으로 병합 속도를 테스트합니다.

Usage:
    python scripts/merge_shard_fast.py \
        --temp-dir /path/to/.temp_mp_results \
        --shard-id 0 \
        --method both \
        --output-path /path/to/shard_0_output.npz
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.cache_teacher_logits_mp import save_sparse_data


def parse_args():
    parser = argparse.ArgumentParser(description='Fast merge test script')
    
    parser.add_argument('--temp-dir', type=str, required=True,
                        help='Temporary directory with GPU shard files')
    parser.add_argument('--shard-id', type=int, required=True,
                        help='Shard ID to merge')
    parser.add_argument('--method', type=str, required=True,
                        choices=['topk', 'random', 'both'],
                        help='Sampling method')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Output path for merged file')
    parser.add_argument('--num-gpus', type=int, default=8,
                        help='Number of GPUs')
    
    return parser.parse_args()


def merge_fast_way(temp_dir: str, shard_id: int, method: str, num_gpus: int, output_path: str, global_offset: int = 0):
    """
    빠른 병합 방식: 배열 직접 합치기
    """
    print("🚀 Fast merge (array concatenation)")
    start_time = time.time()
    
    # GPU 파일들 로드
    gpu_data_list = []
    for gid in tqdm(range(num_gpus), desc="Loading GPU files"):
        temp_file_gpu = os.path.join(temp_dir, f'shard_{shard_id}_gpu_{gid}.npz')
        if os.path.exists(temp_file_gpu):
            data = np.load(temp_file_gpu, allow_pickle=True)
            gpu_data_list.append({key: data[key] for key in data.keys()})
        else:
            print(f"⚠️  File not found: {temp_file_gpu}")
    
    if not gpu_data_list:
        print("❌ No GPU data found!")
        return
    
    load_time = time.time() - start_time
    print(f"   Load time: {load_time:.2f}s")
    
    # 배열 직접 합치기
    merge_start = time.time()
    
    if method == 'both':
        print("   Concatenating arrays...")
        with tqdm(total=6, desc="   Concatenating", leave=False) as pbar:
            # Top-K 배열 합치기
            all_topk_token_ids = np.concatenate([d['topk_token_ids'] for d in gpu_data_list], axis=0)
            pbar.update(1)
            all_topk_probs = np.concatenate([d['topk_probs'] for d in gpu_data_list], axis=0)
            pbar.update(1)
            
            # Sparse 배열 합치기
            all_sparse_token_ids = np.concatenate([d['sparse_token_ids'] for d in gpu_data_list], axis=0)
            pbar.update(1)
            all_sparse_counts = np.concatenate([d['sparse_counts'] for d in gpu_data_list], axis=0)
            pbar.update(1)
            all_sparse_lengths = np.concatenate([d['sparse_lengths'] for d in gpu_data_list], axis=0)
            pbar.update(1)
            
            # 공통 배열 합치기
            all_seq_lens = np.concatenate([d['seq_lens'] for d in gpu_data_list], axis=0)
            all_local_indices = np.concatenate([d['local_indices'] for d in gpu_data_list], axis=0)
            all_global_indices = np.concatenate([d['global_indices'] for d in gpu_data_list], axis=0)
            pbar.update(1)
        
        concat_time = time.time() - merge_start
        print(f"   Concatenate time: {concat_time:.2f}s")
        
        # local_idx로 정렬 (벡터화)
        print("   Sorting by local_idx...")
        sort_start = time.time()
        with tqdm(total=8, desc="   Sorting arrays", leave=False) as pbar:
            sort_idx = np.argsort(all_local_indices)
            pbar.update(1)
            all_topk_token_ids = all_topk_token_ids[sort_idx]
            pbar.update(1)
            all_topk_probs = all_topk_probs[sort_idx]
            pbar.update(1)
            all_sparse_token_ids = all_sparse_token_ids[sort_idx]
            pbar.update(1)
            all_sparse_counts = all_sparse_counts[sort_idx]
            pbar.update(1)
            all_sparse_lengths = all_sparse_lengths[sort_idx]
            pbar.update(1)
            all_seq_lens = all_seq_lens[sort_idx]
            pbar.update(1)
            all_local_indices = all_local_indices[sort_idx]
            all_global_indices = all_global_indices[sort_idx]
            pbar.update(1)
        sort_time = time.time() - sort_start
        print(f"   Sort time: {sort_time:.2f}s")
        
        # 딕셔너리로 변환
        print("   Converting to dictionaries...")
        dict_start = time.time()
        num_sequences = len(all_seq_lens)
        all_sparse_data_merged = []
        for i in tqdm(range(num_sequences), desc="   Converting to dict"):
            all_sparse_data_merged.append({
                'topk': {
                    'token_ids': all_topk_token_ids[i],
                    'probs': all_topk_probs[i],
                    'k': int(gpu_data_list[0]['topk_k'])
                },
                'sparse': {
                    'token_ids': all_sparse_token_ids[i],
                    'counts': all_sparse_counts[i],
                    'lengths': all_sparse_lengths[i],
                    'num_samples': int(gpu_data_list[0]['sparse_num_samples'])
                },
                'local_idx': int(all_local_indices[i]),
                'global_idx': int(all_global_indices[i]),
                'shard_id': shard_id,
                'seq_len': int(all_seq_lens[i]),
                'method': 'both'
            })
        dict_time = time.time() - dict_start
        print(f"   Dict conversion time: {dict_time:.2f}s")
        
        # 메모리 해제
        del gpu_data_list, all_topk_token_ids, all_topk_probs
        del all_sparse_token_ids, all_sparse_counts, all_sparse_lengths
        del all_seq_lens, all_local_indices, all_global_indices, sort_idx
    
    else:
        # topk 또는 random 메서드
        print("   Concatenating arrays...")
        with tqdm(total=5 if method == 'random' else 4, desc="   Concatenating", leave=False) as pbar:
            all_token_ids = np.concatenate([d['token_ids'] for d in gpu_data_list], axis=0)
            pbar.update(1)
            all_values = np.concatenate([d['values'] for d in gpu_data_list], axis=0)
            pbar.update(1)
            all_seq_lens = np.concatenate([d['seq_lens'] for d in gpu_data_list], axis=0)
            pbar.update(1)
            all_local_indices = np.concatenate([d['local_indices'] for d in gpu_data_list], axis=0)
            pbar.update(1)
            all_global_indices = np.concatenate([d['global_indices'] for d in gpu_data_list], axis=0)
            pbar.update(1)
            
            if method == 'random':
                all_lengths = np.concatenate([d['lengths'] for d in gpu_data_list], axis=0)
        
        concat_time = time.time() - merge_start
        print(f"   Concatenate time: {concat_time:.2f}s")
        
        # local_idx로 정렬
        print("   Sorting by local_idx...")
        sort_start = time.time()
        with tqdm(total=5 if method == 'random' else 4, desc="   Sorting arrays", leave=False) as pbar:
            sort_idx = np.argsort(all_local_indices)
            pbar.update(1)
            all_token_ids = all_token_ids[sort_idx]
            pbar.update(1)
            all_values = all_values[sort_idx]
            pbar.update(1)
            all_seq_lens = all_seq_lens[sort_idx]
            pbar.update(1)
            all_local_indices = all_local_indices[sort_idx]
            all_global_indices = all_global_indices[sort_idx]
            pbar.update(1)
            if method == 'random':
                all_lengths = all_lengths[sort_idx]
        sort_time = time.time() - sort_start
        print(f"   Sort time: {sort_time:.2f}s")
        
        # 딕셔너리로 변환
        print("   Converting to dictionaries...")
        dict_start = time.time()
        num_sequences = len(all_seq_lens)
        all_sparse_data_merged = []
        for i in tqdm(range(num_sequences), desc="   Converting to dict"):
            data = {
                'token_ids': all_token_ids[i],
                'local_idx': int(all_local_indices[i]),
                'global_idx': int(all_global_indices[i]),
                'shard_id': shard_id,
                'seq_len': int(all_seq_lens[i]),
            }
            if method == 'random':
                data['counts'] = all_values[i]
                data['lengths'] = all_lengths[i]
                data['num_samples'] = int(gpu_data_list[0]['num_samples'])
            else:
                data['probs'] = all_values[i]
                data['k'] = int(gpu_data_list[0]['k'])
            all_sparse_data_merged.append(data)
        dict_time = time.time() - dict_start
        print(f"   Dict conversion time: {dict_time:.2f}s")
        
        # 메모리 해제
        del gpu_data_list, all_token_ids, all_values, all_seq_lens
        del all_local_indices, all_global_indices, sort_idx
        if method == 'random':
            del all_lengths
    
    merge_time = time.time() - merge_start
    print(f"   Total merge time: {merge_time:.2f}s")
    
    # 최종 파일 저장
    print("   Saving final file...")
    save_start = time.time()
    with tqdm(total=1, desc="   Saving to disk") as pbar:
        save_sparse_data(all_sparse_data_merged, output_path, method, shard_id, global_offset)
        pbar.update(1)
    save_time = time.time() - save_start
    print(f"   Save time: {save_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"\n✅ Fast merge completed!")
    print(f"   Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"   Sequences: {len(all_sparse_data_merged):,}")
    print(f"   Output: {output_path}")




def main():
    args = parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # 빠른 방식으로 병합
    merge_fast_way(args.temp_dir, args.shard_id, args.method, args.num_gpus, args.output_path, global_offset=0)


if __name__ == '__main__':
    main()
