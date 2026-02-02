"""
Teacher Logits Caching Script - Multi-GPU Parallel Version (Queue-based)

각 GPU worker가 큐에서 shard를 하나씩 가져와서 처리합니다.
- 메모리 효율적: shard 완료 후 즉시 메모리 해제
- 동적 로드 밸런싱: 빠른 GPU가 더 많은 shard 처리
- RAM 사용량 안정적: 여러 shard 결과를 메모리에 누적하지 않음

8개 GPU면 이론상 8배 속도 향상.

Usage:
    python scripts/cache_teacher_logits_mp.py \
        --teacher-model-path /path/to/teacher \
        --data-dir /data/jykim/DB/miniplm_refined_corpus \
        --output-dir /data/jykim/DB/miniplm_refined_corpus_logits_both \
        --method both --topk 100 --num-samples 50 \
        --num-gpus 8
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from glob import glob
from typing import List, Dict, Tuple, Optional
from queue import Empty
from concurrent.futures import ThreadPoolExecutor
import time
import gc

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils.indexed_dataset import MMapIndexedDataset
from data_utils.sparse_sampler import SparseLogitSampler, TopKSampler


def parse_args():
    parser = argparse.ArgumentParser(description='Cache teacher logits (Multi-GPU parallel)')
    
    # Model
    parser.add_argument('--teacher-model-path', type=str, required=True,
                        help='Path to teacher model')
    parser.add_argument('--model-type', type=str, default='qwen',
                        choices=['qwen', 'llama', 'gpt2'],
                        help='Model type')
    
    # Data
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing data_*.bin/idx files')
    parser.add_argument('--data-prefix', type=str, default='data',
                        help='Prefix of data files (default: data)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for cached logits')
    
    # Sampling method
    parser.add_argument('--method', type=str, default='topk',
                        choices=['topk', 'random', 'both'],
                        help='Sampling method: topk, random, or both')
    parser.add_argument('--topk', type=int, default=50,
                        help='K for top-k sampling')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='N for random sampling')
    
    # Processing
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for teacher inference (per GPU)')
    parser.add_argument('--max-length', type=int, default=1024,
                        help='Maximum sequence length')
    parser.add_argument('--start-shard', type=int, default=0,
                        help='Starting shard index')
    parser.add_argument('--end-shard', type=int, default=-1,
                        help='Ending shard index (-1 for all)')
    parser.add_argument('--dtype', type=str, default='bf16',
                        choices=['fp32', 'fp16', 'bf16'],
                        help='Model dtype')
    
    # Multi-GPU
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='Number of GPUs to use (default: all available)')
    
    return parser.parse_args()


def get_shard_paths(data_dir: str, prefix: str = 'data') -> List[str]:
    """데이터 디렉토리에서 모든 shard 경로 찾기"""
    bin_files = sorted(glob(os.path.join(data_dir, f'{prefix}_*.bin')))
    
    shard_paths = []
    for bin_file in bin_files:
        path = bin_file[:-4]
        if os.path.exists(path + '.idx'):
            shard_paths.append(path)
    
    def get_shard_num(path):
        basename = os.path.basename(path)
        name_no_ext = basename.rsplit('.', 1)[0] if '.' in basename else basename
        try:
            return int(name_no_ext.split('_')[-1])
        except ValueError:
            return 0
    
    shard_paths.sort(key=get_shard_num)
    return shard_paths


def get_shard_sizes(shard_paths: List[str]) -> List[int]:
    """각 shard의 크기 계산"""
    sizes = []
    for path in tqdm(shard_paths, desc="Probing shard sizes"):
        dataset = MMapIndexedDataset(path, skip_warmup=True)
        sizes.append(len(dataset))
        del dataset
    return sizes


def load_or_compute_shard_info(output_dir: str, all_shard_paths: List[str]) -> Tuple[List[int], List[int], int]:
    """Shard 크기 정보를 캐시에서 로드하거나 계산"""
    metadata_path = os.path.join(output_dir, 'metadata.json')
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                cached_meta = json.load(f)
            
            if (cached_meta.get('total_shards') == len(all_shard_paths) and 
                'shard_sizes' in cached_meta and 
                'shard_offsets' in cached_meta):
                
                all_shard_sizes = cached_meta['shard_sizes']
                global_offsets = cached_meta['shard_offsets']
                total_sequences = cached_meta['total_sequences']
                print(f"✅ Loaded shard info from cached metadata.json")
                return all_shard_sizes, global_offsets, total_sequences
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Could not load cached metadata: {e}")
    
    print("Computing shard sizes...")
    all_shard_sizes = get_shard_sizes(all_shard_paths)
    
    global_offsets = [0]
    for size in all_shard_sizes[:-1]:
        global_offsets.append(global_offsets[-1] + size)
    
    total_sequences = sum(all_shard_sizes)
    
    return all_shard_sizes, global_offsets, total_sequences


def save_sparse_data(sparse_data_list: list, output_path: str, method: str, shard_id: int, global_offset: int):
    """Sparse data를 npz 형식으로 저장 (bin 순서대로 저장됨 → 같은 인덱스 = 같은 시퀀스)"""
    all_seq_lens = []
    all_local_indices = []
    all_global_indices = []

    if method == 'both':
        all_topk_token_ids = []
        all_topk_probs = []
        all_sparse_token_ids = []
        all_sparse_counts = []
        all_sparse_lengths = []

        for data in sparse_data_list:
            all_seq_lens.append(data['seq_len'])
            all_local_indices.append(data['local_idx'])
            all_global_indices.append(data['global_idx'])

            all_topk_token_ids.append(data['topk']['token_ids'])
            all_topk_probs.append(data['topk']['probs'])

            all_sparse_token_ids.append(data['sparse']['token_ids'])
            all_sparse_counts.append(data['sparse']['counts'])
            all_sparse_lengths.append(data['sparse']['lengths'])

        save_dict = {
            'topk_token_ids': np.array(all_topk_token_ids, dtype=object),
            'topk_probs': np.array(all_topk_probs, dtype=object),
            'topk_k': np.int16(sparse_data_list[0]['topk']['k']),
            'sparse_token_ids': np.array(all_sparse_token_ids, dtype=object),
            'sparse_counts': np.array(all_sparse_counts, dtype=object),
            'sparse_lengths': np.array(all_sparse_lengths, dtype=object),
            'sparse_num_samples': np.int16(sparse_data_list[0]['sparse']['num_samples']),
            'seq_lens': np.array(all_seq_lens, dtype=np.int32),
            'local_indices': np.array(all_local_indices, dtype=np.int32),
            'global_indices': np.array(all_global_indices, dtype=np.int64),
            'shard_id': np.int32(shard_id),
            'global_offset': np.int64(global_offset),
            'method': np.array('both'),
        }
    else:
        all_token_ids = []
        all_values = []
        all_lengths = []
        
        for data in sparse_data_list:
            all_token_ids.append(data['token_ids'])
            all_seq_lens.append(data['seq_len'])
            all_local_indices.append(data['local_idx'])
            all_global_indices.append(data['global_idx'])

            if method == 'random':
                all_values.append(data['counts'])
                all_lengths.append(data['lengths'])
            else:
                all_values.append(data['probs'])
        
        save_dict = {
            'token_ids': np.array(all_token_ids, dtype=object),
            'values': np.array(all_values, dtype=object),
            'seq_lens': np.array(all_seq_lens, dtype=np.int32),
            'local_indices': np.array(all_local_indices, dtype=np.int32),
            'global_indices': np.array(all_global_indices, dtype=np.int64),
            'shard_id': np.int32(shard_id),
            'global_offset': np.int64(global_offset),
            'method': np.array(method),
        }
        
        if method == 'random':
            save_dict['lengths'] = np.array(all_lengths, dtype=object)
            save_dict['num_samples'] = np.array(sparse_data_list[0]['num_samples'])
        else:
            save_dict['k'] = np.array(sparse_data_list[0]['k'])

    np.savez(output_path, **save_dict)  # 비압축: 5-10배 빠름, 파일 크기 ~3배
    
    # 메모리 해제
    del save_dict
    if method == 'both':
        del all_topk_token_ids, all_topk_probs, all_sparse_token_ids, all_sparse_counts, all_sparse_lengths
    else:
        del all_token_ids, all_values
        if method == 'random':
            del all_lengths
    del all_seq_lens, all_local_indices, all_global_indices


def merge_partial_results(partial_data_list: list, method: str) -> list:
    """여러 GPU의 부분 결과를 합치기"""
    merged_data = []
    
    for partial_data in partial_data_list:
        if method == 'both':
            # both 메서드의 경우
            num_items = len(partial_data['seq_lens'])
            for i in range(num_items):
                merged_data.append({
                    'topk': {
                        'token_ids': partial_data['topk_token_ids'][i],
                        'probs': partial_data['topk_probs'][i],
                        'k': int(partial_data['topk_k'])
                    },
                    'sparse': {
                        'token_ids': partial_data['sparse_token_ids'][i],
                        'counts': partial_data['sparse_counts'][i],
                        'lengths': partial_data['sparse_lengths'][i],
                        'num_samples': int(partial_data['sparse_num_samples'])
                    },
                    'local_idx': int(partial_data['local_indices'][i]),
                    'global_idx': int(partial_data['global_indices'][i]),
                    'shard_id': int(partial_data['shard_id']),
                    'seq_len': int(partial_data['seq_lens'][i]),
                    'method': 'both'
                })
        else:
            # topk 또는 random 메서드
            num_items = len(partial_data['seq_lens'])
            for i in range(num_items):
                data = {
                    'token_ids': partial_data['token_ids'][i],
                    'local_idx': int(partial_data['local_indices'][i]),
                    'global_idx': int(partial_data['global_indices'][i]),
                    'shard_id': int(partial_data['shard_id']),
                    'seq_len': int(partial_data['seq_lens'][i]),
                }
                
                if method == 'random':
                    data['counts'] = partial_data['values'][i]
                    data['lengths'] = partial_data['lengths'][i]
                    data['num_samples'] = int(partial_data['num_samples'])
                else:
                    data['probs'] = partial_data['values'][i]
                    data['k'] = int(partial_data['k'])
                
                merged_data.append(data)
    
    # local_idx 순서로 정렬 (원래 shard 내 순서 유지)
    merged_data.sort(key=lambda x: x['local_idx'])
    
    return merged_data


def worker_fn(
    gpu_id: int,
    shard_list: List[Tuple[int, str, int]],  # 처리할 shard 리스트
    args,
    progress_queue: mp.Queue,
    tokenizer_path: str,
    num_gpus: int,  # 전체 GPU 수
    barrier: mp.Barrier,  # 모든 GPU가 같은 shard를 처리하도록 동기화
    temp_dir: str,  # 임시 파일 저장 디렉토리
):
    """
    단일 GPU에서 shard를 처리하는 worker 함수
    모든 GPU가 같은 shard를 처리하되, 데이터를 분산 처리합니다.
    
    Args:
        gpu_id: 사용할 GPU ID
        shard_list: 처리할 shard 리스트 (shard_id, shard_path, global_offset)
        args: 명령줄 인자
        progress_queue: 진행 상황 보고용 Queue
        tokenizer_path: Teacher 모델 경로 (tokenizer 로드용)
        num_gpus: 전체 GPU 수
        barrier: 모든 GPU 동기화용 Barrier
        temp_dir: 임시 파일 저장 디렉토리 (메모리 절약)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # GPU 설정
    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'
    
    dtype_map = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }
    torch_dtype = dtype_map[args.dtype]
    
    # 모델 로드
    print(f"[GPU {gpu_id}] Loading teacher model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)  # .to(device)가 가장 안전한 방법
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    print(f"[GPU {gpu_id}] Model loaded, ready to process shards from queue")
    
    # Sampler 초기화
    if args.method == 'both':
        sampler_topk = TopKSampler(k=args.topk)
        sampler_sparse = SparseLogitSampler(num_samples=args.num_samples)
    elif args.method == 'topk':
        sampler_topk = TopKSampler(k=args.topk)
        sampler_sparse = None
    else:
        sampler_topk = None
        sampler_sparse = SparseLogitSampler(num_samples=args.num_samples)
    
    # shard 리스트를 순회하며 처리
    for shard_idx, shard_info in enumerate(shard_list):
        shard_id, shard_path, global_offset = shard_info
        shard_name = os.path.basename(shard_path)
        output_path = os.path.join(args.output_dir, f'{shard_name}.npz')
        
        # 모든 GPU가 같은 shard를 처리하도록 동기화
        barrier.wait()
        
        # 이미 존재하면 스킵 (모든 GPU가 체크)
        if os.path.exists(output_path):
            if gpu_id == 0:  # GPU 0만 메시지 전송
                progress_queue.put(('skip', gpu_id, shard_id, shard_name, 0))
            barrier.wait()  # 모든 GPU가 스킵 확인
            continue
        
        # 데이터셋 로드
        dataset = MMapIndexedDataset(shard_path, skip_warmup=True)
        shard_size = len(dataset)
        
        # 데이터 분산: 각 GPU가 처리할 범위 계산
        chunk_size = shard_size // num_gpus
        start_idx = gpu_id * chunk_size
        end_idx = start_idx + chunk_size if gpu_id < num_gpus - 1 else shard_size
        
        if gpu_id == 0:  # GPU 0만 메시지 전송 (오버헤드 감소)
            progress_queue.put(('start', gpu_id, shard_id, shard_name, end_idx - start_idx))
        
        all_sparse_data = []
        batch_input_ids = []
        batch_local_indices = []
        failed_batches = 0
        batch_count = 0
        total_sequences = end_idx - start_idx
        
        # tqdm with GPU position (멀티프로세싱에서 겹치지 않게)
        pbar = tqdm(
            range(start_idx, end_idx), 
            desc=f"[GPU {gpu_id}] {shard_name}",
            position=gpu_id,
            leave=False,
            ncols=100
        )
        
        import time
        batch_start_time = time.time()
        
        for idx in pbar:
            data = dataset[idx].astype(np.int64)
            data = data[:args.max_length + 1]
            
            batch_input_ids.append(data[:-1])
            batch_local_indices.append(idx)
            
            # 배치 처리 조건: 배치 사이즈 도달 또는 마지막 인덱스
            is_last_idx = (idx == end_idx - 1)
            if len(batch_input_ids) >= args.batch_size or is_last_idx:
                batch_count += 1
                actual_batch_size = len(batch_input_ids)
                
                # Pad batch
                max_len = max(len(seq) for seq in batch_input_ids)
                # MAX_LENGTH 제한 적용
                max_len = min(max_len, args.max_length)
                padded_batch = np.full((actual_batch_size, max_len), pad_id, dtype=np.int64)
                attention_mask = np.zeros((actual_batch_size, max_len), dtype=np.int64)
                
                # 각 시퀀스의 실제 유효 길이 저장 (패딩 전 원본 길이, max_len 제한 적용)
                actual_seq_lens = []
                for i, seq in enumerate(batch_input_ids):
                    seq_len = min(len(seq), max_len)
                    actual_seq_lens.append(seq_len)  # 실제 유효 길이 저장
                    padded_batch[i, :seq_len] = seq[:seq_len]
                    attention_mask[i, :seq_len] = 1
                
                input_ids = torch.tensor(padded_batch, device=device, dtype=torch.long)
                attn_mask = torch.tensor(attention_mask, device=device, dtype=torch.long)
                
                # 변수 초기화 (예외 발생 시 안전하게 삭제하기 위해)
                outputs = None
                logits = None
                probs = None
                
                try:
                    with torch.no_grad():
                        inference_start = time.time()
                        outputs = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
                        logits = outputs.logits
                        # float 변환 (안정성)
                        probs = torch.softmax(logits.float(), dim=-1)
                        inference_time = time.time() - inference_start
                        
                        # 첫 배치에서 속도 확인
                        if batch_count == 1 and gpu_id == 0:
                            print(f"\n[GPU {gpu_id}] 첫 배치 처리 시간: {inference_time:.3f}초 (배치 크기: {actual_batch_size})")
                    
                    # 배치 단위 샘플링 (벡터화로 속도 개선)
                    # probs shape: [batch_size, max_len, vocab_size]
                    batch_size, max_len, vocab_size = probs.shape
                    
                    if args.method == 'both':
                        # Top-K: 배치 단위로 처리
                        topk_batch = sampler_topk.sample_batch(probs)  # [batch, seq, K]
                        topk_token_ids = topk_batch['token_ids']  # [batch, seq, K]
                        topk_probs = topk_batch['probs']  # [batch, seq, K]
                        
                        # Random: flatten하여 한번에 처리
                        probs_flat = probs.reshape(-1, vocab_size)  # [batch*seq, vocab]
                        sparse_batch = sampler_sparse.sample_flat(probs_flat)  # [batch*seq, max_unique]
                        sparse_token_ids = sparse_batch['token_ids']  # [batch*seq, max_unique]
                        sparse_counts = sparse_batch['counts']  # [batch*seq, max_unique]
                        sparse_lengths = sparse_batch['lengths']  # [batch*seq]
                        
                        # 각 시퀀스별로 결과 분해
                        for i in range(actual_batch_size):
                            local_idx = batch_local_indices[i]
                            global_idx = global_offset + local_idx
                            actual_len = actual_seq_lens[i]
                            
                            # Top-K: 유효 길이까지만 추출
                            topk_data = {
                                'token_ids': topk_token_ids[i, :actual_len],  # [actual_len, K]
                                'probs': topk_probs[i, :actual_len],  # [actual_len, K]
                                'k': topk_batch['k']
                            }
                            
                            # Sparse: 유효 길이까지만 추출
                            start_idx = i * max_len
                            end_idx = start_idx + actual_len
                            sparse_data = {
                                'token_ids': sparse_token_ids[start_idx:end_idx],  # [actual_len, max_unique]
                                'counts': sparse_counts[start_idx:end_idx],  # [actual_len, max_unique]
                                'lengths': sparse_lengths[start_idx:end_idx],  # [actual_len]
                                'num_samples': sparse_batch['num_samples']
                            }
                            
                            combined_data = {
                                'topk': topk_data,
                                'sparse': sparse_data,
                                'local_idx': local_idx,
                                'global_idx': global_idx,
                                'shard_id': shard_id,
                                'seq_len': actual_len,
                                'method': 'both'
                            }
                            all_sparse_data.append(combined_data)
                    
                    elif args.method == 'topk':
                        # Top-K: 배치 단위로 처리
                        topk_batch = sampler_topk.sample_batch(probs)  # [batch, seq, K]
                        topk_token_ids = topk_batch['token_ids']  # [batch, seq, K]
                        topk_probs = topk_batch['probs']  # [batch, seq, K]
                        
                        # 각 시퀀스별로 결과 분해
                        for i in range(actual_batch_size):
                            local_idx = batch_local_indices[i]
                            global_idx = global_offset + local_idx
                            actual_len = actual_seq_lens[i]
                            
                            sparse_data = {
                                'token_ids': topk_token_ids[i, :actual_len],  # [actual_len, K]
                                'probs': topk_probs[i, :actual_len],  # [actual_len, K]
                                'k': topk_batch['k'],
                                'local_idx': local_idx,
                                'global_idx': global_idx,
                                'shard_id': shard_id,
                                'seq_len': actual_len
                            }
                            all_sparse_data.append(sparse_data)

                    else:  # random
                        # Random: flatten하여 한번에 처리
                        probs_flat = probs.reshape(-1, vocab_size)  # [batch*seq, vocab]
                        sparse_batch = sampler_sparse.sample_flat(probs_flat)  # [batch*seq, max_unique]
                        sparse_token_ids = sparse_batch['token_ids']  # [batch*seq, max_unique]
                        sparse_counts = sparse_batch['counts']  # [batch*seq, max_unique]
                        sparse_lengths = sparse_batch['lengths']  # [batch*seq]
                        
                        # 각 시퀀스별로 결과 분해
                        for i in range(actual_batch_size):
                            local_idx = batch_local_indices[i]
                            global_idx = global_offset + local_idx
                            actual_len = actual_seq_lens[i]
                            
                            start_idx = i * max_len
                            end_idx = start_idx + actual_len
                            
                            sparse_data = {
                                'token_ids': sparse_token_ids[start_idx:end_idx],  # [actual_len, max_unique]
                                'counts': sparse_counts[start_idx:end_idx],  # [actual_len, max_unique]
                                'lengths': sparse_lengths[start_idx:end_idx],  # [actual_len]
                                'num_samples': sparse_batch['num_samples'],
                                'local_idx': local_idx,
                                'global_idx': global_idx,
                                'shard_id': shard_id,
                                'seq_len': actual_len
                            }
                            all_sparse_data.append(sparse_data)
                            
                except RuntimeError as e:
                    failed_batches += 1
                    # 첫 번째 에러는 상세히 출력
                    if failed_batches == 1:
                        print(f"\n[GPU {gpu_id}] ❌ RuntimeError in shard {shard_id}, batch {batch_count}: {e}")
                        import traceback
                        traceback.print_exc()
                    elif failed_batches <= 3:
                        print(f"[GPU {gpu_id}] ❌ RuntimeError #{failed_batches} in shard {shard_id}: {e}")
                    
                    if failed_batches > 10:
                        print(f"[GPU {gpu_id}] ❌ Too many failures ({failed_batches}), stopping shard {shard_id}")
                        break
                finally:
                    # 안전하게 메모리 해제
                    batch_input_ids = []
                    batch_local_indices = []
                    if input_ids is not None:
                        del input_ids
                    if attn_mask is not None:
                        del attn_mask
                    if outputs is not None:
                        del outputs
                    if logits is not None:
                        del logits
                    if probs is not None:
                        del probs
                # 배치마다 empty_cache 호출은 오버헤드가 큼 - 주기적으로만 호출
                # torch.cuda.empty_cache()
                
                # 주기적으로 속도 확인 (100배치마다)
                if batch_count % 100 == 0 and gpu_id == 0:
                    elapsed = time.time() - batch_start_time
                    seqs_processed = batch_count * args.batch_size
                    speed = seqs_processed / elapsed if elapsed > 0 else 0
                    print(f"\n[GPU {gpu_id}] 배치 {batch_count}: {speed:.1f} seq/s, 예상 남은 시간: {(total_sequences - seqs_processed) / speed / 3600:.1f}시간")
                    batch_start_time = time.time()
        
        pbar.close()
        
        # 각 GPU의 결과를 임시 파일로 저장 (메모리 절약)
        temp_file = None
        if all_sparse_data:
            num_sequences = len(all_sparse_data)  # 삭제 전에 길이 저장
            # 임시 파일 경로
            temp_file = os.path.join(temp_dir, f'shard_{shard_id}_gpu_{gpu_id}.npz')
            os.makedirs(temp_dir, exist_ok=True)
            
            # 임시 파일로 저장 (메모리에서 즉시 해제)
            save_sparse_data(all_sparse_data, temp_file, args.method, shard_id, global_offset)
            del all_sparse_data  # 메모리 즉시 해제
            
            if gpu_id == 0:
                progress_queue.put(('partial_done', gpu_id, shard_id, shard_name, num_sequences))
        else:
            if gpu_id == 0:
                progress_queue.put(('partial_fail', gpu_id, shard_id, shard_name, 0))
        
        # 메모리 해제 (shard 완료 후 즉시)
        del dataset
        # 주기적으로만 호출하여 오버헤드 감소
        if shard_id % 5 == 0:  # 5개 shard마다 한 번만
            torch.cuda.empty_cache()
        
        # 모든 GPU가 완료할 때까지 대기
        barrier.wait()
        
        # GPU 0이 모든 결과를 합쳐서 최종 파일로 저장
        if gpu_id == 0:
            try:
                # 임시 파일에서 배열 직접 로드 및 합치기 (벡터화로 최적화)
                def load_gpu_arrays(gid):
                    """단일 GPU 데이터를 배열로 로드"""
                    temp_file_gpu = os.path.join(temp_dir, f'shard_{shard_id}_gpu_{gid}.npz')
                    if os.path.exists(temp_file_gpu):
                        data = np.load(temp_file_gpu, allow_pickle=True)
                        result = {key: data[key] for key in data.keys()}
                        os.remove(temp_file_gpu)  # 즉시 삭제
                        return result
                    return None
                
                # 병렬로 모든 GPU 데이터 로드
                with ThreadPoolExecutor(max_workers=min(8, num_gpus)) as executor:
                    gpu_data_list = list(executor.map(load_gpu_arrays, range(num_gpus)))
                
                # None 제거
                gpu_data_list = [d for d in gpu_data_list if d is not None]
                
                if not gpu_data_list:
                    progress_queue.put(('fail', -1, shard_id, shard_name, 0))
                else:
                    # 배열 직접 합치기 (벡터화)
                    if args.method == 'both':
                        # Top-K 배열 합치기
                        all_topk_token_ids = np.concatenate([d['topk_token_ids'] for d in gpu_data_list], axis=0)
                        all_topk_probs = np.concatenate([d['topk_probs'] for d in gpu_data_list], axis=0)
                        
                        # Sparse 배열 합치기
                        all_sparse_token_ids = np.concatenate([d['sparse_token_ids'] for d in gpu_data_list], axis=0)
                        all_sparse_counts = np.concatenate([d['sparse_counts'] for d in gpu_data_list], axis=0)
                        all_sparse_lengths = np.concatenate([d['sparse_lengths'] for d in gpu_data_list], axis=0)
                        
                        # 공통 배열 합치기
                        all_seq_lens = np.concatenate([d['seq_lens'] for d in gpu_data_list], axis=0)
                        all_local_indices = np.concatenate([d['local_indices'] for d in gpu_data_list], axis=0)
                        all_global_indices = np.concatenate([d['global_indices'] for d in gpu_data_list], axis=0)

                        # local_idx로 정렬 (벡터화)
                        sort_idx = np.argsort(all_local_indices)
                        all_topk_token_ids = all_topk_token_ids[sort_idx]
                        all_topk_probs = all_topk_probs[sort_idx]
                        all_sparse_token_ids = all_sparse_token_ids[sort_idx]
                        all_sparse_counts = all_sparse_counts[sort_idx]
                        all_sparse_lengths = all_sparse_lengths[sort_idx]
                        all_seq_lens = all_seq_lens[sort_idx]
                        all_local_indices = all_local_indices[sort_idx]
                        all_global_indices = all_global_indices[sort_idx]

                        # 딕셔너리로 변환 (한 번만)
                        num_sequences = len(all_seq_lens)
                        all_sparse_data_merged = []
                        for i in tqdm(range(num_sequences), desc=f"[GPU 0] Converting to dict", leave=False):
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
                        
                        # 메모리 해제
                        del gpu_data_list, all_topk_token_ids, all_topk_probs
                        del all_sparse_token_ids, all_sparse_counts, all_sparse_lengths
                        del all_seq_lens, all_local_indices, all_global_indices, sort_idx
                        gc.collect()  # 명시적 GC 호출
                    
                    else:
                        # topk 또는 random 메서드
                        all_token_ids = np.concatenate([d['token_ids'] for d in gpu_data_list], axis=0)
                        all_values = np.concatenate([d['values'] for d in gpu_data_list], axis=0)
                        all_seq_lens = np.concatenate([d['seq_lens'] for d in gpu_data_list], axis=0)
                        all_local_indices = np.concatenate([d['local_indices'] for d in gpu_data_list], axis=0)
                        all_global_indices = np.concatenate([d['global_indices'] for d in gpu_data_list], axis=0)

                        if args.method == 'random':
                            all_lengths = np.concatenate([d['lengths'] for d in gpu_data_list], axis=0)

                        # local_idx로 정렬
                        sort_idx = np.argsort(all_local_indices)
                        all_token_ids = all_token_ids[sort_idx]
                        all_values = all_values[sort_idx]
                        all_seq_lens = all_seq_lens[sort_idx]
                        all_local_indices = all_local_indices[sort_idx]
                        all_global_indices = all_global_indices[sort_idx]
                        if args.method == 'random':
                            all_lengths = all_lengths[sort_idx]

                        # 딕셔너리로 변환
                        num_sequences = len(all_seq_lens)
                        all_sparse_data_merged = []
                        for i in tqdm(range(num_sequences), desc=f"[GPU 0] Converting to dict", leave=False):
                            data = {
                                'token_ids': all_token_ids[i],
                                'local_idx': int(all_local_indices[i]),
                                'global_idx': int(all_global_indices[i]),
                                'shard_id': shard_id,
                                'seq_len': int(all_seq_lens[i]),
                            }
                            if args.method == 'random':
                                data['counts'] = all_values[i]
                                data['lengths'] = all_lengths[i]
                                data['num_samples'] = int(gpu_data_list[0]['num_samples'])
                            else:
                                data['probs'] = all_values[i]
                                data['k'] = int(gpu_data_list[0]['k'])
                            all_sparse_data_merged.append(data)
                        
                        # 메모리 해제
                        del gpu_data_list, all_token_ids, all_values, all_seq_lens
                        del all_local_indices, all_global_indices, sort_idx
                        if args.method == 'random':
                            del all_lengths
                        gc.collect()  # 명시적 GC 호출
                    
                    # 최종 파일 저장
                    save_sparse_data(all_sparse_data_merged, output_path, args.method, shard_id, global_offset)
                    num_sequences = len(all_sparse_data_merged)
                    del all_sparse_data_merged  # 메모리 해제
                    gc.collect()  # 명시적 GC 호출로 메모리 즉시 반환
                    progress_queue.put(('done', -1, shard_id, shard_name, num_sequences))
                        
            except Exception as e:
                print(f"[GPU 0] ❌ Error merging results for {shard_name}: {e}")
                import traceback
                traceback.print_exc()
                progress_queue.put(('fail', -1, shard_id, shard_name, 0))
        
        # 모든 GPU가 합치기 완료 대기
        barrier.wait()
    
    progress_queue.put(('finished', gpu_id, -1, '', 0))


def main():
    args = parse_args()
    
    # CUDA 확인
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    available_gpus = torch.cuda.device_count()
    num_gpus = args.num_gpus if args.num_gpus else available_gpus
    num_gpus = min(num_gpus, available_gpus)
    
    print(f"🚀 Multi-GPU Parallel Caching")
    print(f"   Using {num_gpus} GPUs (available: {available_gpus})")
    
    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get shard paths
    all_shard_paths = get_shard_paths(args.data_dir, args.data_prefix)
    print(f"Found {len(all_shard_paths)} total shards")
    
    # Load/compute shard info
    all_shard_sizes, global_offsets, total_sequences = load_or_compute_shard_info(
        args.output_dir, all_shard_paths
    )
    print(f"Total sequences: {total_sequences:,}")
    
    # Filter shards
    if args.end_shard > 0:
        shard_indices = list(range(args.start_shard, args.end_shard))
    else:
        shard_indices = list(range(args.start_shard, len(all_shard_paths)))
    
    print(f"Processing shards {shard_indices[0]} to {shard_indices[-1]} ({len(shard_indices)} shards)")
    
    # Save metadata
    metadata = {
        'method': args.method,
        'topk': args.topk if args.method in ['topk', 'both'] else None,
        'num_samples': args.num_samples if args.method in ['random', 'both'] else None,
        'teacher_model': args.teacher_model_path,
        'max_length': args.max_length,
        'total_shards': len(all_shard_paths),
        'total_sequences': total_sequences,
        'shard_sizes': all_shard_sizes,
        'shard_offsets': global_offsets,
        'data_dir': args.data_dir,
        'num_gpus_used': num_gpus,
    }
    
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    # Shard 리스트 생성 (모든 worker가 같은 리스트 사용)
    mp.set_start_method('spawn', force=True)
    progress_queue = mp.Queue()
    barrier = mp.Barrier(num_gpus)  # 모든 GPU 동기화용
    
    # 임시 파일 디렉토리 생성 (메모리 절약을 위해 디스크 사용)
    temp_dir = os.path.join(args.output_dir, '.temp_mp_results')
    os.makedirs(temp_dir, exist_ok=True)
    
    # 모든 shard를 리스트로 생성
    shard_list = []
    for shard_id in shard_indices:
        shard_path = all_shard_paths[shard_id]
        global_offset = global_offsets[shard_id]
        shard_list.append((shard_id, shard_path, global_offset))
    
    print(f"  Processing {len(shard_list)} shards")
    print(f"  All GPUs will process the same shard with data distribution")
    print(f"  Temporary files: {temp_dir}")
    
    # Start workers (모든 GPU worker 시작)
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_fn,
            args=(gpu_id, shard_list, args, progress_queue, args.teacher_model_path, num_gpus, barrier, temp_dir)
        )
        p.start()
        processes.append(p)
    
    # Monitor progress
    finished_gpus = 0
    total_processed = 0
    start_time = time.time()
    
    with tqdm(total=len(shard_indices), desc="Overall Progress") as pbar:
        while finished_gpus < len(processes):
            try:
                msg = progress_queue.get(timeout=60)
                status, gpu_id, shard_id, shard_name, count = msg
                
                if status == 'start':
                    pass  # Started processing
                elif status == 'partial_done':
                    pass  # Partial processing done (waiting for merge)
                elif status == 'done':
                    total_processed += 1
                    pbar.update(1)
                    pbar.set_postfix({'shard': shard_name, 'seqs': count})
                elif status == 'skip':
                    total_processed += 1
                    pbar.update(1)
                elif status == 'fail' or status == 'partial_fail':
                    if status == 'fail':
                        total_processed += 1
                        pbar.update(1)
                        print(f"\n❌ Failed: {shard_name}")
                elif status == 'finished':
                    finished_gpus += 1
                    print(f"\n[GPU {gpu_id}] ✅ Worker finished")
            except Empty:
                # Check if processes are still alive
                alive = sum(1 for p in processes if p.is_alive())
                if alive == 0:
                    break
    
    # Wait for all processes
    for p in processes:
        p.join()
    
    # 임시 디렉토리 정리
    try:
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"  Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        print(f"  Warning: Could not clean up temp directory: {e}")
    
    elapsed = time.time() - start_time
    print(f"\n✅ Completed! Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"   Processed {total_processed} shards")
    print(f"   Output: {args.output_dir}")


if __name__ == '__main__':
    main()
