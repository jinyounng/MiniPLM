"""
Teacher Logits Caching Script

기존 텍스트 데이터를 유지하면서 Teacher logits만 sparse 형태로 캐싱

Usage:
    # Top-K 방식
    python scripts/cache_teacher_logits.py \
        --teacher-model-path /path/to/teacher \
        --data-dir /data/jykim/DB/miniplm_refined_corpus \
        --output-dir /data/jykim/DB/miniplm_refined_corpus_logits_topk \
        --method topk --topk 50
    
    # Random Sampling 방식
    python scripts/cache_teacher_logits.py \
        --teacher-model-path /path/to/teacher \
        --data-dir /data/jykim/DB/miniplm_refined_corpus \
        --output-dir /data/jykim/DB/miniplm_refined_corpus_logits_sparse \
        --method random --num-samples 50
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from glob import glob
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils.indexed_dataset import MMapIndexedDataset
from data_utils.sparse_sampler import SparseLogitSampler, TopKSampler


def parse_args():
    parser = argparse.ArgumentParser(description='Cache teacher logits in sparse format')
    
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
                        help='Sampling method: topk, random, or both (saves both simultaneously)')
    parser.add_argument('--topk', type=int, default=50,
                        help='K for top-k sampling')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='N for random sampling')
    
    # Processing
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for teacher inference')
    parser.add_argument('--max-length', type=int, default=1024,
                        help='Maximum sequence length')
    parser.add_argument('--start-shard', type=int, default=0,
                        help='Starting shard index (for resuming)')
    parser.add_argument('--end-shard', type=int, default=-1,
                        help='Ending shard index (-1 for all)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'auto'],
                        help='Device to use: cuda (single GPU) or auto (multi-GPU auto-distribute)')
    parser.add_argument('--dtype', type=str, default='bf16',
                        choices=['fp32', 'fp16', 'bf16'],
                        help='Model dtype')
    
    # Saving
    parser.add_argument('--save-interval', type=int, default=1000,
                        help='Save interval (sequences per file)')
    
    return parser.parse_args()


def get_shard_paths(data_dir: str, prefix: str = 'data'):
    """데이터 디렉토리에서 모든 shard 경로 찾기"""
    # Find all .bin files
    bin_files = sorted(glob(os.path.join(data_dir, f'{prefix}_*.bin')))
    
    # Extract shard indices and create paths (without extension)
    shard_paths = []
    for bin_file in bin_files:
        # Remove .bin extension
        path = bin_file[:-4]
        # Check if corresponding .idx exists
        if os.path.exists(path + '.idx'):
            shard_paths.append(path)
    
    # Sort by shard number (robust parsing)
    def get_shard_num(path):
        basename = os.path.basename(path)
        # Remove extension first, then get number
        # "data_train_5" -> "data_train_5" -> "5"
        name_no_ext = basename.rsplit('.', 1)[0] if '.' in basename else basename
        try:
            return int(name_no_ext.split('_')[-1])
        except ValueError:
            # Fallback: return 0 if parsing fails
            print(f"  Warning: Could not parse shard number from {basename}")
            return 0
    
    shard_paths.sort(key=get_shard_num)
    return shard_paths


def load_teacher_model(model_path: str, device: str, dtype: str):
    """
    Teacher 모델 로드
    
    Args:
        device: 'cuda' 또는 'auto'
                - 'cuda': 단일 GPU 사용
                - 'auto': 자동으로 여러 GPU에 분산 (CUDA_VISIBLE_DEVICES에 따라)
    """
    print(f"Loading teacher model from {model_path}...")
    
    dtype_map = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }
    torch_dtype = dtype_map[dtype]
    
    # GPU 분산 설정
    if device == 'auto' or (device == 'cuda' and torch.cuda.device_count() > 1):
        # 여러 GPU 사용 가능하면 자동 분산
        device_map = 'auto'
        print(f"  Using device_map='auto' for multi-GPU ({torch.cuda.device_count()} GPUs)")
    else:
        device_map = device
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # GPU 분산 정보 출력
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        print(f"  Model distributed across devices: {model.hf_device_map}")
    else:
        print(f"  Model on device: {next(model.parameters()).device}")
    
    print(f"Teacher model loaded: {model.config.num_hidden_layers} layers, "
          f"vocab_size={model.config.vocab_size}")
    
    return model, tokenizer


def process_shard(
    model,
    tokenizer,
    shard_path: str,
    shard_id: int,
    global_offset: int,
    output_dir: str,
    sampler_topk,
    sampler_sparse,
    args
):
    """
    단일 shard 처리
    
    Args:
        model: Teacher model
        tokenizer: Tokenizer (for pad_token_id)
        shard_id: shard 번호 (0, 1, 2, ...)
        global_offset: 이 shard의 global index 시작점
    """
    # Load dataset
    dataset = MMapIndexedDataset(shard_path, skip_warmup=True)
    shard_name = os.path.basename(shard_path)
    shard_size = len(dataset)
    
    print(f"\nProcessing {shard_name} (shard_id={shard_id}, global_offset={global_offset}): {shard_size} sequences")
    
    # Output file
    output_path = os.path.join(output_dir, f'{shard_name}.npz')
    
    # Skip if already exists
    if os.path.exists(output_path):
        print(f"  Skipping {shard_name} (already exists)")
        del dataset  # 명시적 메모리 해제
        return shard_size
    
    all_sparse_data = []
    batch_input_ids = []
    batch_local_indices = []
    
    # ✅ Tokenizer에서 pad_id 가져오기 (Qwen 호환성)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    failed_batches = 0
    
    for idx in tqdm(range(shard_size), desc=f"  {shard_name}", leave=False):
        # Get sequence
        data = dataset[idx].astype(np.int64)
        
        # Truncate to max_length + 1 (for labels)
        data = data[:args.max_length + 1]
        
        batch_input_ids.append(data[:-1])  # input
        batch_local_indices.append(idx)
        
        # Process batch when full
        if len(batch_input_ids) >= args.batch_size or idx == shard_size - 1:
            # Pad batch
            max_len = max(len(seq) for seq in batch_input_ids)
            padded_batch = np.full((len(batch_input_ids), max_len), pad_id, dtype=np.int64)
            attention_mask = np.zeros((len(batch_input_ids), max_len), dtype=np.int64)
            
            for i, seq in enumerate(batch_input_ids):
                padded_batch[i, :len(seq)] = seq
                attention_mask[i, :len(seq)] = 1
            
            # Convert to tensor
            # device_map='auto'일 때는 모델의 첫 번째 레이어 디바이스 사용
            if hasattr(model, 'hf_device_map') and model.hf_device_map:
                # Multi-GPU: 첫 번째 레이어가 있는 디바이스 사용
                first_device = list(model.hf_device_map.values())[0]
                device_for_input = first_device
            else:
                # Single GPU: args.device 사용
                device_for_input = args.device if args.device != 'auto' else 'cuda'
            
            input_ids = torch.tensor(padded_batch, device=device_for_input, dtype=torch.long)
            attn_mask = torch.tensor(attention_mask, device=device_for_input, dtype=torch.long)
            
            # ✅ Teacher forward with error handling
            try:
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
                    logits = outputs.logits  # [batch, seq_len, vocab]
                    probs = torch.softmax(logits.float(), dim=-1)
                
                # Sample each sequence
                for i, (seq_probs, orig_len) in enumerate(zip(probs, [len(s) for s in batch_input_ids])):
                    local_idx = batch_local_indices[i]
                    global_idx = global_offset + local_idx  # ✅ Global index 계산
                    
                    # Only use valid positions (non-padded)
                    valid_probs = seq_probs[:orig_len]
                    
                    # ✅ Method에 따라 샘플링 (both면 둘 다)
                    if args.method == 'both':
                        # Top-K와 Random Sampling 둘 다 수행
                        topk_data = sampler_topk.sample(valid_probs)
                        sparse_data = sampler_sparse.sample(valid_probs)
                        
                        # 두 결과를 하나의 dict로 합치기
                        combined_data = {
                            'topk': {
                                'token_ids': topk_data['token_ids'],
                                'probs': topk_data['probs'],
                                'k': topk_data['k']
                            },
                            'sparse': {
                                'token_ids': sparse_data['token_ids'],
                                'counts': sparse_data['counts'],
                                'lengths': sparse_data['lengths'],
                                'num_samples': sparse_data['num_samples']
                            },
                            'local_idx': local_idx,
                            'global_idx': global_idx,
                            'shard_id': shard_id,
                            'seq_len': orig_len,
                            'method': 'both'
                        }
                        all_sparse_data.append(combined_data)
                    else:
                        # 기존 방식 (하나만)
                        sparse_data = sampler_topk.sample(valid_probs) if args.method == 'topk' else sampler_sparse.sample(valid_probs)
                        
                        # 저장할 인덱스 정보
                        sparse_data['local_idx'] = local_idx
                        sparse_data['global_idx'] = global_idx
                        sparse_data['shard_id'] = shard_id
                        sparse_data['seq_len'] = orig_len
                        all_sparse_data.append(sparse_data)
                    
            except RuntimeError as e:
                failed_batches += 1
                print(f"\n  ⚠️ Forward pass failed for batch (local_idx={batch_local_indices}): {e}")
                # Skip this batch - 나중에 재처리 가능하도록 로그에 기록
                if failed_batches > 10:
                    print(f"  ❌ Too many failures ({failed_batches}), stopping shard {shard_id}")
                    break
            
            # Clear batch and free GPU memory
            batch_input_ids = []
            batch_local_indices = []
            del input_ids, attn_mask
            if 'logits' in dir():
                del logits, probs, outputs
            torch.cuda.empty_cache()
    
    # Save shard
    if all_sparse_data:
        save_sparse_data(all_sparse_data, output_path, args.method, shard_id, global_offset)
        print(f"  Saved {len(all_sparse_data)} sequences to {output_path}")
        if failed_batches > 0:
            print(f"  ⚠️ {failed_batches} batches failed")
    else:
        print(f"  ❌ No data to save for {shard_name}")
    
    # ✅ 명시적 메모리 해제
    del dataset
    torch.cuda.empty_cache()
    
    return shard_size


def save_sparse_data(sparse_data_list: list, output_path: str, method: str, shard_id: int, global_offset: int):
    """
    Sparse data를 npz 형식으로 저장
    
    저장 형식:
    - method='both': topk와 sparse 둘 다 저장
    - method='topk' or 'random': 하나만 저장
    """
    all_seq_lens = []
    all_local_indices = []
    all_global_indices = []
    
    if method == 'both':
        # Top-K와 Sparse 둘 다 저장
        all_topk_token_ids = []
        all_topk_probs = []
        all_sparse_token_ids = []
        all_sparse_counts = []
        all_sparse_lengths = []
        
        for data in sparse_data_list:
            all_seq_lens.append(data['seq_len'])
            all_local_indices.append(data['local_idx'])
            all_global_indices.append(data['global_idx'])
            
            # Top-K 데이터
            all_topk_token_ids.append(data['topk']['token_ids'])
            all_topk_probs.append(data['topk']['probs'])
            
            # Sparse 데이터
            all_sparse_token_ids.append(data['sparse']['token_ids'])
            all_sparse_counts.append(data['sparse']['counts'])
            all_sparse_lengths.append(data['sparse']['lengths'])
        
        save_dict = {
            # Top-K
            'topk_token_ids': np.array(all_topk_token_ids, dtype=object),
            'topk_probs': np.array(all_topk_probs, dtype=object),
            'topk_k': np.int16(sparse_data_list[0]['topk']['k']),
            
            # Sparse (Random Sampling)
            'sparse_token_ids': np.array(all_sparse_token_ids, dtype=object),
            'sparse_counts': np.array(all_sparse_counts, dtype=object),
            'sparse_lengths': np.array(all_sparse_lengths, dtype=object),
            'sparse_num_samples': np.int16(sparse_data_list[0]['sparse']['num_samples']),
            
            # 공통
            'seq_lens': np.array(all_seq_lens, dtype=np.int32),
            'local_indices': np.array(all_local_indices, dtype=np.int32),
            'global_indices': np.array(all_global_indices, dtype=np.int64),
            'shard_id': np.int32(shard_id),
            'global_offset': np.int64(global_offset),
            'method': np.array('both'),
        }
    else:
        # 기존 방식 (하나만)
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
            else:  # topk
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
    
    np.savez_compressed(output_path, **save_dict)


def get_shard_sizes(shard_paths: list) -> list:
    """각 shard의 크기(시퀀스 수)를 미리 계산"""
    sizes = []
    print("Probing shard sizes...")
    for path in tqdm(shard_paths, desc="Probing"):
        dataset = MMapIndexedDataset(path, skip_warmup=True)
        sizes.append(len(dataset))
        del dataset
    return sizes


def load_or_compute_shard_info(output_dir: str, all_shard_paths: list) -> tuple:
    """
    Shard 크기 정보를 캐시에서 로드하거나, 없으면 계산
    
    Returns:
        (all_shard_sizes, global_offsets, total_sequences)
    """
    metadata_path = os.path.join(output_dir, 'metadata.json')
    
    # ✅ 기존 metadata.json에서 shard 정보 재사용 (속도 개선)
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                cached_meta = json.load(f)
            
            # 캐시된 정보가 현재 shard 수와 일치하는지 확인
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
    
    # 캐시 없으면 새로 계산
    print("Computing shard sizes (first run or metadata mismatch)...")
    all_shard_sizes = get_shard_sizes(all_shard_paths)
    
    # Calculate global offsets
    global_offsets = [0]
    for size in all_shard_sizes[:-1]:
        global_offsets.append(global_offsets[-1] + size)
    
    total_sequences = sum(all_shard_sizes)
    
    return all_shard_sizes, global_offsets, total_sequences


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get ALL shard paths first (for global offset calculation)
    all_shard_paths = get_shard_paths(args.data_dir, args.data_prefix)
    print(f"Found {len(all_shard_paths)} total shards")
    
    # ✅ Shard 정보 로드 또는 계산 (캐싱으로 속도 개선)
    all_shard_sizes, global_offsets, total_sequences = load_or_compute_shard_info(
        args.output_dir, all_shard_paths
    )
    
    print(f"Total sequences across all shards: {total_sequences:,}")
    
    # Filter shards to process
    if args.end_shard > 0:
        shard_indices = list(range(args.start_shard, args.end_shard))
    else:
        shard_indices = list(range(args.start_shard, len(all_shard_paths)))
    
    print(f"Processing shards {shard_indices[0]} to {shard_indices[-1]}")
    
    # Load teacher model
    model, tokenizer = load_teacher_model(args.teacher_model_path, args.device, args.dtype)
    vocab_size = model.config.vocab_size
    
    # ✅ Tokenizer pad_token 설정 확인
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"  Set pad_token_id to eos_token_id: {tokenizer.pad_token_id}")
    print(f"  Using pad_token_id: {tokenizer.pad_token_id}")
    
    # Initialize samplers
    if args.method == 'both':
        sampler_topk = TopKSampler(k=args.topk)
        sampler_sparse = SparseLogitSampler(num_samples=args.num_samples)
        print(f"Using BOTH methods: Top-K (K={args.topk}) + Random Sampling (N={args.num_samples})")
        print(f"  → Teacher forward 한 번으로 둘 다 저장 (시간 절약!)")
    elif args.method == 'topk':
        sampler_topk = TopKSampler(k=args.topk)
        sampler_sparse = None
        print(f"Using Top-K sampling (K={args.topk})")
    else:  # random
        sampler_topk = None
        sampler_sparse = SparseLogitSampler(num_samples=args.num_samples)
        print(f"Using Random Sampling (N={args.num_samples})")
    
    # Save metadata (shard 정보 포함)
    metadata = {
        'method': args.method,
        'topk': args.topk if args.method in ['topk', 'both'] else None,
        'num_samples': args.num_samples if args.method in ['random', 'both'] else None,
        'teacher_model': args.teacher_model_path,
        'vocab_size': vocab_size,
        'max_length': args.max_length,
        'total_shards': len(all_shard_paths),
        'total_sequences': total_sequences,
        'shard_sizes': all_shard_sizes,          # 각 shard의 크기
        'shard_offsets': global_offsets,         # 각 shard의 global offset
        'data_dir': args.data_dir,
        'pad_token_id': tokenizer.pad_token_id,  # ✅ 추가
    }
    
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    # Process each shard
    for shard_id in shard_indices:
        shard_path = all_shard_paths[shard_id]
        global_offset = global_offsets[shard_id]
        
        process_shard(
            model=model,
            tokenizer=tokenizer,  # ✅ tokenizer 전달
            shard_path=shard_path,
            shard_id=shard_id,
            global_offset=global_offset,
            output_dir=args.output_dir,
            sampler_topk=sampler_topk,
            sampler_sparse=sampler_sparse,
            args=args
        )
    
    print(f"\n✅ Completed! Cached logits saved to {args.output_dir}")


if __name__ == '__main__':
    main()

