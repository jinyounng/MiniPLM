"""
Extract Hidden States from Teacher Model - Multi-GPU Version

Teacher 모델로부터 hidden states를 미리 추출하여 저장하는 스크립트입니다.
8개 GPU를 사용하여 병렬 처리합니다.
저장된 파일은 train_ae.py에서 --pre_extracted_path로 사용할 수 있습니다.

Usage:
    python extract_hidden_states.py \
        --data_path /path/to/data_0 \
        --teacher_path /path/to/teacher \
        --output_path /path/to/hidden_states.pt \
        --max_length 1024 \
        --batch_size 32 \
        --num_gpus 8
"""

import os
import sys
import argparse
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
import time
from queue import Empty

# Add parent directory to path for data_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_utils.indexed_dataset import MMapIndexedDataset


def worker_fn(
    gpu_id: int,
    data_path: str,
    teacher_path: str,
    start_idx: int,
    end_idx: int,
    max_length: int,
    batch_size: int,
    progress_queue: mp.Queue,
    temp_dir: str,
    checkpoint_interval: int = 100,
):
    """Worker function for single GPU"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'
    
    # Load model
    print(f"[GPU {gpu_id}] Loading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    teacher_model.eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    # Load dataset
    dataset = MMapIndexedDataset(data_path, skip_warmup=True)
    
    hidden_list = []
    x_token_list = []
    y_token_list = []
    
    batch_input_ids = []
    batch_count = 0
    
    with torch.no_grad():
        for seq_idx in tqdm(range(start_idx, end_idx), desc=f"[GPU {gpu_id}] Extracting", position=gpu_id, leave=False):
            data = dataset[seq_idx].astype(np.int64)
            data = data[:max_length + 1]
            
            if len(data) <= 1:
                continue
            
            batch_input_ids.append(data[:-1])
            
            is_last_seq = (seq_idx == end_idx - 1)
            if len(batch_input_ids) >= batch_size or is_last_seq:
                batch_count += 1
                actual_batch_size = len(batch_input_ids)
                
                max_len = max(len(seq) for seq in batch_input_ids)
                max_len = min(max_len, max_length)
                
                padded_batch = np.full((actual_batch_size, max_len), pad_id, dtype=np.int64)
                attention_mask = np.zeros((actual_batch_size, max_len), dtype=np.int64)
                
                actual_seq_lens = []
                for i, seq in enumerate(batch_input_ids):
                    seq_len = min(len(seq), max_len)
                    actual_seq_lens.append(seq_len)
                    padded_batch[i, :seq_len] = seq[:seq_len]
                    attention_mask[i, :seq_len] = 1
                
                input_ids = torch.tensor(padded_batch, device=device, dtype=torch.long)
                attn_mask = torch.tensor(attention_mask, device=device, dtype=torch.long)
                
                outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=True,
                    use_cache=False
                )
                last_hidden = outputs.hidden_states[-1]  # [B, max_len, hidden_dim]
                teacher_logits = outputs.logits  # [B, max_len, vocab_size]
                
                # argmax를 배치 전체로 한 번에 처리 (훨씬 빠름)
                y_preds_batch = torch.argmax(teacher_logits, dim=-1)  # [B, max_len]
                
                # 배치 전체를 CPU로 한 번에 이동 (개별 이동보다 훨씬 빠름)
                last_hidden_cpu = last_hidden.cpu()
                y_preds_cpu = y_preds_batch.cpu()
                input_ids_cpu = input_ids.cpu().long()
                
                # 벡터화: 각 시퀀스의 valid 부분을 한 번에 슬라이싱 (for loop + .item() 제거)
                for i in range(actual_batch_size):
                    actual_len = actual_seq_lens[i]
                    if actual_len <= 1:
                        continue
                    
                    # valid_len = actual_len - 1 (마지막 position 제외)
                    valid_len = actual_len - 1
                    
                    # 한 번에 슬라이싱 (벡터화)
                    hidden_list.append(last_hidden_cpu[i, :valid_len])  # [valid_len, hidden_dim]
                    x_token_list.append(input_ids_cpu[i, :valid_len])   # [valid_len]
                    y_token_list.append(y_preds_cpu[i, :valid_len])       # [valid_len]
                
                batch_input_ids = []
                del input_ids, attn_mask, outputs, last_hidden, teacher_logits
                
                # 주기적으로 체크포인트 저장 (중간 저장)
                if checkpoint_interval > 0 and batch_count % checkpoint_interval == 0 and len(hidden_list) > 0:
                    import gc
                    
                    checkpoint_file = os.path.join(temp_dir, f'gpu_{gpu_id}_checkpoint.pt')
                    hiddens_checkpoint = torch.cat(hidden_list, dim=0)
                    x_tokens_checkpoint = torch.cat(x_token_list, dim=0)
                    y_tokens_checkpoint = torch.cat(y_token_list, dim=0)
                    
                    torch.save({
                        'hiddens': hiddens_checkpoint,
                        'x_tokens': x_tokens_checkpoint,
                        'y_tokens': y_tokens_checkpoint,
                        'batch_count': batch_count,
                        'seq_idx': seq_idx
                    }, checkpoint_file)
                    
                    # 메모리 정리: 리스트의 각 텐서를 명시적으로 해제
                    # clear()만으로는 부족 - 각 텐서 객체를 del해야 함
                    for tensor in hidden_list:
                        del tensor
                    for tensor in x_token_list:
                        del tensor
                    for tensor in y_token_list:
                        del tensor
                    
                    hidden_list.clear()
                    x_token_list.clear()
                    y_token_list.clear()
                    
                    # 명시적 메모리 해제
                    del hiddens_checkpoint, x_tokens_checkpoint, y_tokens_checkpoint
                    
                    # Python GC 강제 실행 (메모리 즉시 해제)
                    gc.collect()
                    
                    progress_queue.put(('checkpoint', gpu_id, 0))
                
                # torch.cuda.empty_cache() 제거 - CUDA 동기화로 인한 병목 방지
                # 필요시 주기적으로만 호출 (예: 100배치마다)
                # if batch_count % 100 == 0:
                #     torch.cuda.empty_cache()
    
    # Save to temp file (.pt format)
    if len(hidden_list) > 0:
        # concat 사용 (각 시퀀스의 길이가 다르므로 stack 대신 concat)
        hiddens = torch.cat(hidden_list, dim=0)  # [total_tokens, hidden_dim]
        x_tokens = torch.cat(x_token_list, dim=0)  # [total_tokens]
        y_tokens = torch.cat(y_token_list, dim=0)  # [total_tokens]
        
        temp_file = os.path.join(temp_dir, f'gpu_{gpu_id}.pt')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save as torch .pt file (fast loading)
        torch.save({
            'hiddens': hiddens,
            'x_tokens': x_tokens,
            'y_tokens': y_tokens
        }, temp_file)
        
        progress_queue.put(('done', gpu_id, len(hidden_list)))
    else:
        progress_queue.put(('done', gpu_id, 0))
    
    progress_queue.put(('finished', gpu_id, 0))


def _merge_checkpoints(temp_dir: str, output_path: str, num_gpus: int):
    """병합 체크포인트를 최종 파일에 저장 (중간 저장)"""
    import gc
    
    all_hiddens = []
    all_x_tokens = []
    all_y_tokens = []
    
    for gpu_id in range(num_gpus):
        checkpoint_file = os.path.join(temp_dir, f'gpu_{gpu_id}_checkpoint.pt')
        if os.path.exists(checkpoint_file):
            data = torch.load(checkpoint_file, map_location='cpu')
            all_hiddens.append(data['hiddens'])
            all_x_tokens.append(data['x_tokens'])
            all_y_tokens.append(data['y_tokens'])
            # 로드한 데이터는 메모리에서 해제 (원본 파일은 유지)
            del data
    
    if len(all_hiddens) > 0:
        hiddens = torch.cat(all_hiddens, dim=0)
        x_tokens = torch.cat(all_x_tokens, dim=0)
        y_tokens = torch.cat(all_y_tokens, dim=0)
        
        # 중간 저장 파일로 저장 (최종 파일과 별도)
        checkpoint_output = output_path.replace('.pt', '_checkpoint.pt')
        os.makedirs(os.path.dirname(checkpoint_output) if os.path.dirname(checkpoint_output) else '.', exist_ok=True)
        torch.save({
            'hiddens': hiddens,
            'x_tokens': x_tokens,
            'y_tokens': y_tokens
        }, checkpoint_output)
        
        file_size_gb = os.path.getsize(checkpoint_output) / 1e9
        print(f"💾 Checkpoint merged: {checkpoint_output} ({file_size_gb:.2f} GB, {hiddens.shape[0]:,} tokens)")
        
        # 메모리 정리
        del all_hiddens, all_x_tokens, all_y_tokens
        del hiddens, x_tokens, y_tokens
        gc.collect()  # Python GC 강제 실행


def extract_hidden_states(
    data_path: str,
    teacher_path: str,
    output_path: str,
    max_length: int = 1024,
    batch_size: int = 32,
    num_gpus: int = 8,
    max_samples: int = None,
    start_idx: int = 0,
    checkpoint_interval: int = 100,
):
    """Extract hidden states from teacher model using multiple GPUs"""
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")
    
    available_gpus = torch.cuda.device_count()
    num_gpus = min(num_gpus, available_gpus)
    
    print(f"🚀 Multi-GPU Hidden States Extraction")
    print(f"   Using {num_gpus} GPUs (available: {available_gpus})")
    
    # Load dataset to get size
    dataset = MMapIndexedDataset(data_path, skip_warmup=True)
    total_sequences = len(dataset)
    
    end_idx = start_idx + max_samples if max_samples is not None else total_sequences
    end_idx = min(end_idx, total_sequences)
    
    print(f"Processing sequences {start_idx} ~ {end_idx-1} (total: {total_sequences})...")
    
    # Distribute work across GPUs
    total_work = end_idx - start_idx
    chunk_size = total_work // num_gpus
    
    # Setup multiprocessing
    mp.set_start_method('spawn', force=True)
    progress_queue = mp.Queue()
    temp_dir = os.path.join(os.path.dirname(output_path), '.temp_hidden_states')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Start workers
    processes = []
    for gpu_id in range(num_gpus):
        gpu_start = start_idx + gpu_id * chunk_size
        gpu_end = start_idx + (gpu_id + 1) * chunk_size if gpu_id < num_gpus - 1 else end_idx
        
        p = mp.Process(
            target=worker_fn,
            args=(gpu_id, data_path, teacher_path, gpu_start, gpu_end, max_length, batch_size, progress_queue, temp_dir, checkpoint_interval)
        )
        p.start()
        processes.append(p)
    
    # Monitor progress and merge checkpoints periodically
    finished_gpus = 0
    start_time = time.time()
    last_checkpoint_merge = time.time()
    checkpoint_merge_interval = 300  # 5분마다 체크포인트 병합
    
    while finished_gpus < num_gpus:
        try:
            msg = progress_queue.get(timeout=60)
            status, gpu_id, count = msg
            
            if status == 'done':
                print(f"[GPU {gpu_id}] Extracted {count:,} tokens")
            elif status == 'checkpoint':
                print(f"[GPU {gpu_id}] 💾 Checkpoint saved ({count:,} tokens)")
            elif status == 'finished':
                finished_gpus += 1
                print(f"[GPU {gpu_id}] ✅ Finished")
        except Empty:
            alive = sum(1 for p in processes if p.is_alive())
            if alive == 0:
                break
        
        # 주기적으로 체크포인트 병합 (중간 저장)
        current_time = time.time()
        if checkpoint_interval > 0 and (current_time - last_checkpoint_merge) >= checkpoint_merge_interval:
            _merge_checkpoints(temp_dir, output_path, num_gpus)
            last_checkpoint_merge = current_time
    
    # Wait for all processes
    for p in processes:
        p.join()
    
    # Merge results from all GPUs
    print("\nMerging results from all GPUs...")
    all_hiddens = []
    all_x_tokens = []
    all_y_tokens = []
    
    for gpu_id in range(num_gpus):
        temp_file = os.path.join(temp_dir, f'gpu_{gpu_id}.pt')
        if os.path.exists(temp_file):
            data = torch.load(temp_file, map_location='cpu')
            all_hiddens.append(data['hiddens'])
            all_x_tokens.append(data['x_tokens'])
            all_y_tokens.append(data['y_tokens'])
            os.remove(temp_file)
    
    # Concatenate and save
    hiddens = torch.cat(all_hiddens, dim=0)
    x_tokens = torch.cat(all_x_tokens, dim=0)
    y_tokens = torch.cat(all_y_tokens, dim=0)
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal tokens: {hiddens.shape[0]:,}")
    print(f"Extraction took {elapsed_time/60:.2f} minutes ({elapsed_time:.2f} seconds)")
    
    # Save
    print(f"\nSaving to {output_path}...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    torch.save({
        'hiddens': hiddens,
        'x_tokens': x_tokens,
        'y_tokens': y_tokens
    }, output_path)
    
    # Cleanup
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    file_size_gb = os.path.getsize(output_path) / 1e9
    print(f"✅ Saved! File size: {file_size_gb:.2f} GB")
    print(f"\nUsage in train_ae.py:")
    print(f"  --pre_extracted_path {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract hidden states from teacher model (Multi-GPU)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to data file (.bin without extension)")
    parser.add_argument("--teacher_path", type=str, required=True,
                        help="Path to teacher model")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for extracted hidden states (.pt file)")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for extraction")
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="Number of GPUs to use (default: 8)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (default: all)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting index (default: 0)")
    parser.add_argument("--checkpoint_interval", type=int, default=100,
                        help="Save checkpoint every N batches (0 to disable, default: 100)")
    
    args = parser.parse_args()
    
    extract_hidden_states(
        data_path=args.data_path,
        teacher_path=args.teacher_path,
        output_path=args.output_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        max_samples=args.max_samples,
        start_idx=args.start_idx,
        checkpoint_interval=args.checkpoint_interval,
    )


if __name__ == "__main__":
    main()
