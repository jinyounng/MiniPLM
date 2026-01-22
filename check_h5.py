#!/usr/bin/env python3
import h5py
import numpy as np
import glob

h5_files = glob.glob('/home/jiwonyoon/data1/data/logits_hdf5/*.h5')
if h5_files:
    h5_file = sorted(h5_files)[0]
    print(f'Opening: {h5_file}\n')
    
    with h5py.File(h5_file, 'r') as f:
        print('=== HDF5 File Structure ===')
        print(f'Keys at root: {list(f.keys())}')
        print(f'Attributes: {dict(f.attrs)}\n')
        
        def print_structure(name, obj):
            indent = '  ' * (name.count('/'))
            if isinstance(obj, h5py.Dataset):
                print(f'{indent}{name.split("/")[-1]}: Dataset')
                print(f'{indent}  Shape: {obj.shape}')
                print(f'{indent}  Dtype: {obj.dtype}')
                print(f'{indent}  Size: {obj.size:,} elements')
            elif isinstance(obj, h5py.Group):
                print(f'{indent}{name.split("/")[-1]}: Group')
                print(f'{indent}  Keys: {list(obj.keys())}')
                if obj.attrs:
                    print(f'{indent}  Attributes: {dict(obj.attrs)}')
        
        f.visititems(print_structure)
        print()
        
        # 각 그룹 확인
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Group):
                print(f'\n=== Group: {key} ===')
                for subkey in obj.keys():
                    subobj = obj[subkey]
                    if isinstance(subobj, h5py.Dataset):
                        print(f'{subkey}:')
                        print(f'  Shape: {subobj.shape}')
                        print(f'  Dtype: {subobj.dtype}')
                        if subobj.size > 0:
                            if len(subobj.shape) == 1 and subobj.shape[0] <= 20:
                                print(f'  Data: {subobj[:]}')
                            elif len(subobj.shape) == 1:
                                print(f'  First 10: {subobj[:10]}')
                            elif len(subobj.shape) == 2:
                                print(f'  First row (first 20): {subobj[0, :20] if subobj.shape[0] > 0 else "empty"}')
                            elif len(subobj.shape) == 3:
                                if subobj.shape[0] > 0:
                                    print(f'  First sample shape: {subobj[0].shape}')
                                    print(f'  First sample (first 5x5): {subobj[0, :5, :5]}')
                        print()
                if obj.attrs:
                    print(f'Attributes: {dict(obj.attrs)}')
                print()
        
        # 샘플 데이터 상세 확인
        print('\n=== Sample Data Details (First Sequence) ===')
        
        # 첫 번째 시퀀스 데이터 추출
        if 'sparse' in f and 'topk' in f:
            sparse_grp = f['sparse']
            topk_grp = f['topk']
            meta_grp = f['meta']
            
            # 첫 번째 시퀀스 정보
            seq_idx = 0
            seq_len = meta_grp['seq_lens'][seq_idx]
            print(f'\nSequence {seq_idx}: length={seq_len}')
            
            # Sparse 데이터 추출 (작은 샘플만)
            print('\n--- Sparse (Random Sampling) ---')
            sparse_seq_start = sparse_grp['seq_offsets'][seq_idx]
            sparse_seq_end = sparse_grp['seq_offsets'][seq_idx + 1]
            sparse_token_start = sparse_grp['token_offsets'][sparse_seq_start]
            sparse_token_end = sparse_grp['token_offsets'][sparse_seq_end]
            
            # 작은 샘플만 로드 (처음 100개만)
            sample_size = min(100, sparse_token_end - sparse_token_start)
            sparse_tokens = sparse_grp['token_ids_flat'][sparse_token_start:sparse_token_start + sample_size]
            sparse_counts = sparse_grp['counts_flat'][sparse_token_start:sparse_token_start + sample_size]
            
            print(f'  Total unique tokens (estimated): {sparse_token_end - sparse_token_start}')
            print(f'  Sampled tokens (first {sample_size}): {sparse_tokens}')
            print(f'  Sampled counts: {sparse_counts}')
            print(f'  Token range (sample): {sparse_tokens.min()} ~ {sparse_tokens.max()}')
            print(f'  Count range (sample): {sparse_counts.min()} ~ {sparse_counts.max()}')
            print(f'  num_samples attribute: {sparse_grp.attrs.get("num_samples", "N/A")}')
            
            # TopK 데이터 추출 (작은 샘플만)
            print('\n--- TopK ---')
            topk_seq_start = topk_grp['seq_offsets'][seq_idx]
            topk_seq_end = topk_grp['seq_offsets'][seq_idx + 1]
            k = topk_grp.attrs.get('k', 100)
            
            # 첫 번째 위치의 top-k만 로드
            topk_tokens = topk_grp['token_ids_flat'][topk_seq_start:topk_seq_start + k]
            topk_probs = topk_grp['probs_flat'][topk_seq_start:topk_seq_start + k]
            
            print(f'  k: {k}')
            print(f'  Total tokens (seq_len * k): {topk_seq_end - topk_seq_start}')
            print(f'  First position top-10 tokens: {topk_tokens[:10]}')
            print(f'  First position top-10 probs: {topk_probs[:10]}')
            print(f'  Prob range (first position): {topk_probs.min():.6f} ~ {topk_probs.max():.6f}')
            print(f'  Prob sum (first position, top-{k}): {topk_probs.sum():.6f}')
            print(f'  Note: Top-K prob sum < 1.0 is normal (tail probability excluded)')
            
            # 두 번째 위치도 확인
            if seq_len > 1:
                topk_pos1_tokens = topk_grp['token_ids_flat'][topk_seq_start + k:topk_seq_start + 2*k]
                topk_pos1_probs = topk_grp['probs_flat'][topk_seq_start + k:topk_seq_start + 2*k]
                print(f'  Second position prob sum (top-{k}): {topk_pos1_probs.sum():.6f}')
            
            # Meta 정보
            print('\n--- Meta ---')
            print(f'  method: {meta_grp.attrs.get("method", "N/A")}')
            print(f'  num_sequences: {meta_grp.attrs.get("num_sequences", "N/A")}')
            print(f'  shard_id: {meta_grp.attrs.get("shard_id", "N/A")}')
else:
    print('No H5 files found')
