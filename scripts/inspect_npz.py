#!/usr/bin/env python3
"""
NPZ 파일 구조 확인 스크립트

Usage:
    python scripts/inspect_npz.py <npz_file_path>
    python scripts/inspect_npz.py /path/to/data_0.npz
"""

import numpy as np
import os
import sys
import argparse


def inspect_npz(npz_file: str, show_samples: bool = True, max_samples: int = 3):
    """NPZ 파일의 구조와 내용을 확인"""
    
    if not os.path.exists(npz_file):
        print(f"❌ 파일을 찾을 수 없습니다: {npz_file}")
        return
    
    print(f"\n📂 파일: {npz_file}")
    print(f"📊 파일 크기: {os.path.getsize(npz_file) / (1024**3):.2f} GB\n")
    
    # 파일 로드 (lazy loading)
    data = np.load(npz_file, allow_pickle=True)
    
    print("=" * 60)
    print("📋 저장된 키 목록:")
    print("=" * 60)
    
    for key in sorted(data.keys()):
        arr = data[key]
        if isinstance(arr, np.ndarray):
            print(f"\n🔑 {key}:")
            print(f"   - Shape: {arr.shape}")
            print(f"   - Dtype: {arr.dtype}")
            
            if arr.dtype == object:
                print(f"   - Type: object array (각 요소가 다른 크기의 배열)")
                if len(arr) > 0:
                    try:
                        sample = arr[0]
                        if isinstance(sample, np.ndarray):
                            print(f"   - Sample[0] shape: {sample.shape}, dtype: {sample.dtype}")
                            if sample.size > 0:
                                print(f"   - Sample[0] size: {sample.nbytes / (1024**2):.2f} MB")
                    except Exception as e:
                        print(f"   - Sample[0] 로드 오류: {e}")
            else:
                size_mb = arr.nbytes / (1024**2)
                if size_mb < 100:
                    print(f"   - Size: {size_mb:.2f} MB")
                    if arr.size <= 20:
                        print(f"   - Values: {arr}")
                else:
                    print(f"   - Size: {size_mb:.2f} MB")
        else:
            print(f"\n🔑 {key}: {type(arr)} = {arr}")
    
    # 메타데이터 확인
    print("\n" + "=" * 60)
    print("📊 메타데이터:")
    print("=" * 60)
    
    if 'method' in data:
        method = data['method']
        if isinstance(method, np.ndarray):
            method = str(method.item()) if method.size > 0 else 'unknown'
        print(f"Method: {method}")
    
    if 'topk_k' in data:
        print(f"Top-K K: {data['topk_k']}")
    
    if 'sparse_num_samples' in data:
        print(f"Sparse N: {data['sparse_num_samples']}")
    
    if 'seq_lens' in data:
        seq_lens = data['seq_lens']
        print(f"\n📏 Sequence lengths:")
        print(f"   - Total sequences: {len(seq_lens):,}")
        if len(seq_lens) > 0:
            print(f"   - First 5: {seq_lens[:5]}")
            print(f"   - Min: {seq_lens.min()}, Max: {seq_lens.max()}, Mean: {seq_lens.mean():.1f}")
    
    if 'shard_id' in data:
        print(f"\n📦 Shard ID: {data['shard_id']}")
    
    if 'global_offset' in data:
        print(f"📍 Global offset: {data['global_offset']}")
    
    if 'local_indices' in data:
        local_indices = data['local_indices']
        print(f"\n📍 Local indices:")
        print(f"   - First 10: {local_indices[:10]}")
        print(f"   - Last 10: {local_indices[-10:]}")
    
    if 'global_indices' in data:
        global_indices = data['global_indices']
        print(f"\n🌐 Global indices:")
        print(f"   - First 10: {global_indices[:10]}")
        print(f"   - Last 10: {global_indices[-10:]}")
    
    # 샘플 데이터 확인 - 토큰 번호와 logit(prob) 확인
    if show_samples:
        print("\n" + "=" * 60)
        print("📊 토큰 번호 & Logit 확인 (처음 몇 개 시퀀스):")
        print("=" * 60)
        
        try:
            method = str(data['method'].item()) if 'method' in data else 'unknown'
            
            if 'both' in method or 'topk' in method:
                if 'topk_token_ids' in data and len(data['topk_token_ids']) > 0:
                    print(f"\n🔝 Top-K (토큰 번호 + 확률/로짓):")
                    k_value = int(data['topk_k']) if 'topk_k' in data else topk_tokens.shape[1] if len(data['topk_token_ids']) > 0 else 'N/A'
                    print(f"   K={k_value}")
                    
                    for i in range(min(max_samples, len(data['topk_token_ids']))):
                        topk_tokens = data['topk_token_ids'][i]  # [seq_len, K]
                        topk_probs = data['topk_probs'][i]       # [seq_len, K]
                        seq_len = data['seq_lens'][i] if 'seq_lens' in data else len(topk_tokens)
                        
                        print(f"\n   Sequence {i} (seq_len={seq_len}):")
                        print(f"   - Shape: tokens={topk_tokens.shape}, probs={topk_probs.shape}")
                        print(f"   - 토큰 번호 dtype: {topk_tokens.dtype}, 확률 dtype: {topk_probs.dtype}")
                        
                        # 첫 3개 위치 확인
                        for pos in range(min(3, len(topk_tokens))):
                            tokens = topk_tokens[pos]  # [K]
                            probs = topk_probs[pos]    # [K]
                            
                            print(f"\n   Position {pos}:")
                            print(f"      ✅ 토큰 번호 (전체 {len(tokens)}개): {tokens}")
                            print(f"      ✅ 확률/로짓 (전체 {len(probs)}개): {probs}")
                            
                            # 검증
                            print(f"      검증:")
                            print(f"         - 토큰 번호 범위: [{tokens.min()}, {tokens.max()}]")
                            print(f"         - 확률 범위: [{probs.min():.6f}, {probs.max():.6f}]")
                            print(f"         - 확률 합: {probs.sum():.6f}")
                            
                            # 토큰 번호와 확률이 같은 순서인지 확인
                            if len(tokens) == len(probs):
                                print(f"         - 토큰-확률 쌍 개수: {len(tokens)} (일치 ✅)")
                            else:
                                print(f"         - ⚠️ 토큰-확률 개수 불일치!")
                            
                            # 상위 5개만 자세히
                            print(f"      상위 5개:")
                            for j in range(min(5, len(tokens))):
                                print(f"         [{j}] 토큰={tokens[j]}, 확률={probs[j]:.6f}")
            
            if 'both' in method or 'random' in method:
                if 'sparse_token_ids' in data and len(data['sparse_token_ids']) > 0:
                    print(f"\n🎲 Sparse (토큰 번호 + 카운트):")
                    for i in range(min(max_samples, len(data['sparse_token_ids']))):
                        sparse_tokens = data['sparse_token_ids'][i]  # [seq_len, max_unique]
                        sparse_counts = data['sparse_counts'][i]      # [seq_len, max_unique]
                        sparse_lengths = data['sparse_lengths'][i]    # [seq_len]
                        seq_len = data['seq_lens'][i] if 'seq_lens' in data else len(sparse_tokens)
                        
                        print(f"\n   Sequence {i} (seq_len={seq_len}):")
                        print(f"   - Shape: tokens={sparse_tokens.shape}, counts={sparse_counts.shape}")
                        print(f"   - 각 위치별 unique 토큰 수 (first 10): {sparse_lengths[:10]}")
                        
                        # 첫 3개 위치 확인
                        for pos in range(min(3, len(sparse_tokens))):
                            tokens = sparse_tokens[pos]
                            counts = sparse_counts[pos]
                            num_unique = sparse_lengths[pos]
                            print(f"\n   Position {pos} (unique={num_unique}):")
                            print(f"      토큰 번호 (first 10): {tokens[:num_unique][:10]}")
                            print(f"      카운트 (first 10):    {counts[:num_unique][:10]}")
                            print(f"      카운트 합: {counts[:num_unique].sum()}")
        
        except Exception as e:
            print(f"⚠️ 샘플 로드 중 오류: {e}")
            import traceback
            traceback.print_exc()
    
    data.close()
    print("\n✅ 확인 완료!")


def main():
    parser = argparse.ArgumentParser(description='NPZ 파일 구조 확인')
    parser.add_argument('npz_file', type=str, help='확인할 NPZ 파일 경로')
    parser.add_argument('--no-samples', action='store_true', help='샘플 데이터 표시 안 함')
    parser.add_argument('--max-samples', type=int, default=3, help='표시할 샘플 수 (기본값: 3)')
    
    args = parser.parse_args()
    
    inspect_npz(args.npz_file, show_samples=not args.no_samples, max_samples=args.max_samples)


if __name__ == '__main__':
    main()
