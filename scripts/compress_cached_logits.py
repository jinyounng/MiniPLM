"""
Cached logits 파일 압축 스크립트

기존 npz 파일들을 압축하여 디스크 공간을 절약합니다.
비압축 → 압축: 파일 크기 ~30-50% 감소

Usage:
    python scripts/compress_cached_logits.py \
        --input-dir /path/to/logits \
        --output-dir /path/to/compressed_logits \
        --start-shard 0 \
        --end-shard 27
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='Compress cached logits files')
    
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input directory containing .npz files')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as input, overwrites)')
    parser.add_argument('--start-shard', type=int, default=0,
                        help='Starting shard index')
    parser.add_argument('--end-shard', type=int, default=-1,
                        help='Ending shard index (-1 for all)')
    parser.add_argument('--backup', action='store_true',
                        help='Keep original files (backup)')
    
    return parser.parse_args()


def get_shard_files(input_dir: str) -> list:
    """입력 디렉토리에서 모든 shard 파일 찾기"""
    npz_files = sorted(glob(os.path.join(input_dir, '*.npz')))
    
    # .temp_mp_results 디렉토리는 제외
    npz_files = [f for f in npz_files if '.temp_mp_results' not in f]
    
    def get_shard_num(path):
        basename = os.path.basename(path)
        name_no_ext = basename.rsplit('.', 1)[0] if '.' in basename else basename
        try:
            # data_0.npz 형식
            if '_' in name_no_ext:
                return int(name_no_ext.split('_')[-1])
            return 0
        except ValueError:
            return 0
    
    npz_files.sort(key=get_shard_num)
    return npz_files


def compress_shard(input_path: str, output_path: str) -> tuple:
    """
    단일 shard 파일 압축
    
    Returns:
        (original_size, compressed_size) in bytes
    """
    # 원본 파일 로드
    data = np.load(input_path, allow_pickle=True)
    
    # 압축하여 저장
    save_dict = {}
    for key in tqdm(data.keys(), desc="   Loading arrays", leave=False):
        save_dict[key] = data[key]
    
    np.savez_compressed(output_path, **save_dict)
    
    # 파일 크기 확인
    original_size = os.path.getsize(input_path)
    compressed_size = os.path.getsize(output_path)
    
    return original_size, compressed_size


def main():
    args = parse_args()
    
    # Output directory 설정
    if args.output_dir is None:
        args.output_dir = args.input_dir
        print(f"⚠️  Output directory not specified, will overwrite original files")
        if not args.backup:
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
    else:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Shard 파일 찾기
    shard_files = get_shard_files(args.input_dir)
    print(f"Found {len(shard_files)} shard files")
    
    # Shard 범위 필터링
    if args.end_shard > 0:
        shard_indices = list(range(args.start_shard, args.end_shard + 1))
    else:
        shard_indices = list(range(args.start_shard, len(shard_files)))
    
    filtered_files = []
    for f in shard_files:
        basename = os.path.basename(f)
        name_no_ext = basename.rsplit('.', 1)[0]
        try:
            if '_' in name_no_ext:
                shard_id = int(name_no_ext.split('_')[-1])
                if shard_id in shard_indices:
                    filtered_files.append((shard_id, f))
        except ValueError:
            continue
    
    filtered_files.sort(key=lambda x: x[0])
    print(f"Processing {len(filtered_files)} shards (from {shard_indices[0]} to {shard_indices[-1]})")
    
    # 압축 진행
    total_original = 0
    total_compressed = 0
    
    for shard_id, input_path in tqdm(filtered_files, desc="Compressing"):
        basename = os.path.basename(input_path)
        output_path = os.path.join(args.output_dir, basename)
        
        try:
            original_size, compressed_size = compress_shard(input_path, output_path)
            total_original += original_size
            total_compressed += compressed_size
            
            # 백업 옵션이 없으면 원본 삭제
            if not args.backup and args.output_dir == args.input_dir:
                os.remove(input_path)
            
            ratio = compressed_size / original_size * 100
            tqdm.write(f"Shard {shard_id}: {original_size/1024**3:.2f}GB → {compressed_size/1024**3:.2f}GB ({ratio:.1f}%)")
            
        except Exception as e:
            print(f"\n❌ Error compressing {basename}: {e}")
            import traceback
            traceback.print_exc()
            if os.path.exists(output_path):
                os.remove(output_path)
    
    # 결과 요약
    print(f"\n✅ Compression completed!")
    print(f"   Original size: {total_original/1024**3:.2f} GB")
    print(f"   Compressed size: {total_compressed/1024**3:.2f} GB")
    print(f"   Compression ratio: {total_compressed/total_original*100:.1f}%")
    print(f"   Space saved: {(total_original-total_compressed)/1024**3:.2f} GB")
    
    if args.backup:
        print(f"   Original files kept in: {args.input_dir}")
    elif args.output_dir == args.input_dir:
        print(f"   Original files replaced in: {args.output_dir}")


if __name__ == '__main__':
    main()
