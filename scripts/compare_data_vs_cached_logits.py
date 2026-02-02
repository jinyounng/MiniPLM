#!/usr/bin/env python3
"""
원본 학습 데이터 샘플 수 vs 캐시된 logits 샘플 수 비교.

학습 시 사용하는 data_dir(또는 data_path)의 총 시퀀스 수와
캐시된 logits 디렉터리 metadata.json의 total_sequences를 비교해
차이를 출력합니다. (캐시/학습 데이터 불일치로 "No sparse logits found" 발생 여부 확인용)

Usage:
  # 학습 데이터: data_dir 아래 data_0.bin/idx, data_1.bin/idx, ...
  python scripts/compare_data_vs_cached_logits.py \
    --data-dir /path/to/pile_dataset \
    --cached-logits-dir /path/to/miniplm_refined_corpus_logits_topk

  # 단일 shard (예: data_0 한 개만)
  python scripts/compare_data_vs_cached_logits.py \
    --data-path /path/to/pile_dataset/data_0 \
    --cached-logits-dir /path/to/cached_logits
"""

import os
import sys
import json
import argparse
from glob import glob
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.indexed_dataset import MMapIndexedDataset


def get_shard_paths(data_dir: str, prefix: str = "data"):
    """data_dir 아래 prefix_*.bin/idx 쌍 목록 반환 (정렬)."""
    bin_files = sorted(glob(os.path.join(data_dir, f"{prefix}_*.bin")))
    shard_paths = []
    for bin_file in bin_files:
        path = bin_file[:-4]  # .bin 제거
        if os.path.exists(path + ".idx"):
            shard_paths.append(path)
    def shard_num(p):
        name = os.path.basename(p).rsplit(".", 1)[0] if "." in os.path.basename(p) else os.path.basename(p)
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return 0
    shard_paths.sort(key=shard_num)
    return shard_paths


def count_sequences_data_dir(data_dir: str, prefix: str = "data", skip_warmup: bool = True, show_progress: bool = True):
    """data_dir 내 모든 shard 시퀀스 수 합산."""
    shard_paths = get_shard_paths(data_dir, prefix)
    if not shard_paths:
        return 0, [], []
    sizes = []
    it = tqdm(shard_paths, desc="Counting data shards") if show_progress else shard_paths
    for path in it:
        ds = MMapIndexedDataset(path, skip_warmup=skip_warmup)
        sizes.append(len(ds))
        del ds
    total = sum(sizes)
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)
    return total, sizes, offsets


def count_sequences_single_path(data_path: str, skip_warmup: bool = True):
    """단일 shard 경로(예: .../data_0)의 시퀀스 수."""
    if not os.path.exists(data_path + ".idx"):
        return 0
    ds = MMapIndexedDataset(data_path, skip_warmup=skip_warmup)
    n = len(ds)
    del ds
    return n


def get_cached_logits_total(cached_logits_dir: str):
    """metadata.json에서 total_sequences 및 요약 반환."""
    meta_path = os.path.join(cached_logits_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return None, None
    with open(meta_path, "r") as f:
        meta = json.load(f)
    total = meta.get("total_sequences")
    shard_sizes = meta.get("shard_sizes", [])
    return total, meta


def main():
    parser = argparse.ArgumentParser(
        description="Compare original data sample count vs cached logits sample count"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="학습용 원본 데이터 디렉터리 (data_0.bin/idx, data_1.bin/idx, ...)",
    )
    group.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="단일 shard 경로 (예: .../data_0, .bin/.idx 제외)",
    )
    parser.add_argument(
        "--data-prefix",
        type=str,
        default="data",
        help="--data-dir 사용 시 파일 접두사 (default: data)",
    )
    parser.add_argument(
        "--cached-logits-dir",
        type=str,
        required=True,
        help="캐시된 logits 디렉터리 (metadata.json 포함)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Shard 카운트 시 tqdm 비표시",
    )
    args = parser.parse_args()

    # 1) 원본 데이터 샘플 수
    if args.data_dir:
        data_total, data_sizes, data_offsets = count_sequences_data_dir(
            args.data_dir, args.data_prefix, show_progress=not args.no_progress
        )
        data_desc = f"{args.data_dir} (prefix={args.data_prefix}, shards={len(data_sizes)})"
    else:
        data_total = count_sequences_single_path(args.data_path)
        data_sizes = [data_total] if data_total else []
        data_offsets = [0]
        data_desc = args.data_path

    # 2) 캐시된 logits 샘플 수
    cached_total, cached_meta = get_cached_logits_total(args.cached_logits_dir)
    if cached_total is None:
        print("ERROR: metadata.json not found or invalid in --cached-logits-dir")
        sys.exit(1)

    # 3) 출력
    print("=" * 60)
    print("원본 데이터 vs 캐시 logits 샘플 수 비교")
    print("=" * 60)
    print(f"  원본 데이터:  {data_desc}")
    print(f"  원본 샘플 수: {data_total:,}")
    print()
    print(f"  캐시 logits:  {args.cached_logits_dir}")
    print(f"  캐시 샘플 수: {cached_total:,}")
    if cached_meta:
        print(f"  캐시 method:  {cached_meta.get('method', 'N/A')}")
        print(f"  캐시 shards: {len(cached_meta.get('shard_sizes', []))}")
    print()
    diff = data_total - cached_total
    print(f"  차이 (원본 - 캐시): {diff:,}")
    if diff > 0:
        print(f"  → 원본이 {diff:,}개 더 많음. 이 구간은 sparse logits 없이 LM loss만 사용됩니다.")
    elif diff < 0:
        print(f"  → 캐시가 {-diff:,}개 더 많음. (원본 데이터가 캐시보다 적음)")
    else:
        print("  → 일치합니다.")
    print("=" * 60)


if __name__ == "__main__":
    main()
