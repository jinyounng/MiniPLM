#!/usr/bin/env python
"""
Bin 데이터와 캐시(metadata.json)가 같은 데이터/순서인지 검증.

순서가 다르면 같은 인덱스에 다른 시퀀스가 매핑되어 KD loss가 비정상적으로 튐.
학습 전에 한 번 돌려서 확인하는 용도.

Usage:
    python scripts/verify_cache_alignment.py \
        --data-dir /path/to/bin_data \
        --cached-logits-dir /path/to/cached_logits
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils.bin_fingerprint import verify_cache_alignment


def main():
    parser = argparse.ArgumentParser(
        description='Verify bin data and cached logits are aligned (same data_dir, same order)'
    )
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing data_*.bin/.idx (학습에 쓰는 데이터)')
    parser.add_argument('--cached-logits-dir', type=str, required=True,
                       help='Directory containing metadata.json + data_*.h5 (캐시)')
    parser.add_argument('--data-prefix', type=str, default='data',
                       help='Bin 파일 prefix (default: data → data_0.bin, data_1.bin)')
    args = parser.parse_args()

    ok, msg = verify_cache_alignment(
        args.data_dir,
        args.cached_logits_dir,
        data_prefix=args.data_prefix,
    )
    if ok:
        print("✅", msg)
        return 0
    print("❌", msg, file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
