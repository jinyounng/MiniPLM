"""
Bin 데이터 fingerprint로 캐시 정렬 검증

캐시(metadata.json)의 data_dir + shard_sizes가 학습용 bin 데이터와 동일한지
확인할 때 사용. 순서/경로가 다르면 KD loss가 비정상적으로 튐.
"""

import os
import struct
import json
import hashlib
from glob import glob
from typing import List, Tuple


# MMIDIDX 포맷 (distributed_indexed.Index)
_IDX_MAGIC = b'MMIDIDX\x00\x00'
_LEN_OFFSET = 9 + 8 + 1  # magic + version + dtype


def read_idx_length(idx_path: str) -> int:
    """.idx 파일에서 시퀀스 개수(_len)만 읽기 (전체 로드 없음)"""
    with open(idx_path, 'rb') as f:
        magic = f.read(9)
        if magic != _IDX_MAGIC:
            raise ValueError(f"Not MMIDIDX format: {idx_path}")
        f.read(8)   # version
        f.read(1)   # dtype
        length, _ = struct.unpack('<QQ', f.read(16))  # _len, _doc_count
    return length


def get_bin_shard_paths(data_dir: str, prefix: str = 'data') -> List[str]:
    """Bin 디렉토리에서 shard 경로 목록 (data_0, data_1, ... 순)"""
    bin_files = sorted(glob(os.path.join(data_dir, f'{prefix}_*.bin')))
    shard_paths = []
    for bin_file in bin_files:
        path = bin_file[:-4]
        if os.path.exists(path + '.idx'):
            shard_paths.append(path)

    def shard_num(p):
        basename = os.path.basename(p)
        name = basename.rsplit('.', 1)[0] if '.' in basename else basename
        try:
            return int(name.split('_')[-1])
        except ValueError:
            return 0

    shard_paths.sort(key=shard_num)
    return shard_paths


def get_bin_shard_sizes(data_dir: str, prefix: str = 'data') -> Tuple[List[str], List[int]]:
    """
    Bin shard 경로와 각 shard의 시퀀스 개수 반환.
    Returns:
        (shard_paths, shard_sizes)
    """
    paths = get_bin_shard_paths(data_dir, prefix)
    sizes = []
    for p in paths:
        idx_path = p + '.idx'
        sizes.append(read_idx_length(idx_path))
    return paths, sizes


def compute_data_fingerprint(data_dir: str, shard_sizes: List[int]) -> str:
    """
    (data_dir 실경로 + shard_sizes)로 fingerprint 계산.
    캐시 저장 시 이 값을 metadata.json에 넣고,
    학습 시 같은 값이면 bin과 캐시가 같은 데이터/순서임을 보장.
    """
    data_dir_resolved = os.path.realpath(data_dir)
    payload = data_dir_resolved + json.dumps(shard_sizes, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def verify_cache_alignment(
    data_dir: str,
    cached_logits_dir: str,
    data_prefix: str = 'data',
) -> Tuple[bool, str]:
    """
    Bin 데이터와 캐시(metadata.json)가 같은 데이터/순서인지 검증.

    Returns:
        (ok, message)
    """
    metadata_path = os.path.join(cached_logits_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        return False, f"metadata.json not found: {metadata_path}"

    with open(metadata_path, 'r') as f:
        meta = json.load(f)

    cache_data_dir = meta.get('data_dir')
    if not cache_data_dir:
        return False, "metadata.json has no 'data_dir' (old cache?). Re-run cache script to add fingerprint."

    # 1) data_dir 실경로 비교
    data_dir_resolved = os.path.realpath(data_dir)
    cache_data_dir_resolved = os.path.realpath(cache_data_dir)
    if data_dir_resolved != cache_data_dir_resolved:
        return False, (
            f"data_dir 불일치 → 캐시와 학습용 bin이 다른 데이터일 수 있음.\n"
            f"  학습 data_dir:  {data_dir_resolved}\n"
            f"  캐시 data_dir: {cache_data_dir_resolved}\n"
            f"같은 corpus/순서로 만든 캐시를 쓰거나, 해당 data_dir로 캐시를 다시 생성하세요."
        )

    # 2) data_fingerprint 있으면 shard_sizes까지 비교
    cache_fp = meta.get('data_fingerprint')
    if cache_fp:
        paths, sizes = get_bin_shard_sizes(data_dir, data_prefix)
        current_fp = compute_data_fingerprint(data_dir, sizes)
        if current_fp != cache_fp:
            cache_sizes = meta.get('shard_sizes', [])
            return False, (
                f"data_fingerprint 불일치 → bin shard 구조가 캐시와 다름 (순서/개수 변경됨).\n"
                f"  현재 bin shard_sizes (앞 5개): {sizes[:5]} ... (총 {len(sizes)} shards)\n"
                f"  캐시 shard_sizes (앞 5개):     {cache_sizes[:5]} ... (총 {len(cache_sizes)} shards)\n"
                f"같은 bin 데이터로 캐시를 다시 생성하세요."
            )

    return True, "OK: data_dir 및 data_fingerprint 일치 (같은 데이터/순서)"
