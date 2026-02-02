#!/usr/bin/env python
"""
캐시 정렬 검증: bin과 H5 캐시가 같은 시퀀스인지 확인.

1) 시그니처 있음 (H5에 /signature/input_prefix): bin 앞 N개 토큰과 직접 비교 → 100% 일치면 진짜 같은 시퀀스.
2) 시그니처 없음: label이 캐시 sparse token_ids에 있는지로 간접 검증 (기존 방식).

Usage:
    python scripts/verify_cache_by_sample.py \
        --data-dir /path/to/bin_data \
        --cached-logits-dir /path/to/cached_logits
"""

import os
import sys
import json
import argparse
import numpy as np
import h5py
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils.bin_fingerprint import get_bin_shard_sizes
from data_utils.indexed_dataset import MMapIndexedDataset


def _get_shard_local(global_idx: int, shard_offsets: list, shard_sizes: list):
    """global_idx -> (shard_id, local_idx)"""
    for shard_id in range(len(shard_offsets)):
        start = shard_offsets[shard_id]
        end = start + shard_sizes[shard_id]
        if start <= global_idx < end:
            return shard_id, global_idx - start
    return len(shard_offsets) - 1, global_idx - shard_offsets[-1]


def read_bin_sequence(global_idx: int, shard_paths: list,
                      shard_sizes: list, shard_offsets: list):
    """Bin에서 global_idx 시퀀스 한 개 읽기 (numpy 1d)"""
    shard_id, local_idx = _get_shard_local(global_idx, shard_offsets, shard_sizes)
    path = shard_paths[shard_id]
    ds = MMapIndexedDataset(path, skip_warmup=True)
    seq = np.asarray(ds[local_idx]).flatten().astype(np.int64)
    return seq


def has_topk_in_h5(shard_path: str) -> bool:
    """H5에 topk 그룹 있는지"""
    try:
        with h5py.File(shard_path, 'r') as f:
            return 'topk' in f and 'topk/token_ids_flat' in f
    except Exception:
        return False


def has_sparse_in_h5(shard_path: str) -> bool:
    """H5에 sparse 그룹 있는지"""
    try:
        with h5py.File(shard_path, 'r') as f:
            return 'sparse' in f and 'sparse/token_ids_flat' in f
    except Exception:
        return False


def read_cache_topk_hdf5(shard_id: int, local_idx: int, shard_paths_dict: dict):
    """H5에서 topk token_ids 한 시퀀스만 읽기 → [seq_len, K]. 없으면 (0,0) 배열."""
    path = shard_paths_dict[shard_id]
    with h5py.File(path, 'r') as f:
        if 'topk' not in f:
            return np.zeros((0, 0), dtype=np.int32)
        s = int(f['topk/seq_lens'][local_idx])
        K = int(f['topk'].attrs['k'])
        if s == 0:
            return np.zeros((0, K), dtype=np.int32)
        offsets = f['topk/seq_offsets']
        start, end = int(offsets[local_idx]), int(offsets[local_idx + 1])
        ids = f['topk/token_ids_flat'][start:end]
        return ids.reshape(s, K).astype(np.int32)


def read_cache_sparse_hdf5(shard_id: int, local_idx: int,
                           method: str, use_method: str, shard_paths_dict: dict):
    """H5 캐시에서 sparse token_ids 한 시퀀스 읽기 → [seq_len, K]"""
    path = shard_paths_dict[shard_id]
    read_method = use_method if method == 'both' else method
    with h5py.File(path, 'r') as f:
        if read_method == 'topk':
            s = int(f['topk/seq_lens'][local_idx])
            K = int(f['topk'].attrs['k'])
            if s == 0:
                return np.zeros((0, K), dtype=np.int32)
            offsets = f['topk/seq_offsets']
            start, end = int(offsets[local_idx]), int(offsets[local_idx + 1])
            ids = f['topk/token_ids_flat'][start:end]
            return ids.reshape(s, K).astype(np.int32)
        else:  # sparse / random
            seq_offsets = f['sparse/seq_offsets']
            token_offsets = f['sparse/token_offsets']
            tok_start = int(seq_offsets[local_idx])
            tok_end = int(seq_offsets[local_idx + 1])
            seq_len = tok_end - tok_start
            if seq_len == 0:
                return np.zeros((0, 1), dtype=np.int32)
            tok_offs = token_offsets[tok_start:tok_end + 1]
            elem_start, elem_end = int(tok_offs[0]), int(tok_offs[-1])
            ids_all = f['sparse/token_ids_flat'][elem_start:elem_end]
            lengths = np.diff(tok_offs).astype(np.int32)
            max_k = int(np.max(lengths))
            token_ids = np.full((seq_len, max_k), -1, dtype=np.int32)
            pos = 0
            for t in range(seq_len):
                k_t = int(lengths[t])
                if k_t > 0:
                    token_ids[t, :k_t] = ids_all[pos:pos + k_t]
                    pos += k_t
            return token_ids


def read_cache_sparse_with_values_hdf5(shard_id: int, local_idx: int, shard_paths_dict: dict):
    """H5에서 sparse token_ids + counts 한 시퀀스 읽기 → (token_ids, values) 둘 다 [seq_len, max_k].
    values = counts (뽑힌 횟수). 표시 시 count 내림차순 정렬용."""
    path = shard_paths_dict[shard_id]
    with h5py.File(path, 'r') as f:
        if 'sparse' not in f or 'sparse/counts_flat' not in f:
            return None, None
        seq_offsets = f['sparse/seq_offsets']
        token_offsets = f['sparse/token_offsets']
        tok_start = int(seq_offsets[local_idx])
        tok_end = int(seq_offsets[local_idx + 1])
        seq_len = tok_end - tok_start
        if seq_len == 0:
            return np.zeros((0, 1), dtype=np.int32), np.zeros((0, 1), dtype=np.float32)
        tok_offs = token_offsets[tok_start:tok_end + 1]
        elem_start, elem_end = int(tok_offs[0]), int(tok_offs[-1])
        ids_all = np.asarray(f['sparse/token_ids_flat'][elem_start:elem_end])
        counts_all = np.asarray(f['sparse/counts_flat'][elem_start:elem_end], dtype=np.float32)
        lengths = np.diff(tok_offs).astype(np.int32)
        max_k = int(np.max(lengths)) if len(lengths) else 1
        token_ids = np.full((seq_len, max_k), -1, dtype=np.int32)
        values = np.zeros((seq_len, max_k), dtype=np.float32)
        pos = 0
        for t in range(seq_len):
            k_t = int(lengths[t])
            if k_t > 0:
                token_ids[t, :k_t] = ids_all[pos:pos + k_t]
                values[t, :k_t] = counts_all[pos:pos + k_t]
                # position별로 count 내림차순 정렬 (많이 나온 순서)
                order = np.argsort(-values[t, :k_t])
                token_ids[t, :k_t] = token_ids[t, :k_t][order]
                values[t, :k_t] = values[t, :k_t][order]
                pos += k_t
        return token_ids, values


def has_signature_in_h5(shard_path: str) -> bool:
    """H5에 시퀀스 시그니처(/signature/input_prefix) 있는지"""
    try:
        with h5py.File(shard_path, 'r') as f:
            return 'signature' in f and 'input_prefix' in f['signature']
    except Exception:
        return False


def read_cache_signature_hdf5(shard_id: int, local_idx: int, shard_paths_dict: dict):
    """H5에서 시그니처(시퀀스 앞 N개 토큰) 한 행 읽기 → 1d array. 없으면 None."""
    path = shard_paths_dict[shard_id]
    with h5py.File(path, 'r') as f:
        if 'signature' not in f or 'input_prefix' not in f['signature']:
            return None
        dset = f['signature/input_prefix']
        return np.asarray(dset[local_idx]).flatten().astype(np.int64)


def main():
    parser = argparse.ArgumentParser(
        description='Verify bin vs cache alignment by loading a few samples and comparing labels to sparse token_ids'
    )
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Bin 디렉터리 (data_*.bin/.idx)')
    parser.add_argument('--cached-logits-dir', type=str, required=True,
                        help='캐시 디렉터리 (metadata.json + data_*.h5)')
    parser.add_argument('--data-prefix', type=str, default='data')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='뽑을 샘플 개수 (0~num_samples-1 인덱스 사용)')
    parser.add_argument('--indices', type=str, default=None,
                        help='지정 시 사용 (쉼표 구분). 없으면 0..num_samples-1')
    parser.add_argument('--max-positions', type=int, default=10,
                        help='시퀀스당 검사할 position 개수')
    parser.add_argument('--show-bin-tokens', type=int, default=15,
                        help='샘플당 bin 앞 N개 토큰 출력')
    parser.add_argument('--out', type=str, default=None,
                        help='결과 저장할 파일 (없으면 stdout)')
    parser.add_argument('--min-match-ratio', type=float, default=0.2,
                        help='시그니처 없을 때: 이 비율 미만이면 실패 (0~1)')
    parser.add_argument('--signature-len', type=int, default=32,
                        help='시그니처 비교 시 사용할 앞 토큰 개수')
    args = parser.parse_args()

    # 1) Bin shard 구조
    bin_paths, bin_sizes = get_bin_shard_sizes(args.data_dir, args.data_prefix)
    if not bin_paths:
        print('❌ Bin shard 없음:', args.data_dir, file=sys.stderr)
        return 1
    bin_offsets = [0]
    for s in bin_sizes[:-1]:
        bin_offsets.append(bin_offsets[-1] + s)
    total_bin = sum(bin_sizes)

    # 2) 캐시 메타데이터 및 포맷
    meta_path = os.path.join(args.cached_logits_dir, 'metadata.json')
    if not os.path.exists(meta_path):
        print('❌ metadata.json 없음:', meta_path, file=sys.stderr)
        return 1
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    cache_offsets = meta['shard_offsets']
    cache_sizes = meta['shard_sizes']
    method = meta.get('method', 'topk')
    use_method = meta.get('kd_method', method) if method == 'both' else method

    # shard_0.h5 / data_0.h5 둘 다 지원 (shard 우선)
    h5_files = sorted(glob(os.path.join(args.cached_logits_dir, 'shard_*.h5')))
    if not h5_files:
        h5_files = sorted(glob(os.path.join(args.cached_logits_dir, 'data_*.h5')))
    if not h5_files:
        print('❌ H5 캐시 없음 (shard_*.h5 또는 data_*.h5)', file=sys.stderr)
        return 1
    file_pattern = 'shard_' if 'shard_' in os.path.basename(h5_files[0]) else 'data_'

    def shard_num(p):
        name = os.path.basename(p).replace(file_pattern, '').replace('.h5', '')
        try:
            return int(name)
        except ValueError:
            return 0
    h5_files.sort(key=shard_num)
    cache_shard_paths = {shard_num(p): p for p in h5_files}
    first_h5 = list(cache_shard_paths.values())[0]
    show_topk = has_topk_in_h5(first_h5)
    show_sparse = has_sparse_in_h5(first_h5)

    # 3) 검사할 인덱스 (기본: 0..num_samples-1)
    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(',') if x.strip()]
    else:
        indices = list(range(min(args.num_samples, total_bin)))
    indices = [i for i in indices if i < total_bin]

    # 4) 진짜 같은 시퀀스인지: 시그니처(입력 앞 N토큰) 있으면 직접 비교
    use_signature = has_signature_in_h5(list(cache_shard_paths.values())[0])
    if use_signature:
        n_ok = 0
        prefix_len = args.signature_len
        for global_idx in indices:
            try:
                seq = read_bin_sequence(
                    global_idx, bin_paths, bin_sizes, bin_offsets
                )
            except Exception as e:
                print(f"  idx {global_idx}: bin 읽기 실패 — {e}", file=sys.stderr)
                return 1
            shard_id, local_idx = _get_shard_local(global_idx, cache_offsets, cache_sizes)
            if shard_id not in cache_shard_paths:
                print(f"  idx {global_idx}: 캐시 shard_id {shard_id} 없음", file=sys.stderr)
                return 1
            sig = read_cache_signature_hdf5(shard_id, local_idx, cache_shard_paths)
            if sig is None:
                print('  시그니처 읽기 실패', file=sys.stderr)
                return 1
            bin_prefix = seq[:min(prefix_len, len(seq))]
            cache_prefix = sig[:min(prefix_len, len(sig))]
            # 캐시는 짧은 시퀀스에 -1 패딩 → 비교 길이는 -1 전까지
            valid = np.where(cache_prefix >= 0)[0]
            cmp_len = min(len(bin_prefix), len(valid)) if len(valid) else 0
            if cmp_len == 0 or not np.array_equal(bin_prefix[:cmp_len], cache_prefix[:cmp_len]):
                print(f'\n❌ 실패: idx {global_idx} 시그니처 불일치. bin과 캐시가 다른 시퀀스.', file=sys.stderr)
                return 1
            n_ok += 1
        print(f'진짜 같은 시퀀스 검증: 인덱스 {indices} 시그니처(앞 {prefix_len}토큰) 100% 일치.')
        print(f'✅ 통과: bin과 캐시가 같은 시퀀스입니다.')
        return 0

    # 5) 시그니처 없음 → 50개 샘플 뽑아서 한눈에 보이게 정리
    def _log(msg=''):
        lines = msg if isinstance(msg, list) else [msg]
        for line in lines:
            print(line, file=fout)

    fout = open(args.out, 'w', encoding='utf-8') if args.out else sys.stdout
    try:
        _log([
            '캐시에 시퀀스 시그니처 없음 → label-in-sparse 검증 (샘플 {}개 정리)'.format(len(indices)),
            '',
        ])
        matches = 0
        total_checks = 0
        rows = []  # (idx, n_checked, n_matched, ratio, len_ok, ok)

        for global_idx in indices:
            try:
                seq = read_bin_sequence(
                    global_idx, bin_paths, bin_sizes, bin_offsets
                )
            except Exception as e:
                _log('--- 샘플 idx={} ---'.format(global_idx))
                _log('  bin 읽기 실패: {}'.format(e))
                _log('')
                rows.append((global_idx, 0, 0, 0.0, False, False))
                continue
            if len(seq) < 2:
                _log('--- 샘플 idx={} ---'.format(global_idx))
                _log('  시퀀스 너무 짧음 (len={})'.format(len(seq)))
                _log('')
                rows.append((global_idx, 0, 0, 0.0, False, False))
                continue
            labels = seq[1:]

            shard_id, local_idx = _get_shard_local(global_idx, cache_offsets, cache_sizes)
            if shard_id not in cache_shard_paths:
                _log('--- 샘플 idx={} ---'.format(global_idx))
                _log('  캐시 shard_id {} 없음'.format(shard_id))
                _log('')
                rows.append((global_idx, 0, 0, 0.0, False, False))
                continue
            try:
                token_ids_primary = read_cache_sparse_hdf5(
                    shard_id, local_idx, method, use_method, cache_shard_paths
                )
            except Exception as e:
                _log('--- 샘플 idx={} ---'.format(global_idx))
                _log('  캐시 읽기 실패: {}'.format(e))
                _log('')
                rows.append((global_idx, 0, 0, 0.0, False, False))
                continue

            token_ids_topk = None
            token_ids_sparse = None
            if show_topk:
                try:
                    token_ids_topk = read_cache_topk_hdf5(
                        shard_id, local_idx, cache_shard_paths
                    )
                except Exception:
                    pass
            if show_sparse:
                try:
                    tid_sparse, val_sparse = read_cache_sparse_with_values_hdf5(
                        shard_id, local_idx, cache_shard_paths
                    )
                    if tid_sparse is not None:
                        token_ids_sparse = tid_sparse  # 이미 position별 count 내림차순 정렬됨
                    else:
                        token_ids_sparse = read_cache_sparse_hdf5(
                            shard_id, local_idx, 'sparse', 'sparse', cache_shard_paths
                        )
                except Exception:
                    token_ids_sparse = None
                    pass

            bin_seq_len = len(seq)
            cache_seq_len = token_ids_primary.shape[0]
            # next-token: 예측 position 수 = len(labels) = bin_seq_len - 1. 캐시도 position당 1행.
            len_ok = (bin_seq_len - 1) == cache_seq_len

            n_show = min(len(labels), token_ids_primary.shape[0], args.max_positions)
            n_matched = 0
            bin_preview = seq[:min(len(seq), args.show_bin_tokens)]
            _log('--- 샘플 idx={} ---'.format(global_idx))
            _log('  길이: bin 시퀀스={}토큰, 예측 position 수={}  |  캐시 position 수={}  →  일치? {}'.format(
                bin_seq_len, len(labels), cache_seq_len, 'O' if len_ok else 'X'
            ))
            _log('  bin (앞 {}토큰): {}'.format(
                len(bin_preview),
                list(map(int, bin_preview))
            ))
            if show_topk or show_sparse:
                _log('  캐시: topk={}, sparse={} (KD용 method={})'.format(
                    show_topk, show_sparse, use_method
                ))
            _log('  position별: [pos] label → topk 있음?/K개 [토큰]  |  sparse 있음?/많이나온1등=label?/N개 [토큰]')
            for pos in range(n_show):
                gt = int(labels[pos])
                row_primary = token_ids_primary[pos, :]
                valid_primary = row_primary[row_primary >= 0]
                hit_primary = len(valid_primary) > 0 and gt in valid_primary
                if hit_primary:
                    n_matched += 1
                total_checks += 1

                topk_str = ''
                if token_ids_topk is not None and pos < token_ids_topk.shape[0]:
                    row_t = token_ids_topk[pos, :]
                    valid_t = row_t[row_t >= 0]
                    k_t = len(valid_t)
                    hit_t = k_t > 0 and gt in valid_t
                    top1_t = k_t > 0 and int(valid_t[0]) == gt
                    topk_str = '  {} top1={}  /  {}개 {}'.format(
                        'O' if hit_t else 'X', 'O' if top1_t else 'X', k_t, list(map(int, valid_t[:8]))
                    )
                else:
                    topk_str = '  -  /  -' if show_topk else ''

                sparse_str = ''
                if token_ids_sparse is not None and pos < token_ids_sparse.shape[0]:
                    row_s = token_ids_sparse[pos, :]
                    valid_s = row_s[row_s >= 0]
                    n_s = len(valid_s)
                    hit_s = n_s > 0 and gt in valid_s
                    # sparse는 이미 count 내림차순 → 첫 토큰이 가장 많이 나온 토큰
                    top1_s = n_s > 0 and int(valid_s[0]) == gt
                    sparse_str = '  {} top1={}  /  {}개 {}'.format(
                        'O' if hit_s else 'X', 'O' if top1_s else 'X', n_s, list(map(int, valid_s[:8]))
                    )
                else:
                    sparse_str = '  -  /  -' if show_sparse else ''

                _log('    [{}] label={}  |  topk:{}  |  sparse:{}'.format(
                    pos, gt, topk_str, sparse_str
                ))
            ratio_row = n_matched / n_show if n_show else 0.0
            matches += n_matched
            ok = ratio_row >= args.min_match_ratio and len_ok
            rows.append((global_idx, n_show, n_matched, ratio_row, len_ok, ok))
            _log('  → 길이일치 {}  KD용({}) 일치: {}/{} ({:.0%})  {}'.format(
                'O' if len_ok else 'X', use_method, n_matched, n_show, ratio_row, 'OK' if ok else 'FAIL'
            ))
            _log('')

        if total_checks < 3:
            _log('⚠️ 검사 횟수 부족 (최소 3회 필요). 인덱스/데이터 확인 필요.')
            if args.out:
                fout.close()
            return 2

        # 요약 테이블 (길이일치 | idx | 검사수 | 일치수 | 비율 | OK?)
        _log('========== 요약 (idx | 길이일치 | 검사수 | 일치수 | 비율 | OK?) ==========')
        for r in rows:
            _log('  {:6d}  |  {}  |  {:3d}  |  {:3d}  | {:5.0%} | {}'.format(
                r[0], 'O' if r[4] else 'X', r[1], r[2], r[3], 'O' if r[5] else 'X'
            ))
        _log('')
        ratio = matches / total_checks
        _log('전체: {} position 중 {} 일치 (비율 {:.2%})'.format(
            total_checks, matches, ratio
        ))

        if ratio < args.min_match_ratio:
            _log('')
            _log('❌ 실패: 일치 비율이 {} 미만. bin과 캐시가 다른 데이터/순서일 가능성 큼.'.format(
                args.min_match_ratio
            ))
            if args.out:
                fout.close()
            return 1
        _log('')
        _log('✅ 통과: 같은 인덱스에서 bin label이 캐시 sparse token_ids에 포함됨.')
        _log('  (진짜 시퀀스 일치 검증을 하려면 캐시 생성 시 /signature/input_prefix 저장 필요)')
    finally:
        if args.out:
            fout.close()
            print('결과 저장: {}'.format(args.out), file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
