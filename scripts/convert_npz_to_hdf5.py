#!/usr/bin/env python3
"""
NPZ -> HDF5 (v_final_one: 실제로 메모리 안 터지게)

핵심 전략
1) NPZ(zip) 그대로 np.load로 뽑아쓰면, 내부 .npy 디코드/버퍼 때문에 피크가 튀는 케이스가 있음.
2) 그래서 먼저 NPZ를 "그대로" 디스크에 unzip 해서 .npy 파일로 만든 뒤,
3) .npy는 mmap_mode='r'로 읽어서 (가능한 건 memmap으로) streaming write.
4) token_offsets는 토큰 chunk 단위 cumsum + flush.
5) sparse payload도 토큰 chunk 단위로 write해서 "buf_ids/concat 폭발" 없음.

요구되는 입력 키:
- method, seq_lens, topk_k, sparse_num_samples
- topk_token_ids, topk_probs
- sparse_token_ids, sparse_counts, sparse_lengths

출력 구조:
    /topk/token_ids_flat (int32)
    /topk/probs_flat     (float16)
    /topk/seq_offsets    (int64)  # element offsets
    /topk/seq_lens       (int32)  # stored length (tk.shape[0])
    /topk attrs: k

    /sparse/token_ids_flat (int32)
    /sparse/counts_flat    (uint8)
    /sparse/seq_offsets    (int64)  # token offsets base (token index, not element)
    /sparse/token_offsets  (int64)  # element offsets per token (len = total_tokens+1)
    /sparse attrs: num_samples

    /meta/seq_lens (int32) # original metadata
"""

import os
import argparse
import zipfile
import shutil
import numpy as np
import h5py
from glob import glob
from tqdm import tqdm
import json
import gc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--start-shard", type=int, default=0)
    p.add_argument("--end-shard", type=int, default=None)
    p.add_argument("--compression", type=str, default="none", choices=["none", "gzip", "lzf"])
    p.add_argument("--unzip-cache-dir", type=str, default=None,
                   help="NPZ를 풀어둘 디렉토리. 지정 안 하면 output-dir 아래에 shard별로 생성.")
    p.add_argument("--topk_chunk_elems", type=int, default=1_000_000,
                   help="TopK flat write chunk (elements).")
    p.add_argument("--sparse_chunk_tokens", type=int, default=4096,
                   help="Sparse write chunk (tokens).")
    p.add_argument("--tokoff_buf", type=int, default=1_000_000,
                   help="token_offsets 버퍼 길이 (int64 원소 개수).")
    p.add_argument("--keep-unzip-cache", action="store_true",
                   help="변환 후 unzip 캐시를 삭제하지 않음 (기본: 삭제)")
    p.add_argument("--delete-npz", action="store_true",
                   help="변환 성공 후 원본 NPZ 파일 삭제 (디스크 절약)")
    return p.parse_args()


def shard_id_from_path(npz_path: str) -> int:
    base = os.path.basename(npz_path)
    return int(base.replace("data_", "").replace(".npz", ""))


def ensure_unzipped(npz_path: str, unzip_dir: str) -> str:
    """
    npz(zip) 안의 파일들을 unzip_dir에 풀어놓고, 'npz_manifest.json' 생성.
    이미 manifest가 있으면 재사용.
    """
    os.makedirs(unzip_dir, exist_ok=True)
    manifest_path = os.path.join(unzip_dir, "npz_manifest.json")
    if os.path.exists(manifest_path):
        print(f"  [unzip] using cached: {unzip_dir}")
        return unzip_dir

    # 안전: zipfile은 스트리밍으로 파일을 디스크에 씀 (RAM 거의 안 씀)
    with zipfile.ZipFile(npz_path, "r") as zf:
        names = zf.namelist()
        for name in tqdm(names, desc="  Extracting NPZ", leave=False):
            zf.extract(name, unzip_dir)

    with open(manifest_path, "w") as f:
        json.dump({"npz": npz_path, "files": names}, f, indent=2)

    return unzip_dir


def npy_load_maybe_memmap(path: str, allow_pickle: bool = True):
    """
    .npy는 mmap 시도. object dtype이면 mmap이 의미 없거나 실패 가능 -> 자동 fallback.
    """
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=allow_pickle)
        return arr
    except Exception:
        return np.load(path, allow_pickle=allow_pickle)


def get_sparse_actual_len(sp_lens, sp_ids, sp_counts):
    return min(len(sp_lens), sp_ids.shape[0], sp_counts.shape[0])


def convert_shard(npz_path: str, output_dir: str, unzip_root: str,
                  compression: str, topk_chunk_elems: int,
                  sparse_chunk_tokens: int, tokoff_buf: int):
    sid = shard_id_from_path(npz_path)
    out_path = os.path.join(output_dir, f"shard_{sid}.h5")

    unzip_dir = os.path.join(unzip_root, f"shard_{sid}_unzipped")
    print(f"\n[Shard {sid}] unzip -> {unzip_dir}")
    ensure_unzipped(npz_path, unzip_dir)

    # NPZ 내부 파일명은 보통 'arr_0.npy' 이런 식이 아니라, key 기반 '<key>.npy' 로 저장됨.
    # numpy가 저장한 npz는 key+'.npy' 형태가 맞음.
    def p(key): return os.path.join(unzip_dir, f"{key}.npy")

    # keys
    method = str(np.load(p("method"), allow_pickle=True).item())
    if method != "both":
        raise RuntimeError(f"method={method} not supported (expected 'both')")

    meta_seq_lens = np.load(p("seq_lens")).astype(np.int32)
    num_seq = len(meta_seq_lens)

    topk_k = int(np.load(p("topk_k")).item())
    sparse_num_samples = int(np.load(p("sparse_num_samples")).item())

    # main arrays (가능한 건 memmap)
    topk_ids_obj   = npy_load_maybe_memmap(p("topk_token_ids"), allow_pickle=True)
    topk_probs_obj = npy_load_maybe_memmap(p("topk_probs"), allow_pickle=True)

    sparse_ids_obj    = npy_load_maybe_memmap(p("sparse_token_ids"), allow_pickle=True)
    sparse_counts_obj = npy_load_maybe_memmap(p("sparse_counts"), allow_pickle=True)
    sparse_lens_obj   = npy_load_maybe_memmap(p("sparse_lengths"), allow_pickle=True)

    comp_payload = None if compression == "none" else compression

    print(f"[Shard {sid}] seqs={num_seq:,} topk_k={topk_k} sparse_num_samples={sparse_num_samples}")

    # -------------------------
    # Pass1 (size only, shape guard)
    # -------------------------
    topk_seq_offsets = np.zeros(num_seq + 1, dtype=np.int64)
    topk_seq_lens = np.zeros(num_seq, dtype=np.int32)

    sparse_seq_offsets = np.zeros(num_seq + 1, dtype=np.int64)  # token index base
    total_tokens = 0
    total_sparse_elems = 0

    for i in tqdm(range(num_seq), desc=f"Pass1 shard{sid}", leave=False):
        tk = topk_ids_obj[i]
        if getattr(tk, "ndim", 0) > 1:
            s, k = tk.shape
            if k != topk_k:
                raise RuntimeError(f"TopK K mismatch at seq {i}: {k} != {topk_k}")
            topk_seq_lens[i] = s
            topk_seq_offsets[i+1] = topk_seq_offsets[i] + s * k
        else:
            topk_seq_lens[i] = 0
            topk_seq_offsets[i+1] = topk_seq_offsets[i]

        sp_lens = sparse_lens_obj[i]
        sp_ids = sparse_ids_obj[i]
        sp_counts = sparse_counts_obj[i]
        actual = get_sparse_actual_len(sp_lens, sp_ids, sp_counts)
        sp_lens = sp_lens[:actual]

        sparse_seq_offsets[i+1] = sparse_seq_offsets[i] + actual
        total_tokens += actual
        total_sparse_elems += int(np.sum(sp_lens))

    total_topk = int(topk_seq_offsets[-1])

    # -------------------------
    # Pass2 (streaming write)
    # -------------------------
    os.makedirs(output_dir, exist_ok=True)
    CHUNK_ELEMS = 256 * 1024

    with h5py.File(out_path, "w") as f:
        # TopK
        g_topk = f.create_group("topk")
        ds_tk_ids = g_topk.create_dataset(
            "token_ids_flat", shape=(total_topk,), dtype=np.int32,
            chunks=(min(CHUNK_ELEMS, max(1, total_topk)),) if total_topk > 0 else None,
            compression=comp_payload
        )
        ds_tk_probs = g_topk.create_dataset(
            "probs_flat", shape=(total_topk,), dtype=np.float16,
            chunks=(min(CHUNK_ELEMS, max(1, total_topk)),) if total_topk > 0 else None,
            compression=comp_payload
        )
        g_topk.create_dataset("seq_offsets", data=topk_seq_offsets, compression=None)
        g_topk.create_dataset("seq_lens", data=topk_seq_lens, compression=None)
        g_topk.attrs["k"] = topk_k

        # Sparse
        g_sp = f.create_group("sparse")
        ds_sp_ids = g_sp.create_dataset(
            "token_ids_flat", shape=(total_sparse_elems,), dtype=np.int32,
            chunks=(min(CHUNK_ELEMS, max(1, total_sparse_elems)),) if total_sparse_elems > 0 else None,
            compression=comp_payload
        )
        ds_sp_cts = g_sp.create_dataset(
            "counts_flat", shape=(total_sparse_elems,), dtype=np.uint8,
            chunks=(min(CHUNK_ELEMS, max(1, total_sparse_elems)),) if total_sparse_elems > 0 else None,
            compression=comp_payload
        )
        g_sp.create_dataset("seq_offsets", data=sparse_seq_offsets, compression=None)
        ds_tokoff = g_sp.create_dataset(
            "token_offsets", shape=(total_tokens + 1,), dtype=np.int64,
            chunks=(min(CHUNK_ELEMS, max(1, total_tokens + 1)),) if total_tokens > 0 else None,
            compression=None
        )
        g_sp.attrs["num_samples"] = sparse_num_samples

        # Meta
        g_meta = f.create_group("meta")
        g_meta.create_dataset("seq_lens", data=meta_seq_lens, compression=None)
        g_meta.attrs["num_sequences"] = num_seq
        g_meta.attrs["shard_id"] = sid
        g_meta.attrs["method"] = "both"

        # streaming positions
        topk_pos = 0
        sparse_pos = 0

        # token_offsets streaming buffer
        tokbuf = np.empty(tokoff_buf, dtype=np.int64)
        tokbuf_i = 0
        tokoff_write = 0
        running = np.int64(0)

        # token_offsets[0]=0
        tokbuf[0] = 0
        tokbuf_i = 1

        def flush_tokbuf():
            nonlocal tokbuf_i, tokoff_write
            if tokbuf_i > 0:
                ds_tokoff[tokoff_write: tokoff_write + tokbuf_i] = tokbuf[:tokbuf_i]
                tokoff_write += tokbuf_i
                tokbuf_i = 0

        for i in tqdm(range(num_seq), desc=f"Pass2 shard{sid}", leave=False):
            # ---- TopK: chunk write, ravel/view 우선 ----
            tk_ids = topk_ids_obj[i]
            tk_probs = topk_probs_obj[i]
            if getattr(tk_ids, "ndim", 0) > 1:
                # view 가능하면 view. copy 최소화.
                flat_ids = np.asarray(tk_ids).reshape(-1)
                flat_probs = np.asarray(tk_probs).reshape(-1)
                n = len(flat_ids)

                for j in range(0, n, topk_chunk_elems):
                    end = min(j + topk_chunk_elems, n)
                    m = end - j
                    ds_tk_ids[topk_pos:topk_pos+m] = flat_ids[j:end]
                    ds_tk_probs[topk_pos:topk_pos+m] = flat_probs[j:end].astype(np.float16, copy=False)
                    topk_pos += m

            # ---- Sparse ----
            sp_ids = sparse_ids_obj[i]
            sp_cts = sparse_counts_obj[i]
            sp_lens = sparse_lens_obj[i]
            actual = get_sparse_actual_len(sp_lens, sp_ids, sp_cts)
            sp_lens = sp_lens[:actual]

            # token_offsets: chunk cumsum + flush
            if actual > 0:
                lens64 = sp_lens.astype(np.int64, copy=False)
                cs = np.cumsum(lens64, dtype=np.int64) + running
                running = cs[-1]

                # cs를 tokbuf에 밀어넣기
                idx = 0
                while idx < len(cs):
                    space = len(tokbuf) - tokbuf_i
                    take = min(space, len(cs) - idx)
                    tokbuf[tokbuf_i: tokbuf_i + take] = cs[idx: idx + take]
                    tokbuf_i += take
                    idx += take
                    if tokbuf_i == len(tokbuf):
                        flush_tokbuf()

            # sparse payload: token chunk 단위로 write (버퍼 폭발 없음)
            t = 0
            while t < actual:
                te = min(t + sparse_chunk_tokens, actual)
                chunk_lens = sp_lens[t:te]
                chunk_k = int(np.sum(chunk_lens))

                if chunk_k > 0:
                    # chunk_k만큼만 작은 버퍼
                    buf_ids = np.empty(chunk_k, dtype=np.int32)
                    buf_cts = np.empty(chunk_k, dtype=np.uint8)
                    pos = 0
                    for tt in range(t, te):
                        k_t = int(sp_lens[tt])
                        if k_t:
                            buf_ids[pos:pos+k_t] = sp_ids[tt, :k_t]
                            buf_cts[pos:pos+k_t] = sp_cts[tt, :k_t]
                            pos += k_t

                    ds_sp_ids[sparse_pos:sparse_pos+chunk_k] = buf_ids
                    ds_sp_cts[sparse_pos:sparse_pos+chunk_k] = buf_cts
                    sparse_pos += chunk_k

                t = te

        # flush leftover token_offsets
        flush_tokbuf()

        # ---- 검증 (필수) ----
        assert tokbuf_i == 0, f"token_offsets buffer not flushed: {tokbuf_i}"
        assert tokoff_write == total_tokens + 1, f"token_offsets count mismatch: {tokoff_write} != {total_tokens + 1}"
        assert int(running) == total_sparse_elems, f"running_offset mismatch: {running} != {total_sparse_elems}"
        assert sparse_pos == total_sparse_elems, f"sparse_pos mismatch: {sparse_pos} != {total_sparse_elems}"
        assert topk_pos == total_topk, f"topk_pos mismatch: {topk_pos} != {total_topk}"
        # 마지막 값 체크
        last = int(ds_tokoff[total_tokens])
        assert last == total_sparse_elems, f"token_offsets[-1] mismatch: {last} != {total_sparse_elems}"

    # stats
    in_sz = os.path.getsize(npz_path)
    out_sz = os.path.getsize(out_path)
    print(f"[Shard {sid}] DONE: {in_sz/1e9:.1f}GB -> {out_sz/1e9:.1f}GB ({out_sz/in_sz*100:.1f}%)")
    return num_seq, out_sz, unzip_dir


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    npz_files = sorted(glob(os.path.join(args.input_dir, "data_*.npz")))
    if not npz_files:
        raise SystemExit(f"No npz files in {args.input_dir}")

    def sid(p): return shard_id_from_path(p)
    shards = sorted([(sid(p), p) for p in npz_files], key=lambda x: x[0])
    if args.end_shard is not None:
        shards = [(i, p) for i, p in shards if args.start_shard <= i < args.end_shard]
    else:
        shards = [(i, p) for i, p in shards if i >= args.start_shard]

    unzip_root = args.unzip_cache_dir or os.path.join(args.output_dir, "_npz_unzipped_cache")
    os.makedirs(unzip_root, exist_ok=True)

    # metadata.json 복사 + format 기록
    meta_in = os.path.join(args.input_dir, "metadata.json")
    if os.path.exists(meta_in):
        with open(meta_in, "r") as f:
            meta = json.load(f)
        meta["format"] = "hdf5_flat_final_one"
        meta["compression"] = args.compression
        meta["unzip_cache_dir"] = unzip_root
        with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    total_seq = 0
    total_size = 0
    for i, npz_path in shards:
        n, sz, unzip_dir = convert_shard(
            npz_path, args.output_dir, unzip_root,
            args.compression, args.topk_chunk_elems,
            args.sparse_chunk_tokens, args.tokoff_buf
        )
        total_seq += n
        total_size += sz
        
        # 디스크 절약: unzip 캐시 즉시 삭제 (기본 동작)
        if not args.keep_unzip_cache and os.path.isdir(unzip_dir):
            shutil.rmtree(unzip_dir, ignore_errors=True)
            print(f"  [cleanup] removed unzip cache")
        
        # 디스크 절약: 원본 NPZ 삭제 (옵션)
        if args.delete_npz and os.path.exists(npz_path):
            npz_size = os.path.getsize(npz_path) / 1e9
            os.remove(npz_path)
            print(f"  [cleanup] deleted original NPZ ({npz_size:.1f}GB)")
        
        gc.collect()

    # 모든 변환 완료 후 빈 캐시 디렉토리 정리
    if not args.keep_unzip_cache and os.path.isdir(unzip_root):
        try:
            os.rmdir(unzip_root)  # 빈 경우에만 삭제
        except OSError:
            pass

    print(f"\nALL DONE: seqs={total_seq:,}, size={total_size/1e12:.2f} TB")


if __name__ == "__main__":
    main()
