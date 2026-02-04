#!/usr/bin/env python3
"""
results/eval_harness 아래 모델별 결과를 모아서 CSV로 저장.

각 모델 디렉터리: results/eval_harness/<model_folder>/end_tasks/100000/0shot/log.txt
에서 JSON 블록을 파싱해 task별 점수를 수집하고, 모델별로 한 행이 되도록 CSV 출력.

Usage:
    python scripts/eval/collect_eval_harness_to_csv.py
    python scripts/eval/collect_eval_harness_to_csv.py --results-dir /path/to/eval_harness --out summary.csv
"""

import os
import re
import json
import argparse
import csv
from pathlib import Path


# 표준 task 순서 (log에 나오는 순서와 맞추기)
TASK_COLUMNS = [
    "hellaswag",
    "lambada_openai",
    "winogrande",
    "openbookqa",
    "arc_easy",
    "arc_challenge",
    "piqa",
]


def find_json_blocks(text):
    """텍스트에서 {...} JSON 블록들을 순서대로 추출."""
    blocks = []
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        j = i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            i += 1
            continue
        block = text[i : j + 1]
        try:
            data = json.loads(block)
            blocks.append(data)
        except json.JSONDecodeError:
            pass
        i = j + 1
    return blocks


def parse_log(log_path):
    """log.txt에서 task별 점수 추출. 반환: dict task_name -> score."""
    if not os.path.isfile(log_path):
        return None
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()
    blocks = find_json_blocks(text)
    scores = {}
    for block in blocks:
        for k, v in block.items():
            if k == "avg":
                scores["avg"] = v
                continue
            if isinstance(v, (int, float)):
                scores[k] = v
    return scores if scores else None


def shorten_model_name(folder_name):
    """폴더명을 읽기 쉬운 모델명으로 줄임 (모델별 구분 가능하도록)."""
    s = folder_name
    size = ""
    m = re.search(r"qwen_(\d+(?:\.\d+)?[BM]?)", s, re.I)
    if m:
        size = m.group(1)
    if "offline_kd" in s and "sparse_kd" in s:
        stage = "2stage" if "2stage-sparse" in s or "-2stage-" in s else "scr"
        return f"offline_kd_sparse_{size}_{stage}"
    if "offline_kd" in s and "topk" in s:
        stage = "2stage" if "2stage-topk" in s or "-2stage-" in s else "scr"
        return f"offline_kd_topk_{size}_{stage}"
    if "vanilla_kd" in s:
        return f"vanilla_kd_{size}"
    if "2stage-sft" in s:
        return f"pretrain_sft_{size}"
    if "pretrain" in s and "qwen" in s:
        return f"pretrain_{size}"
    return s[:55]


def main():
    parser = argparse.ArgumentParser(description="Collect eval_harness results by model into CSV")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="eval_harness 루트 (기본: 프로젝트/results/eval_harness)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="출력 CSV 경로 (기본: results/eval_harness/eval_summary.csv)",
    )
    parser.add_argument(
        "--full-name",
        action="store_true",
        help="모델명을 폴더 전체 이름으로 사용",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[2]
    results_dir = Path(args.results_dir) if args.results_dir else base / "results" / "eval_harness"
    out_path = Path(args.out) if args.out else results_dir / "eval_summary.csv"

    if not results_dir.is_dir():
        print("❌ 결과 디렉터리 없음:", results_dir, flush=True)
        return 1

    # 모델 디렉터리: results/eval_harness 아래 1depth 폴더
    model_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    rows = []
    for model_dir in model_dirs:
        log_path = model_dir / "end_tasks" / "100000" / "0shot" / "log.txt"
        if not log_path.exists():
            # 다른 구조 시도
            for p in (model_dir / "end_tasks").rglob("log.txt"):
                log_path = p
                break
            else:
                print("⚠️ log 없음:", model_dir.name, flush=True)
                continue
        scores = parse_log(log_path)
        if not scores:
            print("⚠️ 파싱 결과 없음:", log_path, flush=True)
            continue
        model_name = model_dir.name if args.full_name else shorten_model_name(model_dir.name)
        row = {"model": model_name, "folder": model_dir.name}
        task_vals = []
        for task in TASK_COLUMNS:
            v = scores.get(task, "")
            row[task] = round(v, 4) if isinstance(v, (int, float)) else v
            if isinstance(v, (int, float)):
                task_vals.append(v)
        row["avg"] = round(sum(task_vals) / len(task_vals), 4) if task_vals else ""
        rows.append(row)

    if not rows:
        print("❌ 수집된 결과 없음.", flush=True)
        return 1

    columns = ["model", "folder"] + TASK_COLUMNS + ["avg"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print("✅ 저장:", out_path, "({} 모델)".format(len(rows)), flush=True)
    return 0


if __name__ == "__main__":
    exit(main())
