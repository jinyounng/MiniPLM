#!/bin/bash
# Eval AE / RVQ: hidden → logits 복원 품질 (Logit MSE/KL, Hidden MSE/Cosine)
#
# (1) bash 파일 안에서 여러 모델 지정해서 한 번에 평가:
#     아래 EVAL_CKPTS / EVAL_LATENT_DIMS 수정 후 인자 없이 실행
# (2) 인자로 단일 모델 평가:
#     bash eval_ae_rvq_metrics.sh ae /path/to/best_ae_ld25.pt
#     bash eval_ae_rvq_metrics.sh /path/to/best_ae_zonly_ld40.pt

set -e

# -----------------------------------------------------------------------------
# 여기서 평가할 모델 경로 (여러 개 가능). 지정하면 인자 없이 실행 시 이 목록으로 평가
# EVAL_LATENT_DIMS: ae/ae_zonly용 latent_dim (개수 맞추거나 비우면 파일명/기본값으로 추론)
# -----------------------------------------------------------------------------
EVAL_CKPTS=(
  "/home/jiwonyoon/data1/projects/MiniPLM/checkpoints/AE/logit_only/layernorm/best_ae_ld25.pt"
  "/home/jiwonyoon/data1/projects/MiniPLM/checkpoints/AE/logit_only/layernorm/best_ae_ld40.pt"
  "/home/jiwonyoon/data1/projects/MiniPLM/checkpoints/AE/zonly/best_ae_zonly_ld25.pt"
  "/home/jiwonyoon/data1/projects/MiniPLM/checkpoints/AE/zonly/best_ae_zonly_ld40.pt"
  "/home/jiwonyoon/data1/projects/MiniPLM/checkpoints/AE/RVQ/best_rvq_s25_c1024_d1024_enc3_dec3.pt"
  "/home/jiwonyoon/data1/projects/MiniPLM/checkpoints/AE/RVQ/best_rvq_s25_c1024_d1024_enc4_dec4.pt"
)
EVAL_LATENT_DIMS=(25 40 25 40 25 25)   # 비우면 () → 각 ckpt마다 기본값/파일명 사용

# -----------------------------------------------------------------------------
BASE_PATH=${BASE_PATH:-/home/jiwonyoon/data1/projects/MiniPLM}
TEACHER_PATH=${TEACHER_PATH:-/home/jiwonyoon/data1/checkpoints/qwen/7B}
DATA_PATH=${DATA_PATH:-/home/jiwonyoon/data1/data/pile_dataset/data_1}
export PYTHONPATH="${BASE_PATH}"
MAX_SAMPLES=${MAX_SAMPLES:-2000}
MAX_LENGTH=${MAX_LENGTH:-512}
BATCH_SIZE=${BATCH_SIZE:-32}
# -----------------------------------------------------------------------------

run_one() {
  local MODEL_TYPE="$1"
  local CKPT="$2"
  local LATENT_ARG="${3:-}"
  local LATENT_DIM
  if [[ "$MODEL_TYPE" == "ae" ]]; then
    LATENT_DIM=${LATENT_ARG:-25}
    python "${BASE_PATH}/scripts/AE/train/eval_ae_rvq_metrics.py" \
      --model_type ae \
      --checkpoint_path "$CKPT" \
      --teacher_path "$TEACHER_PATH" \
      --data_path "$DATA_PATH" \
      --latent_dim "$LATENT_DIM" \
      --max_samples "$MAX_SAMPLES" \
      --max_length "$MAX_LENGTH" \
      --batch_size "$BATCH_SIZE"
  elif [[ "$MODEL_TYPE" == "ae_zonly" ]]; then
    LATENT_DIM=${LATENT_ARG:-40}
    python "${BASE_PATH}/scripts/AE/train/eval_ae_rvq_metrics.py" \
      --model_type ae_zonly \
      --checkpoint_path "$CKPT" \
      --teacher_path "$TEACHER_PATH" \
      --data_path "$DATA_PATH" \
      --latent_dim "$LATENT_DIM" \
      --max_samples "$MAX_SAMPLES" \
      --max_length "$MAX_LENGTH" \
      --batch_size "$BATCH_SIZE"
  elif [[ "$MODEL_TYPE" == "rvq" ]]; then
    NUM_STAGES=${NUM_STAGES:-4}
    NUM_CODES=${NUM_CODES:-1024}
    COMPRESSED_DIM=${COMPRESSED_DIM:-1024}
    python "${BASE_PATH}/scripts/AE/train/eval_ae_rvq_metrics.py" \
      --model_type rvq \
      --checkpoint_path "$CKPT" \
      --teacher_path "$TEACHER_PATH" \
      --data_path "$DATA_PATH" \
      --num_stages "$NUM_STAGES" \
      --num_codes "$NUM_CODES" \
      --compressed_dim "$COMPRESSED_DIM" \
      --max_samples "$MAX_SAMPLES" \
      --max_length "$MAX_LENGTH" \
      --batch_size "$BATCH_SIZE"
  else
    echo "Unknown model_type: $MODEL_TYPE" >&2
    return 1
  fi
}

infer_type() {
  local p="$1"
  if [[ "$p" == *zonly* ]]; then echo ae_zonly; elif [[ "$p" == *rvq* ]]; then echo rvq; else echo ae; fi
}

# 목록이 있으면 목록으로 평가 (인자 무시)
if [[ ${#EVAL_CKPTS[@]} -gt 0 ]]; then
  for i in "${!EVAL_CKPTS[@]}"; do
    ckpt="${EVAL_CKPTS[$i]}"
    [[ -f "$ckpt" ]] || { echo "Skip (not found): $ckpt"; continue; }
    model_type=$(infer_type "$ckpt")
    latent=""
    [[ ${#EVAL_LATENT_DIMS[@]} -gt 0 && ${#EVAL_LATENT_DIMS[@]} -gt $i ]] && latent="${EVAL_LATENT_DIMS[$i]}"
    echo "===== Eval [$((i+1))/${#EVAL_CKPTS[@]}] $model_type $ckpt (latent=${latent:-auto}) ====="
    run_one "$model_type" "$ckpt" "$latent"
  done
  exit 0
fi

# 인자로 단일 모델
if [[ $# -eq 1 && ("$1" == */* || "$1" == *.pt) ]]; then
  CKPT="$1"
  MODEL_TYPE=$(infer_type "$1")
  run_one "$MODEL_TYPE" "$CKPT" ""
elif [[ $# -ge 2 ]]; then
  run_one "$1" "$2" "${3:-}"
else
  echo "Usage: $0 [ae|ae_zonly|rvq] <checkpoint_path> [latent_dim]"
  echo "   OR: set EVAL_CKPTS (and optional EVAL_LATENT_DIMS) in this script and run with no args"
  exit 1
fi
