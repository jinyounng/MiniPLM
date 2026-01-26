#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-"$(cd "${SCRIPT_DIR}/../../.." && pwd)"}"
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-2030}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Edit this list to choose which checkpoints to evaluate.
# Only the paths listed here will be run.
MODELS=(
  "/home/jiwonyoon/data1/projects/MiniPLM/results/offline_kd/sparse_kd/qwen_200M/t100K-w2K-bs32-lr0.0006cosine6e-05-G2-N8-NN1-scr/offline-topk-a0.5/100000"
  "/home/jiwonyoon/data1/projects/MiniPLM/results/offline_kd/sparse_kd/qwen_500M/t100K-w2K-bs32-lr0.0003cosine3e-05-G2-N8-NN1-scr/offline-topk-a0.5/100000"
  "/home/jiwonyoon/data1/projects/MiniPLM/results/offline_kd/topk/qwen_200M/t100K-w2K-bs32-lr0.0006cosine6e-05-G2-N8-NN1-scr/offline-topk-a0.5/100000"
  "/home/jiwonyoon/data1/projects/MiniPLM/results/pretrain/qwen_1.2B/t100K-w2K-bs32-lr0.00025cosine2.5e-05-G2-N8-NN1-scr/100000"
  "/home/jiwonyoon/data1/projects/MiniPLM/results/pretrain/qwen_200M/t100K-w2K-bs64-lr0.0006cosine6e-05-G1-N8-NN1-scr/100000"
  "/home/jiwonyoon/data1/projects/MiniPLM/results/pretrain/qwen_500M/t100K-w2K-bs32-lr0.0006cosine6e-05-G2-N8-NN1-scr/100000"
  "/home/jiwonyoon/data1/projects/MiniPLM/results/vanilla_kd/miniplm_refined_corpus/qwen_200M/t100K-w2K-bs16-lr0.0006cosine6e-05-G4-N8-NN1-scr/7B-kd0.5/100000"
)

if [ ${#MODELS[@]} -eq 0 ]; then
  echo "No models configured. Edit MODELS in this script."
  exit 1
fi

DISTRIBUTED_ARGS=(
  --nproc_per_node "$GPUS_PER_NODE"
  --nnodes "$NNODES"
  --node_rank "$NODE_RANK"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
)

TYPE="eval_harness"
DATA_NAME="end_tasks"
EVAL_DATA_NAMES="lambada_openai"
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}
SEED=${SEED:-10}

export NCCL_DEBUG=""
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="${BASE_PATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export TOKENIZERS_PARALLELISM=false

for CKPT in "${MODELS[@]}"; do
  if [ ! -d "$CKPT" ]; then
    echo "Skipping: ${CKPT} (not a directory)"
    continue
  fi

  CKPT_NAME="$(basename "$CKPT")"
  CKPT_TAG="${CKPT#${BASE_PATH}/results/}"
  CKPT_TAG="${CKPT_TAG//\//_}"
  if [ -z "$CKPT_TAG" ] || [ "$CKPT_TAG" = "$CKPT" ]; then
    CKPT_TAG="$CKPT_NAME"
  fi

  SAVE_PATH="${BASE_PATH}/results/${TYPE}/${CKPT_TAG}"

  OPTS=(
    --type "$TYPE"
    --model-type qwen
    --base-path "$BASE_PATH"
    --model-path "$CKPT"
    --ckpt-name "$CKPT_NAME"
    --n-gpu "$GPUS_PER_NODE"
    --n-nodes "$NNODES"
    --data-name "$DATA_NAME"
    --eval-data-names "$EVAL_DATA_NAMES"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --save "$SAVE_PATH"
    --wandb-group eval_harness
    --wandb-name "$CKPT_TAG"
    --wandb-mode disabled
    --seed "$SEED"
    --deepspeed
    --deepspeed_config "${BASE_PATH}/configs/deepspeed/ds_config.json"
  )

  CMD=(torchrun "${DISTRIBUTED_ARGS[@]}" "${BASE_PATH}/eval_main.py" "${OPTS[@]}")

  echo "${CMD[*]}"
  echo "PYTHONPATH=${PYTHONPATH}"
  mkdir -p "${SAVE_PATH}"
  "${CMD[@]}"
done
