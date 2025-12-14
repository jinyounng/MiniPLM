#!/bin/bash
# Cache Teacher Logits - Random Sampling (Sparse) Method
#
# Teacher 확률 분포에서 N번 샘플링하여 unique 토큰과 발생 횟수를 저장합니다.
# - 장점: Unbiased estimator (기대값 정확)
# - 단점: variance 있음, 샘플 수에 따라 정확도 변화

BASE_PATH=${1-"/data/jykim/MiniPLM"}
TEACHER_MODEL=${2-"/data/jykim/models/Qwen2.5-1.5B"}  # Teacher 모델 경로 수정 필요

# Data paths
DATA_DIR="/data/jykim/DB/miniplm_refined_corpus"
OUTPUT_DIR="/data/jykim/DB/miniplm_refined_corpus_logits_sparse"

# Sampling parameters
NUM_SAMPLES=50  # 샘플 수 (N)
BATCH_SIZE=8
MAX_LENGTH=1024

# Processing range (shard 단위로 resume 가능)
START_SHARD=0
END_SHARD=-1  # -1 means all

export PYTHONPATH=${BASE_PATH}
export CUDA_VISIBLE_DEVICES=0

python ${BASE_PATH}/scripts/cache_teacher_logits.py \
    --teacher-model-path ${TEACHER_MODEL} \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --method random \
    --num-samples ${NUM_SAMPLES} \
    --batch-size ${BATCH_SIZE} \
    --max-length ${MAX_LENGTH} \
    --start-shard ${START_SHARD} \
    --end-shard ${END_SHARD} \
    --dtype bf16

echo "✅ Sparse (Random Sampling) caching completed! Output: ${OUTPUT_DIR}"

