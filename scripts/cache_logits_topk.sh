#!/bin/bash
# Cache Teacher Logits - Top-K Method
# 
# 상위 K개 토큰과 확률값을 저장합니다.
# - 장점: 확률값 정확, 구현 간단
# - 단점: Biased (tail 확률 무시)

BASE_PATH=${1-"/data/jykim/MiniPLM"}
TEACHER_MODEL=${2-"/data/jykim/models/Qwen2.5-1.5B"}  # Teacher 모델 경로 수정 필요

# Data paths
DATA_DIR="/data/jykim/DB/miniplm_refined_corpus"
OUTPUT_DIR="/data/jykim/DB/miniplm_refined_corpus_logits_topk"

# Sampling parameters
TOPK=50
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
    --method topk \
    --topk ${TOPK} \
    --batch-size ${BATCH_SIZE} \
    --max-length ${MAX_LENGTH} \
    --start-shard ${START_SHARD} \
    --end-shard ${END_SHARD} \
    --dtype bf16

echo "✅ Top-K caching completed! Output: ${OUTPUT_DIR}"

