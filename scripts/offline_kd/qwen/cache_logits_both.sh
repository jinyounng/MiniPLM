#!/bin/bash
# Cache Teacher Logits - BOTH Methods (Top-K + Random Sampling)
#
# Teacher forward 한 번으로 Top-K와 Random Sampling 둘 다 저장합니다.
# - 장점: 시간 절약 (forward 한 번만), 두 방법 모두 비교 가능
# - 저장: 같은 파일에 topk와 sparse 데이터 모두 포함
#
# 학습 시에는 --kd-method로 선택:
#   --kd-method topk   : Top-K 사용
#   --kd-method sparse : Random Sampling 사용

BASE_PATH=${1-"/data/jykim/MiniPLM"}
TEACHER_MODEL=${2-"/data/jykim/models/Qwen2.5-7B"}  # Teacher 모델 경로 수정 필요

# GPU 설정 (4대 사용)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Data paths
DATA_DIR="/data/jykim/DB/miniplm_refined_corpus"
OUTPUT_DIR="/data/jykim/DB/miniplm_refined_corpus_logits_both"

# Sampling parameters
TOPK=100              # Top-K의 K
NUM_SAMPLES=50       # Random Sampling의 N
BATCH_SIZE=512       
MAX_LENGTH=1024

# Processing range (shard 단위로 resume 가능)
START_SHARD=0
END_SHARD=-1  # -1 means all

export PYTHONPATH=${BASE_PATH}

python ${BASE_PATH}/scripts/cache_teacher_logits.py \
    --teacher-model-path ${TEACHER_MODEL} \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --method both \
    --topk ${TOPK} \
    --num-samples ${NUM_SAMPLES} \
    --batch-size ${BATCH_SIZE} \
    --max-length ${MAX_LENGTH} \
    --start-shard ${START_SHARD} \
    --end-shard ${END_SHARD} \
    --device auto \
    --dtype bf16

echo " BOTH (Top-K + Sparse) caching completed! Output: ${OUTPUT_DIR}"
echo "   학습 시 --kd-method로 선택: topk 또는 sparse"

