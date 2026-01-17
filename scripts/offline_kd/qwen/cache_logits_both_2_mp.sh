#!/bin/bash
# Cache Teacher Logits - Multi-GPU Parallel Version (Queue-based)
#
# 각 GPU worker가 큐에서 shard를 하나씩 가져와서 처리합니다.
# 8개 GPU면 이론상 8배 속도 향상!
#
# 메모리 효율적:
#   - Shard 완료 후 즉시 메모리 해제 (RAM 사용량 안정적)
#   - 여러 shard 결과를 메모리에 누적하지 않음
#   - 동적 로드 밸런싱 (빠른 GPU가 더 많은 shard 처리)
#
# 기존 cache_logits_both.sh 대비:
#   - 모델을 각 GPU에 개별 로드 (device_map='auto' 대신)
#   - 큐 기반 shard 분산 처리 (메모리 효율적)
#   - RAM 4TiB + 8 GPU = 최적 성능

BASE_PATH=${1-"/home/jiwonyoon/data1/projects/MiniPLM"}
TEACHER_MODEL=${2-"/home/jiwonyoon/data1/checkpoints/qwen/7B"}

# GPU 설정 (8대 사용)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

# Triton 캐시 디렉토리 (NFS 회피)
export TRITON_CACHE_DIR="/home/jiwonyoon/data1/.triton_cache"

# Data paths
DATA_DIR="/home/jiwonyoon/data1/data/pile_dataset"
OUTPUT_DIR="/home/jiwonyoon/data1/data/miniplm_refined_corpus_logits_both"

# Sampling parameters
TOPK=100              # Top-K의 K
NUM_SAMPLES=50        # Random Sampling의 N
BATCH_SIZE=52       # GPU당 배치 사이즈 (데이터 분산 처리로 메모리 여유 있음)
MAX_LENGTH=1024

# Processing range
START_SHARD=45
END_SHARD=-1

export PYTHONPATH=${BASE_PATH}

echo "🚀 Starting Multi-GPU Parallel Caching..."
echo "   GPUs: ${NUM_GPUS}"
echo "   Teacher: ${TEACHER_MODEL}"
echo "   Output: ${OUTPUT_DIR}"

python ${BASE_PATH}/scripts/cache_teacher_logits_mp.py \
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
    --num-gpus ${NUM_GPUS} \
    --dtype bf16

echo ""
echo "✅ BOTH (Top-K + Sparse) caching completed!"
echo "   Output: ${OUTPUT_DIR}"
echo "   학습 시 --kd-method로 선택: topk 또는 sparse"

