#!/bin/bash
# Sparse Online KD Training - Random Sampling Method (500M)
#
# Teacher 확률분포에서 N번 랜덤 샘플링하여 KD loss 추정
# 장점: Unbiased estimator (기대값이 정확)
# 단점: Top-K보다 약간 느림, variance 있음

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo
export MASTER_ADDR=192.168.129.92
export MASTER_PORT=29700

BASE_PATH=${1-"/home/jiwonyoon/data1/projects/MiniPLM"}
MASTER_PORT=${2-2050}
GPUS_PER_NODE=${3-8}
NNODES=1

DISTRIBUTED_ARGS="--num_gpus $GPUS_PER_NODE \
                  --num_nodes $NNODES \
                  --master_port $MASTER_PORT"

# type
TYPE="sparse_kd"
# model
CKPT_NAME="qwen/500M"
CKPT="/home/jiwonyoon/data1/checkpoints/${CKPT_NAME}"
TEACHER_CKPT_NAME="7B"
TEACHER_MODEL_PATH="/home/jiwonyoon/data1/checkpoints/qwen/7B"
# data
DATA_DIR="/home/jiwonyoon/data1/data/pile_dataset"
DATA_NAME="miniplm_refined_corpus"
WANDB_NAME="500M-sparse-kd-random50"
# hp
BATCH_SIZE=8
LR=0.0004
LR_MIN=0.00004
GRAD_ACC=8
# length
MAX_LENGTH=1024
# runtime
SAVE_PATH="${BASE_PATH}/results/${TYPE}"
# seed
SEED=10

# ========== Random Sampling Settings ==========
KD_METHOD="sparse"
NUM_SAMPLES=50        # 샘플링 횟수 N (50, 100 등)
KD_RATIO=0.5          # KD loss 가중치 (0~1)
KD_TEMPERATURE=1.0    # Temperature for softmax
# ==============================================


OPTS=""
# type
OPTS+=" --type ${TYPE}"
# model
OPTS+=" --model-type qwen"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-model-type qwen"
OPTS+=" --teacher-model-path ${TEACHER_MODEL_PATH}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --n-nodes ${NNODES}"
OPTS+=" --from-scratch"
# data
OPTS+=" --data-name ${DATA_NAME}"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 8"
OPTS+=" --bin-data"
OPTS+=" --no-shuffle"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --lr-min ${LR_MIN}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 2000"
OPTS+=" --scheduler-name cosine"
OPTS+=" --weight-decay 0.1"
OPTS+=" --clip-grad 1.0"
OPTS+=" --adam-beta 0.9"
OPTS+=" --adam-beta2 0.98"
OPTS+=" --adam-eps 1e-6"
OPTS+=" --total-iters 18000"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
# sparse kd - Random Sampling
OPTS+=" --kd-method ${KD_METHOD}"
OPTS+=" --num-samples ${NUM_SAMPLES}"
OPTS+=" --kd-ratio ${KD_RATIO}"
OPTS+=" --kd-temperature ${KD_TEMPERATURE}"
# runtime
OPTS+=" --do-train"
OPTS+=" --save-interval 1000"
OPTS+=" --log-interval 10"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --no-eval-when-start"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# wandb
OPTS+=" --wandb-group sparse_kd"
OPTS+=" --wandb-name ${WANDB_NAME}"


export NCCL_DEBUG=""
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
export OMP_NUM_THREADS=16
CMD="deepspeed ${DISTRIBUTED_ARGS} ${BASE_PATH}/train.py ${OPTS} $@"

echo "=========================================="
echo "Sparse KD Training - Random Sampling (500M)"
echo "  Num Samples: ${NUM_SAMPLES}"
echo "  KD Ratio: ${KD_RATIO}"
echo "  Temperature: ${KD_TEMPERATURE}"
echo "=========================================="
echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}

