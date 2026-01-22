#! /bin/bash
# Offline KD with Sparse Sampling (Random Sampling) - 500M Model
#
# Cached teacher logits를 사용한 Knowledge Distillation
# method: sparse (random sampling) - unbiased estimator


BASE_PATH=${1-"/home/jiwonyoon/data1/projects/MiniPLM"}
MASTER_PORT=${2-2060}
GPUS_PER_NODE=${3-8}
NNODES=1

DISTRIBUTED_ARGS="--num_gpus $GPUS_PER_NODE \
                  --num_nodes $NNODES \
                  --master_port $MASTER_PORT"

# type
TYPE="offline_kd"
# model
CKPT_NAME="qwen/500M"
CKPT="/home/jiwonyoon/data1/checkpoints/${CKPT_NAME}"
# cached logits (Sparse Sampling 사용, HDF5 포맷)
CACHED_LOGITS_DIR="/home/jiwonyoon/data1/data/logits_hdf5"
KD_METHOD="sparse"  # 'topk' or 'sparse' (method='both'일 때 선택)
# data
DATA_DIR="/home/jiwonyoon/data1/data/pile_dataset"
DATA_NAME="miniplm_refined_corpus"
WANDB_NAME="500M-offline-kd-sparse"
# hp (vanilla_kd/500M.sh와 동일)
BATCH_SIZE=32
LR=0.0003
LR_MIN=0.00003
GRAD_ACC=2
# KD hyperparameters
ALPHA=0.5              # KD loss 가중치 (0~1, 클수록 KD에 더 가중치)
KD_TEMPERATURE=1.0     # Temperature scaling (sparse KD에서는 1.0 권장)
# length
MAX_LENGTH=1024
# runtime
SAVE_PATH="${BASE_PATH}/results/${TYPE}/sparse_kd"
# seed
SEED=10


OPTS=""
# type
OPTS+=" --type ${TYPE}"
# model
OPTS+=" --model-type qwen"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --n-nodes ${NNODES}"
# OPTS+=" --gradient-checkpointing"
OPTS+=" --from-scratch"
# offline KD specific
OPTS+=" --cached-logits-dir ${CACHED_LOGITS_DIR}"
OPTS+=" --kd-method ${KD_METHOD}"
OPTS+=" --alpha ${ALPHA}"
OPTS+=" --kd-temperature ${KD_TEMPERATURE}"
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
OPTS+=" --total-iters 100000"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
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
OPTS+=" --wandb-group offline_kd"
OPTS+=" --wandb-name ${WANDB_NAME}"


export NCCL_DEBUG=""
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
export OMP_NUM_THREADS=16
CMD="deepspeed ${DISTRIBUTED_ARGS} ${BASE_PATH}/train.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
