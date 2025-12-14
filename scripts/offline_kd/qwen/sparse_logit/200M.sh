#! /bin/bash

BASE_PATH=${1-"/data/jykim/MiniPLM"}
MASTER_PORT=${2-2030}
GPUS_PER_NODE=${3-4}
NNODES=1
# HOSTFILE=${5-hostfile_8V100_0_1}

DISTRIBUTED_ARGS="--num_gpus $GPUS_PER_NODE \
                  --num_nodes $NNODES \
                  --master_port $MASTER_PORT"
                  #--hostfile $BASE_PATH/configs/hostfiles/$HOSTFILE

# type
TYPE="offline_kd"
# model
CKPT_NAME="qwen/200M_pretrained"
CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}/"
# cached logits (필수!)
CACHED_LOGITS_DIR="/data/jykim/DB/miniplm_refined_corpus_logits_topk"  # 또는 _sparse
# data
DATA_DIR="/data/jykim/DB/miniplm_refined_corpus"
DATA_NAME="dolly"
WANDB_NAME="200M-offline-kd"
# hp
BATCH_SIZE=1
LR=0.0006
LR_MIN=0.00006
GRAD_ACC=64
# KD hyperparameters
ALPHA=0.5              # KD loss 가중치 (0~1, 클수록 KD에 더 가중치)
KD_TEMPERATURE=1.0     # Temperature scaling
# length
MAX_LENGTH=1024
# runtime
SAVE_PATH="${BASE_PATH}/results/${TYPE}"
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
# OPTS+=" --torch-compile reduce-overhead"
# offline KD specific
OPTS+=" --cached-logits-dir ${CACHED_LOGITS_DIR}"
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
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --adam-beta 0.9"
OPTS+=" --adam-beta2 0.98"
OPTS+=" --adam-eps 1e-6"
OPTS+=" --total-iters 20000"
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

