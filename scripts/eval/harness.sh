#! /bin/bash

BASE_PATH=${1-"/home/MiniPLM"}
MASTER_ADDR=localhost
MASTER_PORT=2030
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# type
TYPE="eval_harness"
# model
CKPT_NAME="200M_miniplm"
CKPT="${BASE_PATH}/results/miniplm/200M"
# data
DATA_NAME="end_tasks"
EVAL_DATA_NAMES="hellaswag,lambada_openai,winogrande,openbookqa,arc_easy,arc_challenge,piqa,social_iqa,storycloze_2018"

# hp
EVAL_BATCH_SIZE=64
# runtime
SAVE_PATH="${BASE_PATH}/results/${TYPE}"
# seed
SEED=10
# wandb
WANDB_NAME="200M_miniplm"


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
# data
OPTS+=" --data-name ${DATA_NAME}"
OPTS+=" --eval-data-names ${EVAL_DATA_NAMES}"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
# runtime
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --wandb-group eval_harness"
OPTS+=" --wandb-name ${WANDB_NAME}"
OPTS+=" --wandb-mode disabled"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"


export NCCL_DEBUG=""
# export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
export OMP_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/eval_main.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
