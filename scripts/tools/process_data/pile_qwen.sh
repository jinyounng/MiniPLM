BASE_PATH=$1

export PYTHONPATH=${BASE_PATH}
python3 tools/process_data/tokenize_pile.py \
    --base-path $BASE_PATH \
    --model-path checkpoints/qwen/200M \
    --data-dir data/pile/train \
    --save processed_data/pretrain/ \
    --data-name pile \
    --max-length 1025 \
    --log-interval 10000 \
    --data-process-workers 32 \
    --model-type qwen \
    --chunk-num-per-shard 1000000