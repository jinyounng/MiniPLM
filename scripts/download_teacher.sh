#!/bin/bash
# Download Teacher Model for Knowledge Distillation
#
# Qwen2.5 시리즈 추천:
# - Qwen2.5-1.5B: 작은 Teacher, 빠른 캐싱 (~3GB)
# - Qwen2.5-7B: 중간 크기 Teacher (~15GB) ⭐ 추천
# - Qwen2.5-14B: 큰 Teacher (~30GB)
# - Qwen2.5-32B: 더 큰 Teacher (~65GB)
# - Qwen2.5-72B: 최대 Teacher (~145GB, H200 1장에 fit)

OUTPUT_DIR="/data/jykim/models"
MODEL_NAME=${1-"Qwen/Qwen2.5-7B"}  # 기본: 7B

mkdir -p ${OUTPUT_DIR}

echo "Downloading ${MODEL_NAME} to ${OUTPUT_DIR}..."

# HuggingFace Hub에서 다운로드
python3 -c "
from huggingface_hub import snapshot_download
import os

model_name = '${MODEL_NAME}'
output_dir = '${OUTPUT_DIR}'

# 모델 이름에서 폴더명 추출 (Qwen/Qwen2.5-7B -> Qwen2.5-7B)
local_dir = os.path.join(output_dir, model_name.split('/')[-1])

print(f'Downloading {model_name} to {local_dir}...')
snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True
)
print(f'✅ Downloaded to {local_dir}')
"

echo ""
echo "Usage after download:"
echo "  bash scripts/offline_kd/cache_logits_topk.sh /data/jykim/MiniPLM ${OUTPUT_DIR}/${MODEL_NAME##*/}"

