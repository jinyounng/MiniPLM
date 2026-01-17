# download_all.py
import os

# Triton 캐시 경로 설정 (맨 위에!)
os.environ['TRITON_CACHE_DIR'] = '/tmp/triton_cache'
os.makedirs('/tmp/triton_cache', exist_ok=True)

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm

# 저장 경로 설정
base_dir = "/home/jiwonyoon/data1"
data_dir = f"{base_dir}/data/pile_dataset"
checkpoint_dir = f"{base_dir}/checkpoints/qwen"
os.makedirs(data_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

print("=" * 50)
print("1. Downloading refined corpus (106 files)...")
print("=" * 50)

repo_id = "MiniLLM/pile-diff_samp-qwen_1.8B-qwen_104M-r0.5"

# 모든 .bin과 .idx 파일 다운로드
for i in tqdm(range(53), desc="Downloading data files"):
    for ext in ['bin', 'idx']:
        filename = f"data_{i}.{ext}"
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=data_dir,
                local_dir_use_symlinks=False
            )
        except Exception as e:
            print(f"✗ Failed: {filename} - {e}")

print(f"✓ Dataset saved to {data_dir}")

print("\n" + "=" * 50)
print("2. Downloading configs and tokenizers...")
print("=" * 50)

# 200M, 500M, 1.2B config & tokenizer 다운
for size in ["200M", "500M", "1.2B"]:
    save_path = f"{checkpoint_dir}/{size}"
    os.makedirs(save_path, exist_ok=True)
    
    model_name = f"MiniLLM/MiniPLM-Qwen-{size}"
    
    print(f"\nDownloading {size}...")
    
    # Config 다운로드
    config = AutoConfig.from_pretrained(model_name)
    config.save_pretrained(save_path)
    
    # Tokenizer 다운로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_path)
    
    # Weight 파일 있으면 삭제 (random init 위해)
    for file in os.listdir(save_path):
        if file.endswith(('.safetensors', '.bin')):
            os.remove(os.path.join(save_path, file))
            print(f"  Removed weight file: {file}")
    
    print(f"✓ Config & tokenizer saved to {save_path}")

print("\n" + "=" * 50)
print("Download complete!")
print("=" * 50)
print(f"Dataset: {data_dir}")
print(f"Checkpoints: {checkpoint_dir}")
print("\nReady for training from scratch!")