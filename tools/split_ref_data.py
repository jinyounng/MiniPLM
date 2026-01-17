import sys
base_path = sys.argv[1]
import os
from tqdm import tqdm

input_path = os.path.join(base_path, "processed_data/pretrain/pile/qwen-1025/")
output_path = os.path.join(base_path, "processed_data/pretrain/pile_ref/qwen-1025/")

os.makedirs(output_path, exist_ok=True)

new_idx = 0

# split from the end
for idx in tqdm(range(240, 246)):
    linkto = os.readlink(os.path.join(input_path, f"data_{idx}.bin"))
    os.symlink(linkto, os.path.join(output_path, f"data_{new_idx}.bin"))

    linkto = os.readlink(os.path.join(input_path, f"data_{idx}.idx"))
    os.symlink(linkto, os.path.join(output_path, f"data_{new_idx}.idx"))

    linkto = os.readlink(os.path.join(input_path, f"data_{idx}.index"))
    os.symlink(linkto, os.path.join(output_path, f"data_{new_idx}.index"))
    
    new_idx += 1
