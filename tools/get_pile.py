import datasets
import os
from tqdm import tqdm
import json


data = datasets.load_dataset("monology/pile-uncopyrighted")
print(data)

output_dir = "data/pile/"
os.makedirs(output_dir, exist_ok=True)

max_num_per_shard = 1000000
ofid = 0
did = 0

for split in ["train"]:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    f = open(os.path.join(output_dir, split, f"{ofid}.jsonl"), "w")
    for d in tqdm(data[split]):
        f.write(json.dumps(d) + "\n")
        did += 1
        if did >= max_num_per_shard:
            f.close()
            ofid += 1
            did = 0
            f = open(os.path.join(output_dir, split, f"{ofid}.jsonl"), "w")
    
    f.close()