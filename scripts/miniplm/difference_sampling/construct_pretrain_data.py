import sys
base_path = sys.argv[1]
sys.path.append(base_path)
import os
import torch
import numpy as np
import random
from transformers import AutoTokenizer
from tqdm import tqdm
import re

from data_utils import DistributedMMapIndexedDataset, ChunkedDatasetBuilder, best_fitting_dtype


def main():

    np.random.seed(42)
    random.seed(42)

    model_path = os.path.join(base_path, "checkpoints/qwen/200M/")
    data_path = os.path.join(base_path, "processed_data/pretrain/pile/qwen-1025")

    ratio = float(sys.argv[2])

    score_path = os.path.join(base_path, f"results/lm_infer/pile/diff-qwen_1.8B-qwen_104M/diff_scores.pt")
    output_path = os.path.join(base_path, f"processed_data/pretrain/pile-diff_samp-qwen_1.8B-qwen_104M-r{ratio}/qwen-1025")

    os.makedirs(output_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    dtype = best_fitting_dtype(len(tokenizer))
    print("dtype: ", dtype)

    scores = torch.load(score_path, map_location="cpu")
    dataset = DistributedMMapIndexedDataset(data_path, "data")

    if len(scores) != len(dataset):
        print("Warning: len(scores) != len(dataset) ({} != {})".format(len(scores), len(dataset)))

    sorted_scores, sorted_indices = torch.sort(scores, descending=True)

    kept_indices = sorted_indices[:int(ratio * len(sorted_indices))]
    kept_scores = sorted_scores[:int(ratio * len(sorted_indices))]

    indices = torch.sort(kept_indices)[0]

    builder = ChunkedDatasetBuilder(base_path, output_path, dtype)

    for i, idx in enumerate(tqdm(indices)):
        data = dataset[idx.item()]
        if i == 0:
            print(idx)
            print(data.astype(int))
            print(tokenizer.decode(data.astype(int)))
        builder.add_np_item(data)
    builder.finalize()
    
    
if __name__ == "__main__":
    main()