import sys
base_path = sys.argv[1]
sys.path.append(base_path)
import os
import torch
import numpy as np
from tqdm import tqdm
import re
from data_utils import DistributedMMapIndexedDataset
from transformers import AutoTokenizer
import random
import matplotlib.pyplot as plt
from utils import print_and_save_rank
import json


def save(tokenizer, dataset, indices, scores, large_scores, small_scores, output_path):
    with open(os.path.join(output_path), "w") as f:
        for k, idx in enumerate(tqdm(indices)):
            s = tokenizer.decode(dataset[idx], skip_special_tokens=True)
            score = scores[idx].item()
            large_score = large_scores[idx].item()
            small_score = small_scores[idx].item()
            f.write(f"############## {k}, {idx}, diff: {score}, large_score: {large_score}, small_score: {small_score} #############\n")
            f.write(s + "\n\n\n")


def compute_diff_scores(large_scores, small_scores, output_path):
    diff_scores = small_scores - large_scores
    print_and_save_rank("diff_scores size: {}".format(len(diff_scores)), 
                        os.path.join(output_path, "log.txt"))
    
    max_diff_scores = diff_scores.max()
    min_diff_scores = diff_scores.min()
    print_and_save_rank("max_diff_scores: {}, min_diff_scores: {}".format(
        max_diff_scores, min_diff_scores), os.path.join(output_path, "log.txt"))

    return diff_scores


def load_scores(score_path, name, output_path, use_cache=False):
    cache_path = os.path.join(score_path, f"merged_scores.pt")
    if use_cache and os.path.exists(cache_path):
        print_and_save_rank(f"{name} scores load from {cache_path}", os.path.join(output_path, "log.txt"))
        scores = torch.load(cache_path, map_location="cpu")
    else:    
        p = r"scores_(\d+).pt"

        scores = []
        all_large_file_ids = []
        print(score_path)
        for _, _, files in os.walk(score_path):
            for file in files:
                m = re.match(p, file)
                if m is not None:
                    fid = int(m.group(1))
                    all_large_file_ids.append(fid)
        all_large_file_ids = sorted(all_large_file_ids)
        print(all_large_file_ids)
        for fid in tqdm(all_large_file_ids, desc=f"Loading {name} scores"):
            scores.append(torch.load(os.path.join(score_path, f"scores_{fid}.pt"), map_location="cpu")) 
        scores = torch.cat(scores, dim=0)
        print_and_save_rank("{} score original length: {}".format(name, len(scores)), os.path.join(output_path, "log.txt"))

        torch.save(scores, cache_path)        

    mean_scores = scores.mean()
    max_scores = scores.max()
    min_scores = scores.min()
    print_and_save_rank("{} scores: mean: {}, max: {}, min: {}".format(
        name, mean_scores, max_scores, min_scores), os.path.join(output_path, "log.txt"))

    return scores


def stat(diff_scores, large_scores, small_scores, model_path, data_path, output_path):
    sorted_scores, sorted_indices = torch.sort(diff_scores, descending=True)

    fig, ax1 = plt.subplots()
    ax1.hist(diff_scores.numpy(), bins=10000, density=True, histtype='step')
    ax2 = ax1.twinx()
    ax2.hist(diff_scores.numpy(), bins=10000, cumulative=True, histtype='step', density=True, color='tab:orange')
    plt.savefig(os.path.join(output_path, "dist.png"))

    dataset = DistributedMMapIndexedDataset(data_path, "data")
    
    assert len(dataset) == len(diff_scores), f"{len(dataset)} != {len(diff_scores)}"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    top_indices = sorted_indices[:1000].tolist()
    bottom_indices = sorted_indices[-1000:].tolist()

    save(tokenizer, dataset, top_indices, diff_scores, large_scores, small_scores, 
         os.path.join(output_path, "top.txt"))
    save(tokenizer, dataset, bottom_indices, diff_scores, large_scores, small_scores, 
         os.path.join(output_path, "bottom.txt"))

    top_0001 = sorted_indices[:int(len(sorted_indices) * 0.001)].tolist()
    random.shuffle(top_0001)
    top_0001 = top_0001[:1000]
    save(tokenizer, dataset, top_0001, diff_scores, large_scores, small_scores, 
         os.path.join(output_path, "top_0001.txt"))
    
    top_001 = sorted_indices[:int(len(sorted_indices) * 0.01)].tolist()
    random.shuffle(top_001)
    top_001 = top_001[:1000]
    save(tokenizer, dataset, top_001, diff_scores, large_scores, small_scores, 
         os.path.join(output_path, "top_001.txt"))

    top_01 = sorted_indices[:int(len(sorted_indices) * 0.1)].tolist()
    random.shuffle(top_01)
    top_01 = top_01[:1000]
    save(tokenizer, dataset, top_01, diff_scores, large_scores, small_scores, 
         os.path.join(output_path, "top_01.txt"))

    bottom_0001 = sorted_indices[-int(len(sorted_indices) * 0.001):].tolist()
    random.shuffle(bottom_0001)
    bottom_0001 = bottom_0001[:1000]
    save(tokenizer, dataset, bottom_0001, diff_scores, large_scores, small_scores, 
         os.path.join(output_path, "bottom_0001.txt"))
    
    bottom_001 = sorted_indices[-int(len(sorted_indices) * 0.01):].tolist()
    random.shuffle(bottom_001)
    bottom_001 = bottom_001[:1000]
    save(tokenizer, dataset, bottom_001, diff_scores, large_scores, small_scores, 
         os.path.join(output_path, "bottom_001.txt"))

    bottom_01 = sorted_indices[-int(len(sorted_indices) * 0.1):].tolist()
    random.shuffle(bottom_01)
    bottom_01 = bottom_01[:1000]
    save(tokenizer, dataset, bottom_01, diff_scores, large_scores, small_scores, 
         os.path.join(output_path, "bottom_01.txt"))
    


def main():
    random.seed(42)
    torch.random.manual_seed(42)
    
    model_path = os.path.join(base_path, "checkpoints/qwen/200M/")
    data_path = os.path.join(base_path, "processed_data/pile/qwen-1025")
    
    large_score_path = os.path.join(base_path, "results/lm_infer/qwen_1.8B/")
    small_score_path = os.path.join(base_path, "results/lm_infer/pile/qwen_104M/")
    
    output_path = os.path.join(base_path, "results/lm_infer/pile/diff-qwen_1.8B-qwen_104M/")

    os.makedirs(output_path, exist_ok=True)

    #### load & save & compute ####
    large_scores = load_scores(large_score_path, "large", output_path, use_cache=True)
    small_scores = load_scores(small_score_path, "small", output_path, use_cache=True)

    diff_scores = compute_diff_scores(large_scores, small_scores, output_path)
    torch.save(diff_scores, os.path.join(output_path, "diff_scores.pt"))

    #### stat ####
    stat(diff_scores, large_scores, small_scores, model_path, data_path, output_path)


if __name__ == "__main__":
    main()