from datasets import load_dataset

ds = load_dataset(
    "/data/jykim/DB/pile-diff_samp-qwen_1.8B-qwen_104M-r0.5",
    split="train"
)

print(ds)
print(ds[0])