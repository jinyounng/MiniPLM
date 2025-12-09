from transformers import AutoTokenizer, AutoModelForCausalLM

# 다운로드 위치 지정
cache_dir = "checkpoints/qwen"

# 모델과 토크나이저 다운로드
tokenizer = AutoTokenizer.from_pretrained(
    "MiniLLM/MiniPLM-Qwen-200M",
    cache_dir=cache_dir
)
model = AutoModelForCausalLM.from_pretrained(
    "MiniLLM/MiniPLM-Qwen-200M",
    cache_dir=cache_dir
)

# 나머지 코드는 동일
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))