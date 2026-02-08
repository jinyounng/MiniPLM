# MiniPLM

MiniPLM 기반 학습/평가 환경입니다.

## 설치

```bash
pip install -r requirements.txt
# 또는
bash install.sh
```

## 프로젝트 구조

```
MiniPLM/
├── train.py           # 메인 학습
├── arguments.py       # CLI 인자
├── vanilla_kd/        # Vanilla KD
├── sparse_kd/         # Sparse KD
├── offline_kd/        # Offline KD
├── pretrain/          # 일반 pretrain
├── data_utils/        # 데이터셋
├── scripts/           # 실행 스크립트
│   ├── vanilla_kd/    # Vanilla KD
│   ├── sparse_kd/     # Sparse KD
│   ├── top-k_kd/      # Top-K KD
│   └── eval/          # 평가
└── ...
```

## 실행

**학습 예시**
```bash
bash scripts/vanilla_kd/qwen/200M.sh /PATH/TO/MiniPLM
```

**평가**
```bash
bash scripts/eval/harness.sh /PATH/TO/MiniPLM \
    --model-path /PATH/TO/CKPT \
    --ckpt-name NAME
```

## 데이터

Pile corpus (bin/idx 형식) 사용. 토크나이즈:
```bash
bash scripts/tools/process_data/pile_qwen.sh /PATH/TO/MiniPLM
```

## 참고

- [MiniLLM HuggingFace](https://huggingface.co/MiniLLM)
