# MiniPLM: Knowledge Distillation 연구 환경

이 프로젝트는 **MiniPLM** 환경을 기반으로 한 **Knowledge Distillation (KD)** 연구를 위한 코드베이스입니다.  
Teacher 모델의 지식을 Student 모델로 전달하는 다양한 방법을 비교·실험할 수 있으며, **Logit Sparse Sampling**, **Top-K KD**, **Vanilla KD** 등을 지원합니다.  
또한 **AutoEncoder (AE)** 를 이용한 Teacher hidden state 압축 및 이를 활용한 KD 파이프라인을 포함합니다.

---

## 목차

1. [개요](#1-개요)
2. [설치](#2-설치)
3. [프로젝트 구조](#3-프로젝트-구조)
4. [지원 KD 방법](#4-지원-kd-방법)
5. [AE 압축](#5-ae-압축)
6. [Logit 분석](#6-logit-분석)
7. [실행 방법](#7-실행-방법)
8. [평가](#8-평가)

---

## 1. 개요

### 핵심 기능

| 영역 | 설명 |
|------|------|
| **Vanilla KD** | Teacher 전체 logits를 사용한 전통적 KL divergence 기반 KD |
| **Top-K KD** | Teacher 상위 K개 토큰의 확률만 사용 (sparse, 메모리 효율적) |
| **Logit Sparse Sampling** | Teacher 확률 분포에서 N번 샘플링하여 unbiased estimator로 KD |
| **Offline KD** | 미리 caching한 teacher logits로 학습 (Teacher forward 불필요) |
| **AE 압축** | Teacher hidden state를 latent로 압축, KD 파이프라인에 활용 |

### 지원 모델 및 데이터

- **Teacher**: Qwen (1.5B, 7B 등)
- **Student**: Qwen 200M, 500M, 1.2B 및 Mamba, LLaMA3.1
- **데이터**: Pile corpus (bin/idx 형식)

---

## 2. 설치

```bash
pip install -r requirements.txt
# 또는
bash install.sh
```

### 주요 의존성

- `torch`, `transformers`, `deepspeed`, `accelerate`
- `wandb` (실험 로깅)
- `lm-evaluation-harness` (평가)

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
pip install -e lm-evaluation-harness
```

---

## 3. 프로젝트 구조

```
MiniPLM/
├── train.py                 # 메인 학습 진입점 (type별 trainer 분기)
├── arguments.py             # CLI 인자 및 KD 관련 옵션
├── vanilla_kd/              # Vanilla KD trainer
├── sparse_kd/               # Sparse (Online) KD trainer (top-k / random sampling)
├── offline_kd/              # Offline KD trainer (cached logits)
├── pretrain/                # 일반 pretrain
├── data_utils/              # 데이터셋 (sparse_kd_datasets_hdf5, indexed_dataset 등)
├── train_eval_utils/        # sparse_kd_loss 등
├── logit_analysis/          # Teacher logit K 분석
├── scripts/
│   ├── vanilla_kd/          # Vanilla KD 실행 스크립트
│   ├── sparse_kd/           # Offline KD (sparse sampling) 스크립트
│   ├── top-k_kd/            # Offline KD (top-k) 스크립트
│   ├── offline_kd/          # Logits caching (topk, sparse)
│   ├── AE/                  # AE 학습 및 KD 활용
│   │   ├── train/           # AE, RVQ, top-k logit 등
│   │   └── kd/              # AE latent 기반 KD
│   └── eval/                # 평가 스크립트
└── recon_results/          # AE/RVQ 평가 결과 (JSON)
```

---

## 4. 지원 KD 방법

### 4.1 Vanilla KD

Teacher 전체 logits와 Student logits 간 **KL divergence**로 loss 계산.

- **특징**: Teacher 온라인 forward 필요, 전체 vocab 사용
- **옵션**: `--kd-ratio` (LM loss vs KD loss 비율)

```bash
bash scripts/vanilla_kd/qwen/200M.sh /PATH/TO/MiniPLM
```

### 4.2 Top-K KD (Sparse)

Teacher 상위 K개 토큰의 확률만 사용. Top-K로 정규화 후 cross-entropy loss.

- **장점**: 메모리/계산 효율, 중요한 토큰에 집중
- **단점**: Tail 확률 무시 (biased)

**Online** (sparse_kd, `--kd-method topk`):

```bash
# train.py --type sparse_kd --kd-method topk --topk 50
```

**Offline** (cached logits 사용):

1. Teacher logits 캐싱: `scripts/offline_kd/qwen/cache_logits_topk.sh`
2. Offline KD 학습: `scripts/top-k_kd/qwen/200M.sh`

### 4.3 Logit Sparse Sampling (Random Sampling)

Teacher 확률 분포에서 N번 샘플링하여 KD loss 추정. **Unbiased estimator**.

- **장점**: E[loss] = true KD loss
- **단점**: Variance 존재, 샘플 수 N에 따라 정확도 변화

**Online** (sparse_kd, `--kd-method sparse`):

```bash
# train.py --type sparse_kd --kd-method sparse --num-samples 50
```

**Offline**:

1. 캐싱: `scripts/offline_kd/qwen/cache_logits_sparse.sh`
2. 학습: `scripts/sparse_kd/qwen/200M.sh`

### 4.4 Offline KD 정리

| 방법 | Caching 스크립트 | 학습 스크립트 | `--kd-method` |
|------|------------------|---------------|--------------|
| Top-K | `cache_logits_topk.sh` | `top-k_kd/qwen/200M.sh` | `topk` |
| Sparse (Random) | `cache_logits_sparse.sh` | `sparse_kd/qwen/200M.sh` | `sparse` |

Cached logits는 HDF5 (`data_*.h5` 또는 `shard_*.h5`) 형식으로 저장.

---

## 5. AE 압축

Teacher hidden state를 latent로 압축하여 저장·복원하고, KD에 활용합니다.

### 5.1 Conditional AE (Y-conditioned)

다음 토큰 예측(y)을 condition으로 사용하는 AE.

- **입력**: hidden + teacher embedding(y)
- **출력**: recon (hidden에 대한 reconstuction)
- **Loss**: MSE, cosine, logit KL, logit MSE 등

```bash
bash scripts/AE/train/train_ae.sh
```

### 5.2 Top-K Logit Loss AE

Logit loss 계산 시 teacher 상위 K개 logit만 사용.

```bash
bash scripts/AE/train/train_ae_top-k.sh
# 환경변수: TOPK_LOGIT=5000
```

### 5.3 RVQ (Residual Vector Quantization)

Multi-stage VQ로 latent 압축.

```bash
bash scripts/AE/train/train_RVQ.sh
# 파라미터: num_stages, num_codes, compressed_dim, gamma 등
```

### 5.4 AE Latent 기반 KD

저장된 latent(z)와 y_token만 사용해 Teacher forward 없이 KD.

1. Latent 저장: `scripts/AE/kd/save_ae_latent_5pct.sh`
2. KD 학습: `scripts/AE/kd/1stage_from_latent.sh`

### 5.5 AE 관련 결과

`recon_results/` 에 각 AE/RVQ 설정별 평가 결과가 JSON으로 저장됩니다.

- `logit_only/`: logit-only loss AE
- `topk_logit/`: top-k logit loss AE
- `zonly/`: z-only AE
- `RVQ/`, `condition_RVQ/`: RVQ 설정별 결과

---

## 6. Logit 분석

Teacher의 각 토큰 스텝에서 softmax 확률을 분석합니다.

- **K_0.99 / K_0.999**: 누적확률 99% / 99.9% 달성에 필요한 최소 토큰 수
- **Coverage Rate**: 고정 K(1000, 2000, 5000, 10000)에서 0.99 coverage 달성 비율

```bash
# Multi-GPU
torchrun --nproc_per_node=8 logit_analysis/logit_k_analysis.py \
    --model-path /path/to/teacher \
    --data-dir /path/to/bin/data \
    --output-dir ./analysis_results \
    --max-samples 100000 \
    --distributed

# Single GPU
python logit_analysis/logit_k_analysis.py \
    --model-path /path/to/teacher \
    --data-dir /path/to/bin/data \
    --output-dir ./analysis_results
```

출력: `analysis_report.txt`, `aggregate_stats.json`, `coverage_rate_0.99.json`, `k_distribution.png` 등.

---

## 7. 실행 방법

### 7.1 공통 설정

- `BASE_PATH`: 프로젝트 루트
- `DATA_DIR`, `DATA_NAME`: bin/idx 데이터 경로
- `CKPT`, `TEACHER_MODEL_PATH`: Student/Teacher checkpoint

### 7.2 데이터 준비

Pile tokenization:

```bash
bash scripts/tools/process_data/pile_qwen.sh /PATH/TO/MiniPLM
```

### 7.3 학습 타입별 예시

| 타입 | 설명 | 예시 |
|------|------|------|
| `pretrain` | 일반 pretrain | `scripts/pretrain/qwen/200M.sh` |
| `vanilla_kd` | Vanilla KD | `scripts/vanilla_kd/qwen/200M.sh` |
| `sparse_kd` | Online sparse KD | `--type sparse_kd --kd-method topk/sparse` |
| `offline_kd` | Cached logits KD | `scripts/sparse_kd/qwen/200M.sh` |

### 7.4 주요 인자

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--kd-ratio` | KD loss 비율 | 0.5 |
| `--alpha` | Offline KD에서 KD loss 가중치 | 0.5 |
| `--topk` | Top-K의 K | 50 |
| `--num-samples` | Sparse sampling N | 50 |
| `--cached-logits-dir` | Offline KD용 HDF5 경로 | - |
| `--kd-method` | `topk` / `sparse` | - |

---

## 8. 평가

### LM-Evaluation-Harness

```bash
bash scripts/eval/harness.sh /PATH/TO/MiniPLM \
    --model-path /PATH/TO/TRAINED_CKPT \
    --ckpt-name NAME_OF_CKPT
```

### Language Modeling (PPL)

```bash
bash scripts/eval/lm.sh /PATH/TO/MiniPLM \
    --model-path /PATH/TO/TRAINED_CKPT \
    --ckpt-name NAME_OF_CKPT
```

---

## 참고

- MiniPLM 원본: [paper](https://arxiv.org/abs/2410.17215) | [huggingface](https://huggingface.co/MiniLLM)
- 본 프로젝트는 MiniPLM 구조를 기반으로 **Logit Sparse Sampling**, **Top-K KD**, **Vanilla KD** 비교 및 **AE 압축** 실험을 위한 확장 코드를 포함합니다.
