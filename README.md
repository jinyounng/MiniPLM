# MiniPLM: Knowledge Distillation of Pre-Training Language Models
[paper](https://arxiv.org/abs/2410.17215) | [huggingface](https://huggingface.co/MiniLLM)

<img src="figures/method.png"></img>

#### See also:
+ [MiniLLM](https://github.com/microsoft/LMOps/tree/main/minillm) (Knowledge Distillation of LLMs during SFT)
+ [DPKD](https://github.com/microsoft/LMOps/tree/main/dpkd) (An improved version of MiniLLM with better performance and more simple implementation) 

## 1 Setup
```bash
pip3 install -r requirements.txt
git clone https://github.com/EleutherAI/lm-evaluation-harness
pip3 install -e lm-evaluation-harness
```
or
```bash
bash install.sh
```

## 2 Pre-Training Corpus $\mathcal{D}$
We use [the Pile](https://huggingface.co/datasets/monology/pile-uncopyrighted) as our pre-training corpus. Refer to `tools/get_pile.py` to get the data ready. Run the following command for tokenization:
```bash
bash scripts/tools/process_data/pile_qwen.sh /PATH/TO/MiniPLM
```
The processed data is stored in `processed_data/pretrain/pile/qwen-1025`, containing several shards (a pair of `.bin` and `.idx` files). Each shard contains about 1B tokens. We provide the processed version (100B tokens) for reproducibility.


## 3 Models
### 3.1 Teacher Model
We use [Qwen1.5-1.8B](https://huggingface.co/Qwen/Qwen1.5-1.8B) as the teacher LM. You can download this model can put it in `checkpoints/qwen/1.8B`.
### 3.2 Reference Model
The [reference model](https://huggingface.co/MiniLLM/MiniPLM-Qwen-104M-ref) is a 104M Qwen LM trained on 5B tokens randomly split from the Pile, which should be put in `checkpoints/qwen/104M_ref`.
### 3.3 Pre-Trained Models
The [MiniPLM models](https://huggingface.co/collections/MiniLLM/miniplm-6712c0fdf09ef7e8da7d39bd) and baseline models can be found in the [HuggingFace Hub](https://huggingface.co/MiniLLM).


## 4 Training
### 4.1 MiniPLM
#### Difference Sampling
First, run inference of the teacher LM and the reference LM on the Pile data:
```bash
bash scripts/miniplm/difference_sampling/1.8B.sh /PATH/TO/MiniPLM
bash scripts/miniplm/difference_sampling/104M.sh /PATH/TO/MiniPLM
```
Then, compute the difference scores $r(x,p,p_{\text{ref}})=\frac{\log p(x)}{\log p_{\text{ref}}(x)}$:
```bash
python3 scripts/miniplm/difference_sampling/compute_difference_scores.py /PATH/TO/MiniPLM
```
Finally, construct the refined pre-training corpus with the difference scores:
```bash
python3 scripts/miniplm/difference_sampling/construct_pretrain_data.py /PATH/TO/MiniPLM 0.5 # selection ratio
```
This process constructs a 50B-token corpus from a 100B-token corpus. We open-source the [refined data](https://huggingface.co/datasets/MiniLLM/pile-diff_samp-qwen_1.8B-qwen_104M-r0.5) (50B tokens) for reproducibility.

#### Pre-Training
Before pre-training, you need to put the `config.json` and the tokenizer-related files in `checkpoints/qwen/200M`, `checkpoints/qwen/500M`, and `checkpoints/qwen/1.2B`, which can be downloaded from [our huggingface hub](https://huggingface.co/collections/MiniLLM/miniplm-6712c0fdf09ef7e8da7d39bd).
```bash
bash scripts/miniplm/pretraining/qwen/200M.sh /PATH/TO/MiniPLM
bash scripts/miniplm/pretraining/qwen/500M.sh /PATH/TO/MiniPLM
bash scripts/miniplm/pretraining/qwen/1.2B.sh /PATH/TO/MiniPLM
```

#### KD Across Model Families
To distill the knowledge of Qwen models to Mamba or LLaMA3.1, first prepare the `config.json` and tokenization-related files in `checkpoints/mamba/130M` and `checkpoints/llama3.1/212M`, which can be downloaded from [our huggingface hub](https://huggingface.co/collections/MiniLLM/miniplm-6712c0fdf09ef7e8da7d39bd). Then, convert the Qwen tokenization to the target tokenization:
```bash
bash scripts/tools/convert_tokenization/convert_tokenization_qwen_mamba.sh /PATH/TO/MiniPLM
bash scripts/tools/convert_tokenization/convert_tokenization_qwen_llama3_1.sh /PATH/TO/MiniPLM
```

NOTE: You may need to setup the environments following the official repo of [Mamba](https://github.com/state-spaces/mamba) before runing the mamba experiments.

### 4.2 Baselines
#### Conventional Pre-Training
```bash
bash scripts/pretrain/qwen/200M.sh /PATH/TO/MiniPLM
bash scripts/pretrain/qwen/500M.sh /PATH/TO/MiniPLM
bash scripts/pretrain/qwen/1.2B.sh /PATH/TO/MiniPLM
```

#### Vanilla KD
```bash
bash scripts/vanilla_kd/qwen/200M.sh /PATH/TO/MiniPLM
bash scripts/vanilla_kd/qwen/500M.sh /PATH/TO/MiniPLM
bash scripts/vanilla_kd/qwen/1.2B.sh /PATH/TO/MiniPLM
```

#### SeqKD
```bash
bash scripts/seqkd/qwen/200M.sh /PATH/TO/MiniPLM
bash scripts/seqkd/qwen/500M.sh /PATH/TO/MiniPLM
bash scripts/seqkd/qwen/1.2B.sh /PATH/TO/MiniPLM
```

#### MiniLLM
We use the official codebase of [MiniLLM](https://github.com/microsoft/LMOps/tree/main/minillm) for this baseline.


## 5 Evaluation
#### LM-Evaluation-Harness
```bash
bash scripts/eval/harness.sh /PATH/TO/MiniPLM --model-path /PATH/TO/TRAINED_CKPT --ckpt-name NAME_OF_CKPT
```
NOTE: The `story_cloze` dataset may require manually downloading. Please follow the instructions in this [link](https://huggingface.co/datasets/LSDSem/story_cloze/blob/734b4e1771508f38d8a05f034b48a42986446669/story_cloze.py#L50) to download the test sets. After downloading, you will need to replace the task configuration file `lm-evaluation-harness/tasks/storycloze/storycloze_2018.yaml` with `configs/lm_harness_tasks/storycloze_2018.yaml`that refers to the downloaded directory.

#### Language Modeling
```bash
bash scripts/eval/lm.sh /PATH/TO/MiniPLM --model-path /PATH/TO/TRAINED_CKPT --ckpt-name NAME_OF_CKPT
```

## 6 Citation
```
@article{miniplm,
    title={MiniPLM: Knowledge Distillation for Pre-Training Language Models}, 
    author={Yuxian Gu and Hao Zhou and Fandong Meng and Jie Zhou and Minlie Huang},
    journal={arXiv preprint arXiv:2410.17215},
    year={2024}
}
```