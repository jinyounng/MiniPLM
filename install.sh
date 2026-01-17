export NCCL_DEBUG=""
pip3 install numpy==1.26.4
pip3 install torch
pip3 install transformers
pip3 install deepspeed
pip3 install numerize
pip3 install accelerate
pip3 install datasets
pip3 install wandb
pip3 install matplotlib
pip3 install sentencepiece
pip3 install peft
git clone https://github.com/EleutherAI/lm-evaluation-harness
pip3 install -e lm-evaluation-harness