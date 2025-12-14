# coding=utf-8
# Copyright 2020 The OpenBMB team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import deepspeed
from numerize.numerize import numerize


def add_model_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')
    group.add_argument('--model-path', type=str, help='model path')
    group.add_argument("--ckpt-name", type=str)
    group.add_argument("--model-type", type=str, default=None)
    group.add_argument("--teacher-model-type", type=str, default=None)
    group.add_argument("--n-gpu", type=int, default=1)
    group.add_argument("--n-nodes", type=int, default=1)
    group.add_argument("--teacher-model-path", type=str)
    group.add_argument("--teacher-ckpt-name", type=str)
    group.add_argument("--teacher-model-fp16", action="store_true")
    group.add_argument("--base-model-path", type=str)
    group.add_argument("--base-ckpt-name", type=str)
    group.add_argument("--model-parallel", action="store_true")
    group.add_argument("--model-parallel-size", type=int, default=None)
    group.add_argument("--fp32", action="store_true")
    
    group.add_argument("--attn-impl", type=str, default=None)
    group.add_argument("--xops-attn", action="store_true")
    group.add_argument("--torch-compile", type=str, default=None)
    
    group.add_argument("--ckpt-start", type=int, default=0)
    group.add_argument("--ckpt-end", type=int, default=None)
    
    return parser


def add_runtime_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('runtime', 'runtime configurations')

    group.add_argument("--type", type=str, default=None)
    group.add_argument("--local_rank", type=int, default=None)
    group.add_argument("--do-train", action="store_true")
    group.add_argument("--do-valid", action="store_true")
    group.add_argument("--do-eval", action="store_true")
    group.add_argument("--do-infer", action="store_true")
    group.add_argument('--base-path', type=str, default=None, help='Path to the project base directory.')
    group.add_argument('--load', type=str, default=None,
                       help='Path to a directory containing a model checkpoint.')
    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save-all', action="store_true")
    group.add_argument("--log-interval", type=int, default=10)
    group.add_argument("--mid-log-num", type=int, default=4)
    group.add_argument('--save-interval', type=int, default=10000000,
                       help='number of iterations between saves')
    group.add_argument("--eval-interval", type=int, default=10000000)
    
    group.add_argument('--local-rank', type=int, default=None,
                       help='local rank passed from distributed launcher')
    
    group.add_argument("--save-additional-suffix", type=str, default="")
    
    group.add_argument("--from-scratch", action="store_true")
    group.add_argument("--resume-training", action="store_true")
    group.add_argument("--start-from-global-step", type=int, default=None)
    group.add_argument("--resume-dir", type=str, default=None)
    group.add_argument("--resume-tag", type=str, default=None)
    group.add_argument("--no-eval-when-start", action="store_true")
    group.add_argument("--no-save-when-start", action="store_true")
    
    group.add_argument("--wandb-name", type=str, default=None)
    group.add_argument("--wandb-group", type=str, default=None)
    group.add_argument("--wandb-id", type=str, default=None)
    group.add_argument("--wandb-mode", type=str, default=None)

    group.add_argument('--seed', type=int, default=1234,
                       help='random seed for reproducibility')
    group.add_argument("--seed-data", type=int, default=42)

    return parser


def add_data_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('data', 'data configurations')
    group.add_argument("--dataset-type", type=str, default=None)
    group.add_argument("--data-dir", type=str, default=None)
    group.add_argument("--dev-data-dir", type=str, default=None)
    group.add_argument("--test-data-dir", type=str, default=None)
    group.add_argument("--processed-data-dir", type=str, default=None)
    group.add_argument("--data-process-workers", type=int, default=0)
    group.add_argument("--precompute-data-order", action="store_true")
    group.add_argument("--train-num", type=int, default=None)
    group.add_argument("--train-ratio", type=float, default=1)
    group.add_argument("--dev-num", type=int, default=None)
    group.add_argument("--dev-ratio", type=float, default=1)
    group.add_argument("--test-num", type=int, default=None)
    group.add_argument("--test-ratio", type=float, default=1)
    group.add_argument("--gen-num", type=int, default=None)
    group.add_argument("--infer-num", type=int, default=None)
    group.add_argument("--data-name", type=str, default=None)
    group.add_argument("--prompt-type", type=str, default=None)
    group.add_argument("--num-workers", type=int, default=1)
    group.add_argument("--max-prompt-length", type=int, default=512)
    group.add_argument("--min-prompt-length", type=int, default=128)
    group.add_argument("--ada-max-length", action="store_true")
    group.add_argument("--trunc-data", action="store_true")
    group.add_argument("--json-data", action="store_true")
    group.add_argument("--bin-data", action="store_true")
    group.add_argument("--txt-data", action="store_true")
    group.add_argument("--split-token-id", type=int, default=None)
    group.add_argument("--min-state", type=int, default=0)
    group.add_argument("--max-state", type=int, default=100000)
    group.add_argument("--min-offset", type=int, default=0)
    group.add_argument("--data-split", type=str, default=None)
    group.add_argument("--no-shuffle", action="store_true")
    
    group.add_argument("--eval-ppl", action="store_true")
    group.add_argument("--eval-gen", action="store_true")
    
    
    group.add_argument("--only-prompt", action="store_true")
    group.add_argument("--prompt-data-full-loss", action="store_true",
                       help="Compute loss on the entire sentence in prompt data type.")

    group.add_argument("--chunk-num-per-shard", type=int, default=10000)
    group.add_argument("--max-shard-num", type=int, default=10000000)
    group.add_argument("--max-sample-num", type=int, default=None)
    group.add_argument("--shard-start", type=int, default=0)
    group.add_argument("--shard-end", type=int, default=None)

    return parser


def add_hp_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("hp", "hyper parameter configurations")
    group.add_argument('--batch-size', type=int, default=32,
                       help='Data Loader batch size')
    group.add_argument('--eval-batch-size', type=int, default=32,
                       help='Data Loader batch size')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='gradient clipping')
    group.add_argument('--total-iters', type=int, default=None,
                       help='total number of iterations')
    group.add_argument('--train-iters-per-epoch', type=int, default=None,
                       help='total number of iterations per epoch')
    group.add_argument('--max-length', type=int, default=1024,
                       help='max length of input')
    group.add_argument('--epochs', type=int, default=None,
                       help='total number of epochs to train over all training runs')
    group.add_argument("--gradient-accumulation-steps", type=int, default=1)
    group.add_argument("--gradient-checkpointing", action="store_true")
    
    group.add_argument('--lr', type=float, help='initial learning rate')
    group.add_argument("--lr-min", type=float, default=0.0000001)
    group.add_argument('--weight-decay', type=float, default=1.0e-2,
                       help='weight-decay')
    group.add_argument('--loss-scale', type=float, default=65536,
                       help='loss scale')
    group.add_argument('--optimizer-name', type=str, default='AdamW')
    group.add_argument('--adam-beta', type=float, default=0.9),
    group.add_argument('--adam-beta2', type=float, default=0.999),
    group.add_argument('--adam-eps', type=float, default=1e-8),

    group.add_argument('--warmup-iters', type=int, default=0,
                       help='percentage of data to warmup on (.01 = 1% of all '
                       'training iters). Default 0.01')
    group.add_argument("--scheduler-name", type=str, default="constant")

    return parser


def add_eval_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('evaluation', 'evaluation configurations')
    group.add_argument("--eval-start-ckpt", type=int, default=None)
    group.add_argument("--eval-end-ckpt", type=int, default=None)
    
    # harness
    group.add_argument("--eval-shot", type=int, default=0)
    group.add_argument("--eval-no-calc-stderr", action="store_true")
    group.add_argument("--eval-data-names", type=str, default=None)
    
    return parser


def add_gen_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('generation', 'generation configurations')
    
    group.add_argument("--top-k", type=int, default=None)
    group.add_argument("--top-p", type=float, default=None)
    group.add_argument("--do-sample", action="store_true")
    group.add_argument("--no-repeat-ngram-size", type=int, default=6)
    group.add_argument("--repetition-penalty", type=float, default=None)
    group.add_argument("--num-beams", type=int, default=1)
    group.add_argument("--temperature", type=float, default=1)

    return parser


def add_peft_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('peft', 'peft configurations')
    
    group.add_argument("--peft-path", type=str, default=None)
    group.add_argument("--peft", action="store_true")
    group.add_argument("--teacher-peft-path", type=str, default=None)
    group.add_argument("--teacher-peft", action="store_true")
    
    return parser


def add_kd_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('kd', 'kd configurations')
    
    group.add_argument("--kd-ratio", type=float, default=0.5)
    
    # Offline KD arguments
    group.add_argument("--cached-logits-dir", type=str, default=None,
                       help='Directory containing cached sparse teacher logits (for offline_kd)')
    group.add_argument("--alpha", type=float, default=0.5,
                       help='KD loss weight (0~1, higher = more weight on KD loss)')
    group.add_argument("--kd-temperature", type=float, default=1.0,
                       help='Temperature scaling for KD loss')
    group.add_argument("--kd-method", type=str, default='topk',
                       choices=['topk', 'sparse'],
                       help='Which method to use when cached logits are "both" (topk or sparse)')
    
    return parser


def add_infer_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('inference', 'inference configurations')
    
    group.add_argument("--grouped-infer", action="store_true")
    
    return parser


def base_training_hp_suffix(args):
    suffix = ""
    suffix += (f"e{args.epochs}" if args.epochs is not None else f"t{numerize(args.total_iters)}") + \
        (f"-w{numerize(args.warmup_iters)}" if args.warmup_iters > 0 else "") + \
        (f"-bs{args.batch_size}-lr{args.lr}{args.scheduler_name}{args.lr_min}-G{args.gradient_accumulation_steps}-N{args.n_gpu}-NN{args.n_nodes}") + \
        (f"-mp{args.model_parallel_size}" if args.model_parallel > 0 else "")
    return suffix


def base_infer_hp_suffix(args):
    return ""


def base_model_suffix(args):
    return f"{args.ckpt_name.replace('/', '_')}"


def base_data_suffix(args):
    return f"{args.data_name.replace('/', '_')}"


def gen_path(args):
    s = "sample" if args.do_sample else "greedy"
    s += f"-t{args.temperature}-lp{args.max_prompt_length}-l{args.max_length}-p{args.top_p}-k{args.top_k}"
    return s


def get_parser():
    parser = argparse.ArgumentParser()
    parser = add_model_args(parser)
    parser = add_runtime_args(parser)
    parser = add_data_args(parser)
    parser = add_hp_args(parser)
    parser = add_gen_args(parser)
    parser = add_eval_args(parser)
    parser = add_peft_args(parser)
    parser = add_kd_args(parser)
    parser = add_infer_args(parser)
    parser = deepspeed.add_config_arguments(parser)
    return parser


def get_args():
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    
    assert all(["--" not in x for x in unknown]), unknown
    
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    args.n_gpu = args.n_gpu * args.n_nodes
    
    assert args.model_type is not None
    assert args.data_name is not None
        
    if args.type in ["pretrain"]:
        args.save = os.path.join(
            args.save,
            base_data_suffix(args),
            base_model_suffix(args),
            base_training_hp_suffix(args) + (f"-scr" if args.from_scratch else "") + args.save_additional_suffix
        )
    elif args.type == "vanilla_kd":
        args.save = os.path.join(
            args.save,
            base_data_suffix(args),
            base_model_suffix(args),
            base_training_hp_suffix(args) + (f"-scr" if args.from_scratch else ""),
            f"{args.teacher_ckpt_name.replace('/', '_')}" + f"-kd{args.kd_ratio}" + args.save_additional_suffix,
        )
    elif args.type == "offline_kd":
        # Extract method from cached_logits_dir (topk or sparse)
        logits_method = "topk"  # default
        if args.cached_logits_dir:
            if "sparse" in args.cached_logits_dir:
                logits_method = "sparse"
            elif "topk" in args.cached_logits_dir:
                logits_method = "topk"
        args.save = os.path.join(
            args.save,
            base_data_suffix(args),
            base_model_suffix(args),
            base_training_hp_suffix(args) + (f"-scr" if args.from_scratch else ""),
            f"offline-{logits_method}-a{args.alpha}" + args.save_additional_suffix,
        )
    elif args.type == "seqkd":
        args.save = os.path.join(
            args.save,
            base_data_suffix(args),
            base_model_suffix(args),
            base_training_hp_suffix(args) + (f"-scr" if args.from_scratch else "") + args.save_additional_suffix
        )
    elif args.type == "miniplm":
        args.save = os.path.join(
            args.save,
            base_data_suffix(args),
            base_model_suffix(args),
            base_training_hp_suffix(args) + (f"-scr" if args.from_scratch else "") + args.save_additional_suffix
        )
    elif args.type == "pt_lm_infer":
        args.save = os.path.join(
            args.save,
            base_data_suffix(args),
            base_model_suffix(args),
            base_infer_hp_suffix(args) + args.save_additional_suffix
        )
    elif args.type == "pt_gen_infer":
        args.save = os.path.join(
            base_data_suffix(args),
            base_model_suffix(args),
            gen_path(args) + args.save_additional_suffix
        )
    elif args.type == "eval_harness":
        args.save = os.path.join(
            args.save,
            base_data_suffix(args),
            base_model_suffix(args),
            f"{args.eval_shot}shot" + args.save_additional_suffix
        )
    elif args.type == "eval_lm":
        args.save = os.path.join(
            args.save,
            base_data_suffix(args),
            base_model_suffix(args) + args.save_additional_suffix
        )
    elif args.type == "tokenize":
        pass
    elif args.type == "dummy":
        pass
    else:
        raise NotImplementedError(args.type)

    return args
