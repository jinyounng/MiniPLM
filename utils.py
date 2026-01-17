import torch
import deepspeed
import torch.distributed as dist
import random
import numpy as np
import os
from datetime import timedelta
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import time
from torch.distributed import get_rank
from accelerate import load_checkpoint_and_dispatch, init_empty_weights
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM

try:
    from transformers import mpu, parallel_model_map
except:
    mpu = None
    parallel_model_map = None

WANDB_PROJ_NAME="MiniPLM"
PAD_EOS_MODELS = ["llama", "mistral", "gpt2", "llama3_1", "gptj", "stable_lm", "opt", "qwen", "mamba"]
BOS_MODELS = ["llama", "mistral", "fairseq", "llama3_1"]
POSITION_ID_MODELS = ["gpt2"]


def get_distribution(logits, temperature):
    probs = torch.softmax(logits.to(torch.float32) / (temperature + 1e-10), dim=-1, dtype=torch.float32)
    return probs


def sample(logits, temperature):
    probs = get_distribution(logits, temperature)
    return torch.multinomial(probs, num_samples=1)


def sample_from_draft_model(model, initial_prompt_seq, new_tokens, eos_token_id, temperature=1.0):
    fin_prompt_seq = initial_prompt_seq.detach().clone()
    out_logits = []

    for _ in range(new_tokens):
        sample_token_logits = model(fin_prompt_seq).logits[:, -1, :]
        sample_token = sample(sample_token_logits, temperature=temperature)
        fin_prompt_seq = torch.concat([fin_prompt_seq, sample_token], dim=-1)
        out_logits.append(sample_token_logits)
        if sample_token == eos_token_id:
            break        

    out_logits = torch.stack(out_logits, dim=1)
    return fin_prompt_seq, out_logits


# Logging
def print_args(args):
    """Print arguments."""

    print('arguments:', flush=True)
    for arg in vars(args):
        dots = '.' * (29 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)


def save_rank(log_str, save_path, rank=0):
    if not dist.is_initialized() or dist.get_rank() == rank:
        with open(save_path, "a") as f:
            f.write(log_str + "\n")


def print_rank(*args, rank=0, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == rank:
        print(*args, **kwargs)


def print_and_save_rank(log_str, save_path, rank=0, **kwargs):
    print_rank(log_str, rank=rank, **kwargs)
    save_rank(log_str, save_path, rank=rank)


# Distributed
def all_gather(t, dim=0, world_size=None, group=None, op="cat"):
    if world_size is None:
        world_size = dist.get_world_size()
    all_t = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(all_t, t, group=group)
    if op == "cat":
        all_t = torch.cat(all_t, dim=dim)
    elif op == "stack":
        all_t = torch.stack(all_t, dim=dim)
    return all_t


# Initialize
def set_random_seed(seed, mp=False):
    """Set random seed for reproducability."""
    if dist.is_initialized():
        seed = dist.get_rank() + seed
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # if mp:
        #     mpu.model_parallel_cuda_manual_seed(seed)


def init_distributed(args):
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if args.rank == 0:
        print(f"using world size: {args.world_size}")

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)

    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=300))


def init_distributed_ds(args):
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if args.rank == 0:
        print(f"using world size: {args.world_size}")

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)

    deepspeed.init_distributed(timeout=timedelta(minutes=300))


def initialize(args, do_distributed=True):
    # init distributed
    if do_distributed:
        if args.deepspeed:
            init_distributed_ds(args)
        else:
            init_distributed(args)

    # if args.model_parallel:
    #     assert dist.get_world_size() % args.model_parallel_size == 0 
    #     mpu.initialize_model_parallel(args.model_parallel_size)

    set_random_seed(args.seed, args.model_parallel)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
        
        
# Load and save model
def _get_base_model(args, device, model_path=None, config=None, from_scratch=None, model_cls=None, model_type=None):
    model_path = args.model_path if model_path is None else model_path
    model_type = args.model_type if model_type is None else model_type
    
    print_and_save_rank("Initializing model from {}".format(model_path), os.path.join(args.save, "log.txt"))
    print_and_save_rank(f"Attention Implementation: {args.attn_impl}", os.path.join(args.save, "log.txt"))
    if config is None:
        config = AutoConfig.from_pretrained(model_path, attn_implementation=args.attn_impl)
        
    if args.xops_attn:
        assert args.attn_impl == "eager"
        import xformers
        print_and_save_rank("Xops Attention", os.path.join(args.save, "log.txt"))
        config.use_memory_efficient_attention = True

    st_time = time.time()
    if args.model_parallel:
        config.is_model_parallel = True
        with init_empty_weights():
            model = parallel_model_map[model_type].half()
        load_parallel(model, model_path)

        if mpu.get_data_parallel_rank() == 0:
            print_and_save_rank(' > number of parameters on model parallel rank {}: {}'.format(
                mpu.get_model_parallel_rank(),
                sum([p.nelement() for p in model.parameters()])),
                os.path.join(args.save, "log.txt"),
                flush=True)
    else:
        config.is_model_parallel = False
        from_scratch = from_scratch if from_scratch is not None else args.from_scratch
        model_cls = model_cls if model_cls is not None else AutoModelForCausalLM
        if from_scratch:
            dtype = torch.float32 if args.fp32 else torch.float16
            model = model_cls.from_config(config, attn_implementation=args.attn_impl, torch_dtype=dtype).to(device)
        else:
            dtype = torch.float32 if args.fp32 else torch.float16
            model = model_cls.from_pretrained(model_path, config=config, device_map={"": device}, torch_dtype=dtype)

        if dist.get_rank() == 0:
            print_and_save_rank(' > number of parameters: {}'.format(
                sum([p.nelement() for p in model.parameters()])),
                os.path.join(args.save, "log.txt")
                ,flush=True)
        # model = DDP(model)
        # NOTE: no need for DDP since deepspeed has done
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    ed_time = time.time()
    
    print_and_save_rank(f"Model load time: {ed_time - st_time}s", os.path.join(args.save, "log.txt"))
    
    return model


def _get_peft_model(args, device, peft_path=None, base_model_path=None, from_scratch=None, model_cls=None, model_type=None):
    peft_path = args.peft_path if peft_path is None else peft_path
    from_scratch = args.from_scratch if from_scratch is None else from_scratch
    
    assert peft_path is not None, "PEFT path is not specified"
    print_and_save_rank("Loading peft config from {}".format(peft_path), os.path.join(args.save, "log.txt"))
    peft_config = LoraConfig.from_pretrained(peft_path)
    if from_scratch:    
        if peft_config.base_model_name_or_path is not None:
            assert base_model_path is None, f"Model path cannot be specified as {base_model_path} when loading peft model"
            base_model_path = peft_config.base_model_name_or_path
            base_model = _get_base_model(
                args, device, model_path=base_model_path, config=None, from_scratch=False, model_cls=model_cls, model_type=model_type)
        else:
            base_model = _get_base_model(
                args, device, model_path=base_model_path, config=None, from_scratch=False, model_cls=model_cls, model_type=model_type)

        model = get_peft_model(base_model, peft_config)
    else:
        assert peft_config.base_model_name_or_path is not None, "Base model path is not specified when loading peft model"
        print_and_save_rank("Loading peft model from {}".format(peft_path), os.path.join(args.save, "log.txt"))
        dtype = torch.float32 if args.fp32 else torch.float16
        model = AutoPeftModelForCausalLM.from_pretrained(peft_path, device_map={"": device}, torch_dtype=dtype)

    if get_rank() == 0:
        print_and_save_rank("Peft model:", os.path.join(args.save, "log.txt"))
        model.print_trainable_parameters()
        
    return model


def get_model(args, device, model_path=None, config=None, from_scratch=None, model_cls=None, model_type=None, peft=None, peft_path=None):
    peft = args.peft if peft is None else peft
    if peft:
        return _get_peft_model(args, device,
                peft_path=peft_path, base_model_path=model_path, from_scratch=from_scratch, model_cls=model_cls, model_type=model_type)
    else:
        return _get_base_model(args, device,
                model_path=model_path, config=config, from_scratch=from_scratch, model_cls=model_cls, model_type=model_type)


def get_tokenizer(args, model_path=None, model_type=None, peft=None, peft_path=None):
    peft = args.peft if peft is None else peft
    model_type = args.model_type if model_type is None else model_type
    model_path = args.model_path if model_path is None else model_path
    
    if peft:
        peft_path = args.peft_path if peft_path is None else peft_path
        peft_config = LoraConfig.from_pretrained(peft_path)
        if peft_config.base_model_name_or_path is not None:
            assert model_path is None, f"Model path cannot be specified as {model_path} when loading peft model"
            base_model_path = peft_config.base_model_name_or_path
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        else:
            assert model_path is not None, "Model path is not specified when loading peft model"
            tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if model_type in PAD_EOS_MODELS:
        # print_and_save_rank("tokenizer: pad = eos", os.path.join(args.save, "log.txt"))
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer


def load_parallel(model, load_dir):
    mp_rank = mpu.get_model_parallel_rank()
    assert mpu.get_model_parallel_world_size() != 1
    checkpoint_name = os.path.join(load_dir, f"mp{mpu.get_model_parallel_world_size()}", f"pytorch_model_{mp_rank}.bin")
    assert os.path.exists(checkpoint_name), f"{checkpoint_name} does not exist."
    model = load_checkpoint_and_dispatch(model=model, checkpoint=checkpoint_name, device_map={"": torch.cuda.current_device()}, dtype=torch.float16)
    dist.barrier()
    print(f"Rank {get_rank()}: {checkpoint_name} loaded.")


def save_parallel(model, save_dir):
    mp_rank = mpu.get_model_parallel_rank()
    os.makedirs(os.path.join(save_dir, f"mp{mpu.get_model_parallel_world_size()}"), exist_ok=True)
    checkpoint_name = os.path.join(save_dir, f"mp{mpu.get_model_parallel_world_size()}", f"pytorch_model_{mp_rank}.bin")
    torch.save(model.state_dict(), checkpoint_name)
    print(f"Rank {get_rank()}: {checkpoint_name} saved.")
