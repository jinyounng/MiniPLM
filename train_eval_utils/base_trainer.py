import os
import uuid
import json
import math
import wandb
import random
import deepspeed
import numpy as np
from time import time
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import get_rank
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torch.optim import AdamW, SGD, Adam
from data_utils.prompt_datasets import PromptDataset

from transformers import (
    GenerationConfig,
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from utils import print_rank, save_rank, save_parallel, all_gather, print_and_save_rank
from utils import get_model, get_tokenizer
from utils import WANDB_PROJ_NAME

from .schedulers import WarmupCosineAnnealingLR

try:
    from transformers import mpu
except ImportError:
    mpu = None


class BaseTrainer():
    def __init__(self, args, ds_config, device, do_train=True):
        self.args = args
        self.ds_config = ds_config
        self.device = device
        self.do_train = do_train
        self.tokenizer = get_tokenizer(args)
        self.grad_norm = 0
        self.exp_name = args.save.strip("/").replace(args.base_path.strip("/"), "").replace("_", "").replace("/", "_").strip("_")
        self.wandb_name = self.args.wandb_name if self.args.wandb_name is not None else self.exp_name
        self.group_name = self.args.wandb_group or "pad"
        self.global_steps = None
        self.steps = None
        self.epoch = None
        self.epochs = None
        self.total_steps = None
        self.first_printed = False
        if self.args.start_from_global_step is not None:
            self.last_global_steps = self.args.start_from_global_step
        
        if args.model_parallel:
            self.dp_world_size = mpu.get_data_parallel_world_size()
            self.dp_rank = mpu.get_data_parallel_rank()
            self.dp_group = mpu.get_data_parallel_group()
        else:
            self.dp_world_size = dist.get_world_size()
            self.dp_rank = dist.get_rank()
            self.dp_group = None
    
    def get_model(self, args=None, device=None):
        args = args or self.args
        device = device or self.device
        return get_model(args, device)
    
    def get_optimizer(self, model, args=None):
        args = args or self.args
        if self.args.optimizer_name.lower() == "sgd":
            optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif self.args.optimizer_name.lower() == "adam":
            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_eps, betas=(args.adam_beta, args.adam_beta2))
        elif self.args.optimizer_name.lower() == "adamw":
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_eps, betas=(args.adam_beta, args.adam_beta2))
        else:
            raise ValueError(f"Optimizer of type {self.args.optimizer_name} is not supported yet.")
        print_and_save_rank(f'Optimizer = {optimizer.__class__.__name__}', os.path.join(args.save, "log.txt"))
        return optimizer
        
    def get_lr_scheduler(self, optimizer, args=None):
        args = args or self.args
        assert self.total_steps is not None and self.total_steps > 0
        if args.scheduler_name == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_iters)
        elif args.scheduler_name == "cosine":
            lr_scheduler = WarmupCosineAnnealingLR(
                optimizer,
                T_max=self.total_steps,
                warmup_steps=args.warmup_iters,
                eta_min=args.lr_min)
        elif args.scheduler_name == "noam":
            lr_scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_iters,
                num_training_steps=self.total_steps,
                power=0.5)
        else:
            raise ValueError(f"lr_scheduler of type {args.scheduler_name} is not supported yet.")

        return lr_scheduler
    
    def setup_model_and_optimizer(self, args=None, ds_config=None, device=None, set_optim=True):
        args = args or self.args
        device = device or self.device
        ds_config = ds_config or self.ds_config
        # get the model
        model = self.get_model(args, device)
        # get the optimizer and lr_scheduler
        if set_optim:
            optimizer = self.get_optimizer(model, args)
            lr_scheduler = self.get_lr_scheduler(optimizer, args)
        else:
            optimizer, lr_scheduler = None, None
            
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            lr_scheduler=lr_scheduler,
            mpu=mpu if args.model_parallel else None,
            config_params=ds_config
        )
        
        # get the memory usage
        print_and_save_rank("Model mem\n", torch.cuda.memory_summary(), os.path.join(args.save, "log.txt"))
        
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        if self.args.torch_compile is not None:
            print_and_save_rank(f"Torch Compile Mode: {self.args.torch_compile}", os.path.join(args.save, "log.txt"))
            self.model = torch.compile(self.model, mode=self.args.torch_compile)

    def resume_training(self):
        load_dir = self.args.resume_dir or self.args.save
        if self.args.resume_tag is None:
            with open(os.path.join(load_dir, "latest")) as f:
                tag = f.read().strip()
        else:
            tag = self.args.resume_tag
        self.model.load_checkpoint(load_dir, tag=tag)
        self.last_rng_states = torch.load(os.path.join(load_dir, tag, f"rng_states_{get_rank()}.pt"))
        
        with open(os.path.join(load_dir, tag, "dynamics.json"), "r") as f:
            dynamics = json.load(f)
        self.last_steps = dynamics["step"]
        self.last_epochs = dynamics["epoch"]
        self.last_global_steps = dynamics["global_steps"]
        self.train_dataset.set_skip_offset(dynamics["skip_offset"])
        
        print_and_save_rank(f"Resume from {load_dir} {tag}", os.path.join(self.args.save, "log.txt"))
        print_and_save_rank(f"Resume from step {self.last_steps}, epoch {self.last_epochs}, global step {self.last_global_steps}",
                            os.path.join(self.args.save, "log.txt"))
 
    def prepare_learning(self, args=None):
        args = args or self.args
        self.total_batch_size = args.batch_size * self.dp_world_size * args.gradient_accumulation_steps
        # rank 분할 데이터셋(offline_kd 등): 각 rank가 이미 1/8 데이터를 갖고 있음
        # DistributedSampler를 쓰면 또 8등분해서 1/64만 사용하게 되므로, 로컬 sampler 사용
        # train_iters_per_epoch는 global step 기준이어야 함 (total_iters와 단위 맞춤)
        rank_split = getattr(self.train_dataset, "global_len", None) is not None
        if rank_split:
            per_rank_len = len(self.train_dataset)
            # global step per epoch = per_rank_len / (batch_size * grad_acc)
            # 이렇게 해야 Pretrain과 같은 global step 기준이 됨
            per_rank_samples_per_step = args.batch_size * args.gradient_accumulation_steps
            self.train_iters_per_epoch = per_rank_len // per_rank_samples_per_step
            self.train_dataset.set_num(self.train_iters_per_epoch * per_rank_samples_per_step)
        else:
            effective_len = len(self.train_dataset)
            self.train_iters_per_epoch = int(effective_len / self.total_batch_size)
            self.train_dataset.set_num(self.train_iters_per_epoch * self.total_batch_size)
        assert (args.epochs is not None) ^ (args.total_iters is not None), (args.epochs, args.total_iters)
        self.total_steps = args.total_iters or self.train_iters_per_epoch * args.epochs
        self.epochs = args.epochs or math.ceil(args.total_iters / self.train_iters_per_epoch)
        
        if args.save_interval == -1:
            args.save_interval = self.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = self.train_iters_per_epoch

        if self.args.precompute_data_order and (not self.args.resume_training):
            if get_rank() == 0:
                normal_order = np.arange(0, len(self.train_dataset), dtype=np.int32)
                order = np.stack([np.random.permutation(normal_order) for _ in range(self.epochs)], axis=0)
                # rank 분할이면 per_rank 샘플 수, 아니면 전체 샘플 수
                if rank_split:
                    drop_last_len = self.train_iters_per_epoch * self.args.batch_size * self.args.gradient_accumulation_steps
                else:
                    drop_last_len = self.train_iters_per_epoch * self.total_batch_size
                order = order[:, :drop_last_len]
                np.save(os.path.join(self.args.save, "data_order.npy"), order)
                print("Data order size: ", order.shape)
            dist.barrier()
            self.train_dataset.set_order(path=os.path.join(self.args.save, "data_order.npy"))
                        
        if self.args.resume_training:
            assert self.args.precompute_data_order
            assert os.path.exists(os.path.join(self.args.save, "data_order.npy"))
            self.train_dataset.set_order(path=os.path.join(self.args.save, "data_order.npy"))

        print_and_save_rank(f"Total batch size: {self.total_batch_size}", os.path.join(args.save, "log.txt"))
        print_and_save_rank(f"Total iters: {self.total_steps}", os.path.join(args.save, "log.txt"))
        print_and_save_rank(f"Total epochs: {self.epochs}", os.path.join(args.save, "log.txt"))
        print_and_save_rank(f"Train iters per epoch: {self.train_iters_per_epoch}", os.path.join(args.save, "log.txt"))
        print_and_save_rank(f"Save interval: {args.save_interval}", os.path.join(args.save, "log.txt"))
        print_and_save_rank(f"Eval interval: {args.eval_interval}", os.path.join(args.save, "log.txt"))
    
        self.train_dataloader, self.train_sampler = self.get_train_sampler_dataloader()
    
    def get_train_sampler_dataloader(self):
        rank_split = getattr(self.train_dataset, "global_len", None) is not None
        if rank_split:
            # 각 rank가 이미 서로 다른 데이터 조각을 갖고 있으므로 DistributedSampler 사용 시 1/8만 사용됨 → 전체 사용하려면 로컬 sampler
            shuffle = (not self.args.precompute_data_order) and (not self.args.no_shuffle)
            train_sampler = RandomSampler(self.train_dataset) if shuffle else SequentialSampler(self.train_dataset)
        else:
            train_sampler = DistributedSampler(self.train_dataset, shuffle=((not self.args.precompute_data_order) and (not self.args.no_shuffle)), drop_last=True, rank=self.dp_rank, num_replicas=self.dp_world_size)
        train_dataloader = DataLoader(
            self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size, num_workers=self.args.num_workers, collate_fn=self.train_dataset.collate, drop_last=True)
        return train_dataloader, train_sampler
    
    def prepare_inference(self, args=None):
        pass
     
    def set_datasets(self, args=None, do_train=True):
        args = args or self.args
        if do_train:
            self.train_dataset = PromptDataset(args, self.tokenizer, "train", args.data_dir, args.train_num, ada_max_length=args.ada_max_length)
            print_and_save_rank("### Training Data Number:", len(self.train_dataset), os.path.join(args.save, "log.txt"))
            if args.dev_data_dir is not None:
                self.eval_dataset = PromptDataset(args, self.tokenizer, "dev", args.dev_data_dir, args.dev_num, ada_max_length=args.ada_max_length)
                print_and_save_rank("### Dev Data Number:", len(self.eval_dataset), os.path.join(args.save, "log.txt"))
            else:
                self.eval_dataset = None
        else:
            self.eval_dataset = PromptDataset(args, self.tokenizer, "test", args.test_data_dir, args.dev_num, ada_max_length=args.ada_max_length)

    def compute_loss(self, model_batch, no_model_batch):
        raise NotImplementedError

    def _get_lm_loss_from_logits(self, logits, label, loss_mask):        
        if self.args.model_parallel:
            loss_func = mpu.parallel_cross_entropy
            lm_losses = loss_func(logits.contiguous().float(), label)
        else:
            loss_func = nn.CrossEntropyLoss(reduction="none")
            lm_losses = loss_func(logits.float().view(-1, logits.shape[-1]), label.view(-1))
            lm_losses = lm_losses.view(-1, label.size(-1))
        assert all(torch.sum(loss_mask, dim=-1) > 0)
        lm_loss = torch.sum((lm_losses * loss_mask), dim=-1) / torch.sum(loss_mask, dim=-1)
        return lm_loss

    def compute_lm_loss(self, model_batch, no_model_batch, mean=True):        
        outputs = self.model(**model_batch, use_cache=False)
        logits = outputs.logits

        lm_loss = self._get_lm_loss_from_logits(logits, no_model_batch["label"], no_model_batch["loss_mask"])
        
        if mean:
            lm_loss = lm_loss.mean()            
        
        return lm_loss

    def print_and_save(self, log_str, output_path=None):
        output_path = output_path or self.args.save
        print_rank(log_str)
        save_rank(log_str, os.path.join(output_path, "log.txt"))

    def get_log(self, stats, phase, **kwargs):
        log_prefix = "{} | epoch {}/{} | steps {} | global_steps {}/{}".format(
            phase,
            self.epoch,
            self.epochs,
            self.steps,
            self.global_steps,
            self.total_steps
        )
        
        log_midfix = " | ".join([f"{k}: {v:.4f}" for k,v in stats.items()])
        log_suffix = " | ".join([(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}") for k,v in kwargs.items()])
        
        return log_prefix + " | " + log_midfix + " | " + log_suffix

    def backward(self, loss, loss_stats=None):
        self.model.backward(loss)

    def _all_reduce_loss(self, loss):
        _loss = loss.clone().detach()
        dist.all_reduce(_loss, group=self.dp_group, op=dist.ReduceOp.SUM)
        return (_loss / self.dp_world_size).item()

    def first_print(self, model_batch, no_model_batch, save_name=""):
        if self.dp_rank == 0:
            if "input_ids" in model_batch:
                print("#### input_ids BEGIN ####")
                print(model_batch["input_ids"][0].cpu().tolist())
                print(self.tokenizer.decode(model_batch["input_ids"][0].cpu().tolist(), skip_special_tokens=True))
                print("#### Size:", model_batch["input_ids"].size(), "####")
                print("#### input_ids END ####")
            if "attention_mask" in model_batch:
                print("#### attention_mask BEGIN ####")
                print(model_batch["attention_mask"][0].cpu().tolist())
                print("#### attention_mask END ####")
            if "label" in no_model_batch:
                print("#### label BEGIN ####")
                print(no_model_batch["label"][0].cpu().tolist())
                print("#### label END ####")
            if "loss_mask" in no_model_batch:
                print("#### loss_mask BEGIN ####")
                print(no_model_batch["loss_mask"][0].int().cpu().tolist())
                print("#### loss_mask END ####")
            torch.save(model_batch, os.path.join(self.args.save, f"model_batch_{save_name}_0.pt"))
            torch.save(no_model_batch, os.path.join(self.args.save, f"no_model_batch_{save_name}_0.pt"))

    def set_train(self):
        self.model.train()

    def _train_pass(self, model_batch, no_model_batch, stats):
        self.preforward_callback()
        # forward
        torch.cuda.synchronize()
        forward_time = time()
        loss, loss_stats = self.compute_loss(model_batch, no_model_batch)
        stats.update({k:v for k,v in loss_stats.items() if "NO_LOGGING" not in k})
        torch.cuda.synchronize()
        forward_time = time() - forward_time

        # backward
        backward_time = time()
        self.backward(loss, loss_stats)
        torch.cuda.synchronize()
        backward_time = time() - backward_time

        self.grad_norm = 0.0
        if self.model.is_gradient_accumulation_boundary():
            if self.args.fp32:
                self.grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            else:
                self.grad_norm = self.optimizer.scaled_global_norm() / self.optimizer.cur_scale
                
        # step
        step_time = time()
        self.model.step()
        torch.cuda.synchronize()
        step_time = time() - step_time

        stats["loss"] = self._all_reduce_loss(loss)
            
        elapsed_time = forward_time + backward_time + step_time
        stats["elasped_time"] = elapsed_time

        self.post_backward_callback()

        return stats

    def train(self): 
        self.steps = 0
        self.global_steps = 1
        self.epoch = 0
               
        logging_stats = defaultdict(float)
        if self.args.do_train and self.dp_rank == 0:
            wandb_id = self.args.wandb_id or (str(int(time())) + "-" + str(uuid.uuid4()))
            run = wandb.init(
                id=wandb_id,
                name=self.wandb_name,
                project=WANDB_PROJ_NAME,
                group=self.group_name,
                config=self.args,
                reinit=True,
                tags=[self.args.time_stamp, self.args.data_name],
                mode=self.args.wandb_mode)
        
        if self.args.do_train and self.args.do_valid and not self.args.resume_training and not self.args.no_eval_when_start:
            self.evaluate()
        
        if self.args.do_train and not self.args.resume_training and not self.args.no_save_when_start:
            self.save(self.args.save, global_steps=0)
            
        st_time = time()
        
        assert self.epochs is not None
        assert self.total_steps is not None
        
        for epoch in range(0, self.epochs):
            self.set_train()
            self.epoch = epoch
            if isinstance(self.train_sampler, DistributedSampler):
                self.train_sampler.set_epoch(epoch)
            self.train_dataset.set_epoch(epoch)
            self.preepoch_callback()
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{self.epochs}",
                disable=(self.dp_rank != 0),
                dynamic_ncols=True,
            )
            for it, (model_batch, no_model_batch) in enumerate(progress_bar):
                if self.args.resume_training or (self.args.start_from_global_step is not None):
                    if self.global_steps <= self.last_global_steps:
                        if (self.steps % self.args.gradient_accumulation_steps == 0) and (self.global_steps % 1000 == 0):
                            print_and_save_rank(f"Skipping global step {self.global_steps}", os.path.join(self.args.save, "log.txt"))                        
                        self.steps += 1
                        if self.steps % self.args.gradient_accumulation_steps == 0:
                            self.global_steps += 1
                        continue
                    if (self.steps % self.args.gradient_accumulation_steps == 0):
                        print_and_save_rank(f"Starting from global step {self.global_steps}", os.path.join(self.args.save, "log.txt"))
                        if self.args.resume_training:
                            torch.set_rng_state(self.last_rng_states["torch"])
                            torch.cuda.set_rng_state(self.last_rng_states["cuda"])
                            np.random.set_state(self.last_rng_states["numpy"])
                            random.setstate(self.last_rng_states["python"])

                if not self.first_printed:
                    self.first_print(model_batch, no_model_batch, "train")
                    self.first_printed = True

                self.train_dataset.move_to_device(model_batch, no_model_batch, self.device)
                                
                stats = {}
                stats = self._train_pass(model_batch, no_model_batch, stats)
                                
                # logging
                for k in stats:
                    logging_stats[k] += stats[k]
                
                mid_log_step = self.args.gradient_accumulation_steps // self.args.mid_log_num
                mid_log_step = 1 if mid_log_step == 0 else mid_log_step
                
                # print first step
                if self.steps == 0:
                    print_and_save_rank(self.get_log(stats, "train",
                        it=it,
                        lr="{:.4e}".format(self.lr_scheduler.get_last_lr()[0]),
                        scale=self.optimizer.cur_scale if hasattr(self.optimizer, "cur_scale") else 0),
                        os.path.join(self.args.save, "log.txt"))
                    print_rank("-" * 100)
                    print_rank("-" * 100)
                
                if (self.args.mid_log_num > 0) and ((self.steps+1) % mid_log_step == 0):
                    print_rank(self.get_log(stats, "train",
                                            it=it,
                                            lr="{:.4e}".format(self.lr_scheduler.get_last_lr()[0]),
                                            scale=self.optimizer.cur_scale if hasattr(self.optimizer, "cur_scale") else 0),)

                
                if (self.args.gradient_accumulation_steps == 1 or self.steps > 0) and \
                    (self.global_steps > 0) and \
                        (self.global_steps % self.args.log_interval == 0) and \
                             ((self.steps+1) % self.args.gradient_accumulation_steps == 0):
                    logging_stats = {k:v/(self.args.log_interval*self.args.gradient_accumulation_steps) for k,v in logging_stats.items()}
                    now_time = time()
                    real_step_time = (now_time - st_time) / self.args.log_interval
                    st_time = now_time
                    log_str = self.get_log(logging_stats, "train", 
                                           it=it,
                                           grad_norm="{:.4f}".format(self.grad_norm),
                                           lr="{:.4e}".format(self.lr_scheduler.get_last_lr()[0]),
                                           scale=self.optimizer.cur_scale if hasattr(self.optimizer, "cur_scale") else 0,
                                           step_time=logging_stats.get("elasped_time", 0) * self.args.gradient_accumulation_steps,
                                           real_step_time = real_step_time)
                    
                    if self.dp_rank == 0:
                        wandb_logging_stats = {
                            **logging_stats,
                            "grad_norm": self.grad_norm,
                            "lr": self.lr_scheduler.get_last_lr()[0],
                            "scale": self.optimizer.cur_scale if hasattr(self.optimizer, "cur_scale") else 0,
                            "step_time": logging_stats.get("elasped_time", 0) * self.args.gradient_accumulation_steps,
                        }
                        
                        wandb.log(wandb_logging_stats, step=self.global_steps)
                    
                    print_rank("*" * 100)
                    print_rank(log_str)
                    print_rank(self.args.save)
                    print_rank("*" * 100)
                    save_rank(log_str, os.path.join(self.args.save, "log.txt"))
                    logging_stats = {k:0 for k in logging_stats}
                    
                    # exit(0)

                # save
                if (self.steps > 0) and (self.global_steps > 0) and ((self.steps+1) % self.args.gradient_accumulation_steps == 0) and \
                    (self.global_steps % self.args.save_interval == 0):
                    self.save(self.args.save)

                # eval
                if (self.steps > 0) and (self.global_steps > 0) and ((self.steps+1) % self.args.gradient_accumulation_steps == 0) and \
                    (self.global_steps % self.args.eval_interval == 0):
                    if self.args.do_valid:
                        self.evaluate()
                    self.set_train()

                # end
                if ((self.steps+1) % self.args.gradient_accumulation_steps == 0) and (self.global_steps >= self.total_steps):
                    self.save(self.args.save)
                    if self.args.do_valid:
                        self.evaluate()
                    return
                                
                self.steps += 1
                if self.steps % self.args.gradient_accumulation_steps == 0:
                    self.global_steps += 1
            
            self.post_epoch_callback()

        if self.args.do_infer:
            self.inference()
        
        if self.args.do_eval:
            self.evaluate()

        if self.args.do_train and self.dp_rank == 0:
            run.finish()

    def inference(self):
        pass

    def evaluate(self):
        raise NotImplementedError

    def _avg_loss_cross_dp(self, all_losses):
        all_losses = all_gather(all_losses, dim=1, group=self.dp_group, world_size=self.dp_world_size, op="stack")
        all_losses = all_losses.view(-1)
        avg_loss = all_losses.mean().item()
        return avg_loss

    def evaluate_lm(self, eval_dataset=None):
        eval_dataset = eval_dataset or self.eval_dataset
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False, drop_last=False, rank=self.dp_rank, num_replicas=self.dp_world_size)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=eval_dataset.collate)
        
        self.model.eval()
        all_losses = []
                    
        with torch.no_grad():
            for i, (model_batch, no_model_batch) in enumerate(tqdm(eval_dataloader, f"LM Evaluation", disable=(not get_rank() == 0))):
                if i == 0 and self.dp_rank == 0:
                    self.first_print(model_batch, no_model_batch, f"eval_{eval_dataset.data_name}")
                eval_dataset.move_to_device(model_batch, no_model_batch, self.device)
                loss = self.compute_lm_loss(model_batch, no_model_batch, mean=False)
                all_losses.append(loss)
        
        all_losses = torch.cat(all_losses, dim=0)
        avg_loss = self._avg_loss_cross_dp(all_losses)

        if get_rank() == 0:
            res = {"avg_loss": avg_loss}
        else:
            res = None
        
        dist.barrier()
        return res
        
    def evaluate_gen(self):
        eval_sampler = DistributedSampler(self.eval_dataset, shuffle=False, drop_last=False, rank=self.dp_rank, num_replicas=self.dp_world_size)
        eval_dataloader = DataLoader(
            self.eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=self.eval_dataset.collate_gen)
        
        self.model.eval()
        all_prompt_ids, all_response_ids = [], []
        all_gen_times = []
                    
        with torch.no_grad():
            for model_batch, no_model_batch in tqdm(eval_dataloader, f"Generation Evaluation", disable=(not get_rank() == 0)):
                self.eval_dataset.move_to_device(model_batch, no_model_batch, self.device)
                prompt_ids = model_batch["input_ids"]
                st = time()
                gen_out = self.generate(model_batch)
                gen_time = time() - st
                response_ids = gen_out["sequences"][:, prompt_ids.size(1):]
                all_prompt_ids.append(torch.nn.functional.pad(prompt_ids, (self.args.max_prompt_length-prompt_ids.size(1), 0), value=self.tokenizer.pad_token_id))
                all_response_ids.append(torch.nn.functional.pad(response_ids, (0, self.args.max_length-response_ids.size(1)), value=self.tokenizer.pad_token_id))
                all_gen_times.append(gen_time)
        
        all_prompt_ids = torch.cat(all_prompt_ids, dim=0)
        all_prompt_ids = all_gather(all_prompt_ids, dim=1, group=self.dp_group, world_size=self.dp_world_size, op="stack")
        all_prompt_ids = all_prompt_ids.view(-1, all_prompt_ids.size(-1))
        all_prompt_ids = all_prompt_ids[:len(self.eval_dataset)]
        all_response_ids = torch.cat(all_response_ids, dim=0)
        all_response_ids = all_gather(all_response_ids, dim=1, group=self.dp_group, world_size=self.dp_world_size, op="stack")
        all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
        all_response_ids = all_response_ids[:len(self.eval_dataset)]

        all_gen_times = all_gather(torch.tensor(all_gen_times, device=self.device), dim=1, group=self.dp_group, world_size=self.dp_world_size, op="stack").view(-1)
        gen_time = all_gen_times.sum().item()

        if get_rank() == 0:
            response_strs = self.tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)            
            res = self.compute_metrics(response_strs[:len(self.eval_dataset.answers)], self.eval_dataset.answers)
        else:
            res, response_strs = None, None
        
        dist.barrier()
        return all_prompt_ids, all_response_ids, res, response_strs

    def get_generation_config(self, batch, **kwargs):
        max_new_tokens = self.args.max_length - batch["input_ids"].size(1)
        generation_dict = dict(
            do_sample=self.args.do_sample,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            temperature=self.args.temperature,
            max_length=self.args.max_length,
            min_length=None,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,            
        )
        generation_dict.update(kwargs)
        generation_config, unused_kwargs = GenerationConfig.from_dict(generation_dict, return_unused_kwargs=True)
        if len(unused_kwargs) > 0:
            print_and_save_rank(f"Unused kwargs in generation config: {unused_kwargs}", os.path.join(self.args.save, "log.txt"))
        return generation_config
    
    def generate(self, batch, **kwargs):
        generation_config = self.get_generation_config(batch, **kwargs)
        gen_out = self.model.generate(**batch, generation_config=generation_config)
        return gen_out
    
    def preepoch_callback(self):
        print_and_save_rank(f"new epoch {self.epoch}", os.path.join(self.args.save, "log.txt"))
    
    def preforward_callback(self):
        pass
    
    def post_backward_callback(self):
        pass
    
    def post_epoch_callback(self):
        pass
    
    def save_evals(self, preds, results, directory = None):
        """Creates a checkpoint of the optimizer, scheduler and model"""
        """Creates checkpoint of optimizer, scheduler and a model"""
        base_ckpt_path = directory or self.args.save
        save_dir = os.path.join(base_ckpt_path, "eval", f"{self.global_steps}")
        os.makedirs(save_dir, exist_ok=True)
        
        if get_rank() == 0:
            torch.save(preds, os.path.join(save_dir, "preds.pt"))
            torch.save(results, os.path.join(save_dir, "results.pt"))

    def save(self, directory, global_steps=None):
        global_steps = global_steps if global_steps is not None else self.global_steps
        """Creates a checkpoint of the optimizer, scheduler and model"""
        """Creates checkpoint of optimizer, scheduler and a model"""
        base_ckpt_path = directory or self.args.save
        ckpt_dir = os.path.join(base_ckpt_path, f"{global_steps}")
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.args.model_parallel:
            if get_rank() == 0:
                self.model.module.config.to_json_file(os.path.join(ckpt_dir, "config.json"))
                self.tokenizer.save_pretrained(ckpt_dir)
            if self.dp_rank == 0:
                save_parallel(self.model.module.base_model, ckpt_dir)
        else:
            if get_rank() == 0:
                print(f"Model save to {ckpt_dir}")
                self.tokenizer.save_pretrained(ckpt_dir)

            if self.args.save_all:
                self.model.save_checkpoint(base_ckpt_path, tag=f"{global_steps}")
                rng_states = {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state(),
                    "numpy": np.random.get_state(),
                    "python": random.getstate(),
                }
                torch.save(rng_states, os.path.join(ckpt_dir, f"rng_states_{get_rank()}.pt"))
                if get_rank() == 0:
                    with open(os.path.join(ckpt_dir, "dynamics.json"), "w") as f:
                        json.dump({
                            "step": self.steps,
                            "epoch": self.epoch,
                            "global_steps": global_steps,
                            "skip_offset": (self.epoch, global_steps * self.total_batch_size)
                        }, f)
            if get_rank() == 0:
                self.model.module.save_pretrained(ckpt_dir, safe_serialization=False)
        
        dist.barrier()
