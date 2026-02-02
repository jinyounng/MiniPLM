"""
Offline Knowledge Distillation Trainer

Cached sparse teacher logits를 사용한 Knowledge Distillation
Teacher 모델 forward 없이 미리 계산된 logits 사용
"""

import os
import torch
import torch.distributed as dist
from torch.distributed import get_rank
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler

from utils import print_rank, save_rank
from pretrain.trainer import PreTrainer
from data_utils.sparse_kd_datasets_hdf5 import SparseKDLMDatasetHDF5
from train_eval_utils.sparse_kd_loss import (
    compute_sparse_kd_loss,
    compute_sparse_kd_entropy
)


class OfflineKDPreTrainer(PreTrainer):
    """
    Offline Knowledge Distillation Trainer
    
    Cached sparse teacher logits를 사용하여 학습합니다.
    Teacher 모델 forward가 필요 없어서 빠르고 메모리 효율적입니다.
    """
    
    def __init__(self, args, ds_config, device, do_train=True):
        super().__init__(args, ds_config, device, do_train)
        
        # Cached logits 경로 확인
        assert hasattr(args, 'cached_logits_dir') and args.cached_logits_dir is not None, \
            "cached_logits_dir must be provided for offline_kd"
        
        # Vocabulary size 확인 (loss 계산에 필요)
        # DeepSpeed 모델에서는 .module로 원래 모델에 접근
        if hasattr(self.model, 'module'):
            self.vocab_size = self.model.module.config.vocab_size
        else:
            self.vocab_size = self.model.config.vocab_size
        
        # Temperature (기본 1.0)
        self.temperature = getattr(args, 'kd_temperature', 1.0)
        
        # Alpha: KD loss 가중치 (기본 0.5)
        self.alpha = getattr(args, 'alpha', 0.5)
        
        print_rank(f"✅ OfflineKD Trainer initialized")
        print_rank(f"   Cached logits dir: {args.cached_logits_dir}")
        print_rank(f"   Vocab size: {self.vocab_size}")
        print_rank(f"   Temperature: {self.temperature}")
        print_rank(f"   Alpha (KD weight): {self.alpha}")
    
    def set_datasets(self, args=None, do_train=True):
        """
        SparseKDLMDatasetHDF5만 사용 (HDF5 전용).
        cached_logits_dir에 data_*.h5 또는 shard_*.h5가 있어야 함.
        """
        args = args or self.args
        data_split = args.data_split or "data"

        if args.cached_logits_dir:
            import glob as glob_module
            h5_files = glob_module.glob(os.path.join(args.cached_logits_dir, 'data_*.h5'))
            if not h5_files:
                h5_files = glob_module.glob(os.path.join(args.cached_logits_dir, 'shard_*.h5'))
            if not h5_files:
                raise FileNotFoundError(
                    f"No HDF5 logits shards in {args.cached_logits_dir}. "
                    "Expected data_*.h5 or shard_*.h5"
                )
            print_rank("### Using HDF5 format (memory-efficient)")
        
        if do_train:
            print_rank("### Using data from directory: {}".format(args.data_dir))
            print_rank("### Using cached logits from: {}".format(args.cached_logits_dir))
            print_rank("### Dataset class: SparseKDLMDatasetHDF5")

            assert args.dev_data_dir is None or not os.path.samefile(args.dev_data_dir, args.data_dir)

            self.train_dataset = SparseKDLMDatasetHDF5(
                args, 
                self.tokenizer, 
                data_split, 
                args.data_dir, 
                args.train_num,
                cached_logits_dir=args.cached_logits_dir,
                min_state=self.args.min_state
            )
            print_rank("### Training Data Number: {}".format(len(self.train_dataset)))
            
            if self.args.do_valid and args.dev_data_dir is not None:
                self.eval_dataset = SparseKDLMDatasetHDF5(
                    args,
                    self.tokenizer,
                    data_split,
                    args.dev_data_dir,
                    args.dev_num,
                    cached_logits_dir=args.cached_logits_dir,
                    max_offset=100000
                )
                print_rank("### Dev Data Number: {}".format(len(self.eval_dataset)))
            else:
                self.eval_dataset = None
        else:
            self.eval_dataset = SparseKDLMDatasetHDF5(
                args,
                self.tokenizer,
                data_split,
                args.data_dir,
                args.dev_num,
                cached_logits_dir=args.cached_logits_dir,
                max_offset=100000
            )
    
    def _compute_sparse_kd_lm_loss(
        self, 
        model_batch, 
        no_model_batch, 
        mean=True, 
        output_all_losses=False
    ):
        """
        Sparse KD Loss 계산
        
        Args:
            model_batch: 모델 입력
            no_model_batch: Sparse logits 포함
            mean: 평균 여부
            output_all_losses: 모든 loss 출력 여부
        
        Returns:
            dict with loss, lm_loss, kd_loss, kd_entropy
        """
        # Student forward
        logits = self.model(**model_batch, use_cache=False).logits
        
        # LM loss
        lm_loss = self._get_lm_loss_from_logits(
            logits, 
            no_model_batch["label"], 
            no_model_batch["loss_mask"]
        )
        
        # Sparse KD loss
        sparse_logits = no_model_batch.get('sparse_logits', None)
        
        if sparse_logits is not None:
            kd_loss = compute_sparse_kd_loss(
                student_logits=logits,
                sparse_teacher_logits=sparse_logits,
                loss_mask=no_model_batch["loss_mask"],
                vocab_size=self.vocab_size,
                device=self.device,
                temperature=self.temperature
            )
            
            kd_entropy = compute_sparse_kd_entropy(
                sparse_teacher_logits=sparse_logits,
                loss_mask=no_model_batch["loss_mask"],
                vocab_size=self.vocab_size,
                device=self.device,
                temperature=self.temperature
            )
        else:
            # Sparse logits가 없는 경우 (fallback)
            print_rank("⚠️ No sparse logits found, using LM loss only")
            kd_loss = torch.zeros_like(lm_loss)
            kd_entropy = torch.zeros_like(lm_loss)
        
        # Combined loss: loss = (1 - alpha) * lm_loss + alpha * kd_loss
        # alpha: KD loss 가중치 (0~1, 클수록 KD에 더 가중치)
        loss = (1 - self.alpha) * lm_loss + self.alpha * kd_loss
        
        if mean:
            loss = loss.mean()
            lm_loss = lm_loss.mean()
            kd_loss = kd_loss.mean()
            kd_entropy = kd_entropy.mean()
        
        outputs = {
            "loss": loss,
            "lm_loss": lm_loss,
            "kd_loss": kd_loss,
            "kd_entropy": kd_entropy,
        }
        
        if output_all_losses:
            # Teacher loss는 계산 불가 (logits만 있음)
            outputs["teacher_loss"] = torch.tensor(0.0, device=self.device)
            if mean:
                outputs["teacher_loss"] = outputs["teacher_loss"].mean()
        
        return outputs
    
    def compute_loss(self, model_batch, no_model_batch):
        """
        Loss 계산 및 distributed reduction
        """
        out = self._compute_sparse_kd_lm_loss(model_batch, no_model_batch)
        loss, lm_loss, kd_loss, kd_entropy = \
            out["loss"], out["lm_loss"], out["kd_loss"], out["kd_entropy"]
        
        # Distributed reduction
        dist.all_reduce(lm_loss, group=self.dp_group, op=dist.ReduceOp.SUM)
        lm_loss = lm_loss / self.dp_world_size
        dist.all_reduce(kd_loss, group=self.dp_group, op=dist.ReduceOp.SUM)
        kd_loss = kd_loss / self.dp_world_size
        dist.all_reduce(kd_entropy, group=self.dp_group, op=dist.ReduceOp.SUM)
        kd_entropy = kd_entropy / self.dp_world_size
        
        other_outputs = {
            "lm_loss": lm_loss.item(),
            "kd_loss": kd_loss.item(),
            "kd_entropy": kd_entropy.item()
        }
        
        return loss, other_outputs
    
    def evaluate(self):
        """
        Evaluation with sparse KD loss
        """
        eval_sampler = DistributedSampler(
            self.eval_dataset, 
            shuffle=False, 
            drop_last=False, 
            rank=self.dp_rank, 
            num_replicas=self.dp_world_size
        )
        eval_dataloader = DataLoader(
            self.eval_dataset, 
            sampler=eval_sampler, 
            batch_size=self.args.eval_batch_size, 
            num_workers=self.args.num_workers, 
            collate_fn=self.eval_dataset.collate
        )
        
        self.model.eval()
        all_losses, all_lm_losses, all_kd_losses, all_kd_entropy = [], [], [], []
        
        with torch.no_grad():
            for i, (model_batch, no_model_batch) in tqdm(
                enumerate(eval_dataloader), 
                f"LM Evaluation", 
                disable=(not get_rank() == 0)
            ):
                if i % 10 == 0:
                    print_rank(f"evaluating batch {i}/{len(eval_dataloader)}")
                
                self.eval_dataset.move_to_device(model_batch, no_model_batch, self.device)
                
                out = self._compute_sparse_kd_lm_loss(
                    model_batch, 
                    no_model_batch, 
                    mean=False, 
                    output_all_losses=False
                )
                
                loss, lm_loss, kd_loss, kd_entropy = \
                    out["loss"], out["lm_loss"], out["kd_loss"], out["kd_entropy"]
                
                all_losses.append(loss)
                all_lm_losses.append(lm_loss)
                all_kd_losses.append(kd_loss)
                all_kd_entropy.append(kd_entropy)
        
        # Aggregate losses
        all_losses = torch.cat(all_losses, dim=0)
        avg_loss = self._avg_loss_cross_dp(all_losses)
        
        all_lm_losses = torch.cat(all_lm_losses, dim=0)
        avg_lm_loss = self._avg_loss_cross_dp(all_lm_losses)
        
        all_kd_losses = torch.cat(all_kd_losses, dim=0)
        avg_kd_loss = self._avg_loss_cross_dp(all_kd_losses)
        
        all_kd_entropy = torch.cat(all_kd_entropy, dim=0)
        avg_kd_entropy = self._avg_loss_cross_dp(all_kd_entropy)
        
        if get_rank() == 0:
            res = {
                "avg_loss": avg_loss,
                "avg_lm_loss": avg_lm_loss,
                "avg_kd_loss": avg_kd_loss,
                "avg_kd_entropy": avg_kd_entropy,
            }
            eval_log_str = self.get_log(res, "eval")
            print_rank(eval_log_str)
            save_rank(eval_log_str, os.path.join(self.args.save, "log.txt"))
            print_rank("*" * 100)
        else:
            res = None
        
        dist.barrier()
        return res

