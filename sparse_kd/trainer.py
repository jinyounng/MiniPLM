"""
Sparse Online Knowledge Distillation Trainer

Online으로 Teacher forward를 수행하되, 
Top-K 또는 Random Sampling 방식으로 sparse하게 KD loss 계산.

장점:
- Cached logits 저장 불필요 (디스크 공간 절약)
- Top-K: 빠르고 간단, 중요한 토큰에 집중
- Random Sampling: Unbiased estimator, 전체 분포 추정

Usage:
    --kd-method topk        : Top-K tokens만 사용
    --kd-method sparse      : Random Sampling 사용
    --topk 50               : Top-K의 K 값
    --num-samples 50        : Random Sampling의 N 값
"""

import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import get_rank
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler

from utils import print_rank, get_model, save_rank
from pretrain.trainer import PreTrainer


class SparseKDPreTrainer(PreTrainer):
    """
    Sparse Online KD Trainer
    
    Teacher forward를 online으로 수행하되,
    full vocabulary 대신 sparse 방식으로 KD loss 계산.
    """

    def __init__(self, args, ds_config, device, do_train=True):
        super().__init__(args, ds_config, device, do_train)
        self.setup_teacher_model()
        # KD method 설정
        self.kd_method = getattr(args, 'kd_method', 'topk')
        self.topk = getattr(args, 'topk', 50)
        self.num_samples = getattr(args, 'num_samples', 50)
        self.kd_temperature = getattr(args, 'kd_temperature', 1.0)
        
        print_rank(f"✅ SparseKD Trainer initialized")
        print_rank(f"   KD Method: {self.kd_method}")
        if self.kd_method == 'topk':
            print_rank(f"   Top-K: {self.topk}")
        else:
            print_rank(f"   Num Samples: {self.num_samples}")
        print_rank(f"   Temperature: {self.kd_temperature}")
        print_rank(f"   KD Ratio: {args.kd_ratio}")
        
    def setup_teacher_model(self, args=None, device=None):
        """Teacher 모델 로드"""
        args = args or self.args
        device = device or self.device
        
        assert args.teacher_model_path is not None or args.teacher_peft_path is not None, \
            "teacher_model_path or teacher_peft_path must be provided"
        
        teacher_model = get_model(
            args, device,
            model_path=args.teacher_model_path,
            from_scratch=False,
            peft=args.teacher_peft,
            peft_path=args.teacher_peft_path
        )
        teacher_model.eval()
        self.teacher_model = teacher_model
        print_rank(f"✅ Teacher model loaded from {args.teacher_model_path}")

    def _get_topk_kd_loss(self, student_logits, teacher_logits, loss_mask, k=None):
        """
        Top-K Sparse KD Loss
        
        Teacher의 top-k 토큰에 대해서만 KD loss 계산.
        
        Args:
            student_logits: [batch, seq_len, vocab_size]
            teacher_logits: [batch, seq_len, vocab_size]
            loss_mask: [batch, seq_len]
            k: top-k 값 (None이면 self.topk 사용)
        
        Returns:
            kd_loss: [batch]
        """
        k = k or self.topk
        T = self.kd_temperature
        
        # Teacher top-k 추출
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        topk_probs, topk_indices = teacher_probs.topk(k, dim=-1)  # [batch, seq, k]
        
        # Top-k 확률 정규화 (합이 1이 되도록)
        topk_probs_normalized = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        # Student의 해당 위치 log-softmax
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        
        # Top-k 위치의 student log-probs 추출
        student_topk_log_probs = student_log_probs.gather(-1, topk_indices)
        
        # Cross entropy: -sum(p * log(q))
        kd_loss_per_position = -torch.sum(topk_probs_normalized * student_topk_log_probs, dim=-1)
        
        # loss_mask 적용
        kd_loss = torch.sum(kd_loss_per_position * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1).clamp(min=1)
        
        return kd_loss

    def _get_sparse_kd_loss(self, student_logits, teacher_logits, loss_mask, num_samples=None):
        """
        Random Sampling Sparse KD Loss (Unbiased Estimator)
        
        Teacher 확률분포에서 N번 샘플링하여 KD loss 추정.
        E[loss] = true KD loss (unbiased)
        
        Args:
            student_logits: [batch, seq_len, vocab_size]
            teacher_logits: [batch, seq_len, vocab_size]
            loss_mask: [batch, seq_len]
            num_samples: 샘플 수 (None이면 self.num_samples 사용)
        
        Returns:
            kd_loss: [batch]
        """
        num_samples = num_samples or self.num_samples
        T = self.kd_temperature
        
        batch_size, seq_len, vocab_size = student_logits.shape
        
        # Teacher 확률분포
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        
        # Student log-probs
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        
        # Reshape for multinomial: [batch * seq, vocab]
        teacher_probs_flat = teacher_probs.view(-1, vocab_size)
        student_log_probs_flat = student_log_probs.view(-1, vocab_size)
        
        # N번 샘플링 (replacement=True)
        sampled_indices = torch.multinomial(teacher_probs_flat, num_samples, replacement=True)
        
        # 샘플된 위치의 student log-probs
        sampled_student_log_probs = student_log_probs_flat.gather(-1, sampled_indices)
        
        # 샘플링 기반 cross-entropy 추정: -E[log q(x)] where x ~ p
        kd_loss_per_position = -sampled_student_log_probs.mean(dim=-1)
        
        # Reshape back
        kd_loss_per_position = kd_loss_per_position.view(batch_size, seq_len)
        
        # loss_mask 적용
        kd_loss = torch.sum(kd_loss_per_position * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1).clamp(min=1)
        
        return kd_loss

    def _get_kd_entropy(self, teacher_logits, loss_mask):
        """
        Teacher entropy 계산 (모니터링용)
        """
        T = self.kd_temperature
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits / T, dim=-1)
        
        # Entropy: -sum(p * log(p))
        entropy_per_position = -torch.sum(teacher_probs * teacher_log_probs, dim=-1)
        
        # loss_mask 적용
        entropy = torch.sum(entropy_per_position * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1).clamp(min=1)
        
        return entropy

    def _compute_sparse_kd_lm_loss(self, model_batch, no_model_batch, mean=True, output_all_losses=False):
        """
        Sparse KD + LM Loss 계산
        """
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_logits = self.teacher_model(**model_batch, use_cache=False).logits
        
        # Student forward
        logits = self.model(**model_batch, use_cache=False).logits
        
        # LM loss
        lm_loss = self._get_lm_loss_from_logits(
            logits, 
            no_model_batch["label"], 
            no_model_batch["loss_mask"]
        )
        
        # Sparse KD loss (method에 따라)
        if self.kd_method == 'topk':
            kd_loss = self._get_topk_kd_loss(
                logits, teacher_logits, 
                no_model_batch["loss_mask"],
                k=self.topk
            )
        else:  # sparse (random sampling)
            kd_loss = self._get_sparse_kd_loss(
                logits, teacher_logits, 
                no_model_batch["loss_mask"],
                num_samples=self.num_samples
            )
        
        # Teacher entropy (모니터링)
        kd_entropy = self._get_kd_entropy(teacher_logits, no_model_batch["loss_mask"])
        
        # Combined loss
        loss = (1 - self.args.kd_ratio) * lm_loss + self.args.kd_ratio * kd_loss
        
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
            teacher_loss = self._get_lm_loss_from_logits(
                teacher_logits, 
                no_model_batch["label"], 
                no_model_batch["loss_mask"]
            )
            if mean:
                teacher_loss = teacher_loss.mean()
            outputs["teacher_loss"] = teacher_loss
        
        return outputs
    
    def compute_loss(self, model_batch, no_model_batch):
        """Loss 계산 및 distributed reduction"""
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
        """Evaluation with sparse KD loss"""
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
        all_teacher_losses = []
        
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
                    output_all_losses=True
                )
                
                loss, lm_loss, kd_loss, kd_entropy, teacher_loss = \
                    out["loss"], out["lm_loss"], out["kd_loss"], out["kd_entropy"], out["teacher_loss"]
                
                all_losses.append(loss)
                all_lm_losses.append(lm_loss)
                all_kd_losses.append(kd_loss)
                all_kd_entropy.append(kd_entropy)
                all_teacher_losses.append(teacher_loss)
        
        # Aggregate
        all_losses = torch.cat(all_losses, dim=0)
        avg_loss = self._avg_loss_cross_dp(all_losses)
        
        all_lm_losses = torch.cat(all_lm_losses, dim=0)
        avg_lm_loss = self._avg_loss_cross_dp(all_lm_losses)
        
        all_kd_losses = torch.cat(all_kd_losses, dim=0)
        avg_kd_loss = self._avg_loss_cross_dp(all_kd_losses)
        
        all_kd_entropy = torch.cat(all_kd_entropy, dim=0)
        avg_kd_entropy = self._avg_loss_cross_dp(all_kd_entropy)
        
        all_teacher_losses = torch.cat(all_teacher_losses, dim=0)
        avg_teacher_loss = self._avg_loss_cross_dp(all_teacher_losses)
        
        if get_rank() == 0:
            res = {
                "avg_loss": avg_loss,
                "avg_lm_loss": avg_lm_loss,
                "avg_kd_loss": avg_kd_loss,
                "avg_kd_entropy": avg_kd_entropy,
                "avg_teacher_loss": avg_teacher_loss,
            }
            eval_log_str = self.get_log(res, "eval")
            print_rank(eval_log_str)
            save_rank(eval_log_str, os.path.join(self.args.save, "log.txt"))
            print_rank("*" * 100)
        else:
            res = None
        
        dist.barrier()
        return res
