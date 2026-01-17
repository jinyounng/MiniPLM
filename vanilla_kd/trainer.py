import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import get_rank
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, DistributedSampler
from utils import print_rank, get_model, save_rank

from pretrain.trainer import PreTrainer


class VanillaKDPreTrainer(PreTrainer):
    def __init__(self, args, ds_config, device, do_train=True):
        super().__init__(args, ds_config, device, do_train)
        self.setup_teacher_model()
        
    def setup_teacher_model(self, args=None, device=None):
        args = args or self.args
        device = device or self.device
        assert args.teacher_model_path is not None or args.teacher_peft_path is not None
        teacher_model = get_model(args, device,
                                  model_path=args.teacher_model_path,
                                  from_scratch=False,
                                  peft=args.teacher_peft,
                                  peft_path=args.teacher_peft_path)
        teacher_model.eval()
        self.teacher_model = teacher_model

    def _get_kd_loss(self, logits, teacher_logits, loss_mask):
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(logits)
        logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1)
        kd_loss = -torch.sum(x * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
        return kd_loss

    def _compute_kd_lm_loss(self, model_batch, no_model_batch, mean=True, output_all_losses=False):        
        with torch.no_grad():
            teacher_logits = self.teacher_model(**model_batch, use_cache=False).logits
        
        logits = self.model(**model_batch, use_cache=False).logits

        # lm loss
        lm_loss = self._get_lm_loss_from_logits(logits, no_model_batch["label"], no_model_batch["loss_mask"])
        
        # kd loss
        kd_loss = self._get_kd_loss(logits, teacher_logits, no_model_batch["loss_mask"])
        kd_entropy = self._get_kd_loss(teacher_logits, teacher_logits, no_model_batch["loss_mask"])
        
        # loss
        loss = (1-self.args.kd_ratio) * lm_loss + self.args.kd_ratio * kd_loss
        
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
            teacher_loss = self._get_lm_loss_from_logits(teacher_logits, no_model_batch["label"], no_model_batch["loss_mask"])

            if mean:
                teacher_loss = teacher_loss.mean()

            outputs.update({
                "teacher_loss": teacher_loss,
            })
            
        return outputs
    
    def compute_loss(self, model_batch, no_model_batch):
        out = self._compute_kd_lm_loss(model_batch, no_model_batch)
        loss, lm_loss, kd_loss, kd_entropy = \
            out["loss"], out["lm_loss"], out["kd_loss"], out["kd_entropy"]
        
        dist.all_reduce(lm_loss, group=self.dp_group, op=dist.ReduceOp.SUM)
        lm_loss = lm_loss / self.dp_world_size
        dist.all_reduce(kd_loss, group=self.dp_group, op=dist.ReduceOp.SUM)
        kd_loss = kd_loss / self.dp_world_size
        dist.all_reduce(kd_entropy, group=self.dp_group, op=dist.ReduceOp.SUM)
        kd_entropy = kd_entropy / self.dp_world_size
        other_outputs = {"lm_loss": lm_loss.item(), "kd_loss": kd_loss.item(), "kd_entropy": kd_entropy.item()}
        
        return loss, other_outputs
    
    def evaluate(self):
        eval_sampler = DistributedSampler(self.eval_dataset, shuffle=False, drop_last=False, rank=self.dp_rank, num_replicas=self.dp_world_size)
        eval_dataloader = DataLoader(
            self.eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=self.eval_dataset.collate)
        
        self.model.eval()
        all_losses, all_lm_losses, all_kd_losses, all_kd_entropy = [], [], [], []
        all_teacher_losses = []
                    
        with torch.no_grad():
            for i, (model_batch, no_model_batch) in tqdm(enumerate(eval_dataloader), f"LM Evaluation", disable=(not get_rank() == 0)):
                if i % 10 == 0:
                    print_rank(f"evaluating batch {i}/{len(eval_dataloader)}")
                self.eval_dataset.move_to_device(model_batch, no_model_batch, self.device)
                out = self._compute_kd_lm_loss(
                    model_batch, no_model_batch, mean=False, output_all_losses=True)
                loss, lm_loss, kd_loss, kd_entropy, teacher_loss = \
                    out["loss"], out["lm_loss"], out["kd_loss"], out["kd_entropy"], out["teacher_loss"]
                all_losses.append(loss)
                all_lm_losses.append(lm_loss)
                all_kd_losses.append(kd_loss)
                all_kd_entropy.append(kd_entropy)
                all_teacher_losses.append(teacher_loss)
        
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
            res = {"avg_loss": avg_loss,
                   "avg_lm_loss": avg_lm_loss,
                   "avg_kd_loss": avg_kd_loss,
                   "avg_kd_entropy": avg_kd_entropy,
                   "avg_teacher_loss": avg_teacher_loss,}
            eval_log_str = self.get_log(res, "eval")
            print_rank(eval_log_str)
            save_rank(eval_log_str, os.path.join(self.args.save, "log.txt"))
            print_rank("*" * 100)
        else:
            res = None
        
        dist.barrier()