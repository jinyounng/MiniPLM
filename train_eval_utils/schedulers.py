from torch.optim.lr_scheduler import CosineAnnealingLR
import math


class WarmupCosineAnnealingLR(CosineAnnealingLR):
    def __init__(self, optimizer, T_max, warmup_steps, eta_min=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, T_max, eta_min, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [(base_lr * self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            return [
                self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / (self.T_max - self.warmup_steps)))
                for base_lr in self.base_lrs
            ]
    
    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["warmup_steps"] = self.warmup_steps
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.warmup_steps = state_dict.pop("warmup_steps")
        super().load_state_dict(state_dict)


class SimWarmupLinearScheduler():
    # output warmup linear learning rate
    def __init__(self, num_warmup_steps, num_training_steps, lr):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.lr = lr
        self.step = 0

    def get_lr(self, t):
        if t < self.num_warmup_steps:
            return self.lr * float(t) / float(max(1, self.num_warmup_steps))
        return self.lr * max(0.0, float(self.num_training_steps - t) / float(max(1, self.num_training_steps - self.num_warmup_steps)))
