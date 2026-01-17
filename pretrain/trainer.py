import os
import wandb
from train_eval_utils.base_trainer import BaseTrainer
from utils import print_rank, print_and_save_rank
from torch.distributed import get_rank
from data_utils.lm_datasets import LMDataset


class PreTrainer(BaseTrainer):
    def __init__(self, args, ds_config, device, do_train=True):
        super().__init__(args, ds_config, device, do_train)
        self.set_datasets(do_train=do_train)
        if do_train:
            self.prepare_learning()
        self.setup_model_and_optimizer(set_optim=do_train)
        if do_train and self.args.resume_training:
            self.resume_training()
        elif args.start_from_global_step is not None:
            self.last_global_steps = self.args.start_from_global_step

    def set_datasets(self, args=None, do_train=True):
        args = args or self.args
        data_split = args.data_split or "data"
        if do_train:
            print_and_save_rank("### Using data from directory: {}".format(args.data_dir), os.path.join(args.save, "log.txt"))
            assert args.dev_data_dir is None or not os.path.samefile(args.dev_data_dir, args.data_dir)
            self.train_dataset = LMDataset(args, self.tokenizer, data_split, args.data_dir, args.train_num, min_state=self.args.min_state)
            print_and_save_rank("### Training Data Number:", len(self.train_dataset), os.path.join(args.save, "log.txt"))
            if self.args.do_valid and args.dev_data_dir is not None:
                self.eval_dataset = LMDataset(args, self.tokenizer, data_split, args.dev_data_dir, args.dev_num, max_offset=100000)
                print_and_save_rank("### Dev Data Number:", len(self.eval_dataset), os.path.join(args.save, "log.txt"))
            else:
                self.eval_dataset = None
        else:
            self.eval_dataset = LMDataset(args, self.tokenizer, data_split, args.data_dir, args.dev_num, max_offset=100000)
    
    def compute_loss(self, model_batch, no_model_batch):
        return self.compute_lm_loss(model_batch, no_model_batch), {}

    def evaluate(self):
        assert self.eval_dataset is not None
        lm_res = self.evaluate_lm()
        if get_rank() == 0:
            res = {
                "lm_loss": lm_res["avg_loss"],
            }
            
            if wandb.run is not None:                 
                wandb.log(res, step=self.global_steps)
            
            eval_log_str = self.get_log(res, "eval")
            print_and_save_rank(eval_log_str, os.path.join(self.args.save, "log.txt"))
            print_rank("*" * 100)
