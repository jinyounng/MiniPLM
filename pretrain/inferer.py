import os
from tqdm import tqdm
from time import time
import json
import numpy as np
import random

import torch
import torch.distributed as dist

from utils import get_rank, print_rank, all_gather, print_and_save_rank
from train_eval_utils.base_trainer import BaseTrainer
from data_utils.lm_datasets import LMDataset
from torch.utils.data import DataLoader, DistributedSampler
from data_utils import ChunkedDatasetBuilder, best_fitting_dtype


class PretrainInferer(BaseTrainer):
    def __init__(self, args, ds_config, device):
        super().__init__(args, ds_config, device, do_train=False)
        self.min_offset = 0 # virtual offset
        self.min_shard_idx = 0
        self.set_datasets(do_train=False)
        self.setup_model_and_optimizer(set_optim=False)

    def set_datasets(self, args=None, do_train=True):
        args = args or self.args
        if os.path.exists(os.path.join(args.save, "state.json")):
            with open(os.path.join(args.save, "state.json")) as f:
                state = json.load(f)
            self.min_offset = state["offset"]
            self.min_shard_idx = state["idx"]
            print_and_save_rank("Found state {}".format(state), os.path.join(args.save, "log.txt"))
        else:
            self.min_offset = 0
            self.min_shard_idx = 0
        
        load_num = args.infer_num-self.min_offset if args.infer_num is not None else None
        self.eval_dataset = LMDataset(args, self.tokenizer, args.data_split, args.data_dir, load_num, min_offset=(args.min_offset+self.min_offset),
                                      min_state=self.args.shard_start, max_state=self.args.shard_end)
        self.infer_num = args.infer_num if args.infer_num is not None else (len(self.eval_dataset) + self.min_offset)
        print_and_save_rank("Inference dataset size: {}".format(len(self.eval_dataset)), os.path.join(args.save, "log.txt"))

    def get_dataloader(self, eval_dataset: LMDataset):
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False, drop_last=False, rank=self.dp_rank, num_replicas=self.dp_world_size)
        eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=eval_dataset.collate)
        return eval_dataloader

    def inference(self):
        self._inference_large()
 
    def infer_one_batch(self, model_batch, no_model_batch):
        raise NotImplementedError

    def gather_infer(self, all_infer_output):
        raise NotImplementedError

    def save_infer(self, all_infer_output, infer_stat, save_path, save_idx=None):
        raise NotImplementedError

    def _inference_base(self):
        eval_dataset = self.eval_dataset
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False, drop_last=False, rank=self.dp_rank, num_replicas=self.dp_world_size)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=eval_dataset.collate)
        
        self.model.eval()
        all_infer_output = []
        
        st = time()         
        with torch.no_grad():
            for i, (model_batch, no_model_batch) in enumerate(tqdm(eval_dataloader, f"LM Evaluation", disable=(not get_rank() == 0))):
                if i == 0 and self.dp_rank == 0:
                    self.first_print(model_batch, no_model_batch)
                eval_dataset.move_to_device(model_batch, no_model_batch, self.device)
                infer_out = self.infer_one_batch(model_batch, no_model_batch)
                all_infer_output.append(infer_out)
        
        all_infer_output = self.gather_infer(all_infer_output)
        all_infer_output = all_infer_output[:len(eval_dataset)]

        if get_rank() == 0:
            infer_stat = {"num": len(all_infer_output), "time": time() - st}
            self.save_infer(all_infer_output, infer_stat, self.args.save)            
        else:
            res = None
        
        dist.barrier()
        return res

    def _inference_large(self):
        save_path = self.args.save
        eval_dataset = self.eval_dataset
        eval_dataloader = self.get_dataloader(eval_dataset)

        if dist.get_rank() == 0:
            check_indices = []
            tot = 64
            print("Creating save indices")
            while len(check_indices) < tot:
                x = random.randint(0, len(self.eval_dataset)-1)
                if x not in check_indices:
                    check_indices.append(x)
            check_indices = sorted(check_indices)
            torch.save(check_indices, os.path.join(save_path, f"check_indices_{self.min_offset}.pt"))
            check_insts = []
            for n, cidx in enumerate(tqdm(check_indices)):
                print(f"{n}/{tot}")
                check_insts.append(self.eval_dataset[cidx][1].astype(int))
            torch.save(check_insts, os.path.join(save_path, f"check_insts_{self.min_offset}.pt"))
            print("Save indices create end")
            
        dist.barrier()
        self.model.eval()
        all_infer_output = []
                   
        global_batch_size = self.args.eval_batch_size * self.dp_world_size
        num_per_shard = self.args.save_interval * global_batch_size

        log_str = f"Start from min_shard_idx: {self.min_shard_idx}, min_offset: {self.min_offset}, total infer_num: {self.infer_num}\n"
        log_str += f"Example num per shard: {num_per_shard}"
        print_and_save_rank(log_str, os.path.join(save_path, "log.txt"))
 
        st = time()
        idx = self.min_shard_idx
        offset = self.min_offset
        with torch.no_grad():
            for i, (model_batch, no_model_batch) in enumerate(tqdm(eval_dataloader, f"LM Evaluation", disable=(not get_rank() == 0))):
                if i == 0 and self.dp_rank == 0:
                    self.first_print(model_batch, no_model_batch)
                eval_dataset.move_to_device(model_batch, no_model_batch, self.device)
                infer_out = self.infer_one_batch(model_batch, no_model_batch)
                all_infer_output.append(infer_out)
                ct = time() - st
                if i % self.args.log_interval == 0:
                    print_and_save_rank("Processing {}. {}/{}. {}/{}. Spent time: {:.2f}".format(
                        idx,
                        i,
                        len(eval_dataloader),
                        self.min_offset + i*global_batch_size,
                        self.infer_num,
                        ct
                    ), os.path.join(save_path, "log.txt"))

                dist.barrier()
                if self.args.do_infer and (i+1) % self.args.save_interval == 0:
                    all_infer_output = self.gather_infer(all_infer_output)
                    print_rank(all_infer_output.size())
                    all_infer_output = all_infer_output[:len(self.eval_dataset)-(idx-self.min_shard_idx)*num_per_shard]
                    print_rank(all_infer_output.size())

                    if self.dp_rank == 0:
                        infer_stat = {"num": len(all_infer_output), "time": ct}
                        self.save_infer(all_infer_output, infer_stat, save_path, idx)
                        state = {
                            "idx": idx+1, # next run start from this index
                            "offset": offset + len(all_infer_output)
                        }
                        with open(os.path.join(save_path, "state.json"), "w") as f:
                            json.dump(state, f)
                    dist.barrier()
                    idx += 1
                    offset += len(all_infer_output)
                    all_infer_output = []

        if len(all_infer_output) > 0:
            print_rank("last shard")
            all_infer_output = self.gather_infer(all_infer_output)
            all_infer_output = all_infer_output[:len(self.eval_dataset)-(idx-self.min_shard_idx)*num_per_shard]
            if self.dp_rank == 0:
                infer_stat = {"num": len(all_infer_output), "time": ct}
                self.save_infer(all_infer_output, infer_stat, save_path, idx)
                state = {
                    "idx": idx+1,
                    "offset": offset + len(all_infer_output)
                }
                with open(os.path.join(save_path, "state.json"), "w") as f:
                    json.dump(state, f)
        
        dist.barrier()
        return None
    
    
class PretrainLMInferer(PretrainInferer):
    def __init__(self, args, ds_config, device):
        super().__init__(args, ds_config, device)
    
    def infer_one_batch(self, model_batch, no_model_batch):
        loss = self.compute_lm_loss(model_batch, no_model_batch, mean=False)
        if self.args.torch_compile is not None:
            loss = loss.clone()
        return loss

    def gather_infer(self, all_infer_output):
        all_infer_output = torch.cat(all_infer_output, dim=0)
        all_infer_output = all_gather(all_infer_output, dim=1, op="stack").view(-1)
        return all_infer_output

    def save_infer(self, all_infer_output, infer_stat, save_path, save_idx=None):
        res = {
            "avg_loss": torch.mean(all_infer_output),
            **infer_stat
        }
        eval_log_str = self.get_log(res, "infer")
        print_and_save_rank(eval_log_str, os.path.join(save_path, "log.txt"))
        scores_path = os.path.join(save_path, f"scores_{save_idx}.pt") if save_idx is not None else os.path.join(save_path, "scores.pt")
        torch.save(all_infer_output, scores_path)
        print("Inference Saved to", scores_path)
        
        
class PretrainGenInferer(PretrainInferer):
    def __init__(self, args, ds_config, device):
        super().__init__(args, ds_config, device)
    
    def get_dataloader(self, eval_dataset: LMDataset):
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False, drop_last=False, rank=self.dp_rank, num_replicas=self.dp_world_size)
        eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=eval_dataset.collate_gen)
        return eval_dataloader
    
    def infer_one_batch(self, model_batch, no_model_batch):
        gen_out = self.generate(model_batch)
        tot_ids = gen_out["sequences"]
        tot_ids = torch.nn.functional.pad(tot_ids, (0, self.args.max_length-tot_ids.size(1)), value=self.tokenizer.pad_token_id)
        return tot_ids

    def gather_infer(self, all_infer_output):
        all_infer_output = torch.cat(all_infer_output, dim=0)
        all_infer_output = all_gather(all_infer_output, dim=1, op="stack").view(-1, all_infer_output.size(-1))
        return all_infer_output

    def _trim_padding(self, tot_ids):
        trimmed_ids = []
        for ids in tot_ids:
            
            e = len(ids)-1
            while e >= 0 and ids[e] == self.tokenizer.pad_token_id:
                e -= 1
            if e < 0:
                trimmed_ids.append(ids)
            else:
                trimmed_ids.append(ids[:e+2]) # include the last pad token (= eos token)
            
        return trimmed_ids

    def save_infer(self, all_infer_output, infer_stat, save_path, save_idx=None):
        all_infer_output = all_infer_output.cpu().numpy()
        all_infer_output = self._trim_padding(all_infer_output)
        max_length = max([len(x) for x in all_infer_output])
        min_length = min([len(x) for x in all_infer_output])
        mean_length = np.mean([len(x) for x in all_infer_output])
        
        res = {
            "max_length": max_length,
            "min_length": min_length,
            "mean_length": mean_length,
            **infer_stat
        }
        eval_log_str = self.get_log(res, "infer")
        print_and_save_rank(eval_log_str, os.path.join(save_path, "log.txt"))

        dtype = best_fitting_dtype(self.tokenizer.vocab_size)
        builder = ChunkedDatasetBuilder(
            self.args.base_path, save_path, dtype, "infer", do_shuffle=False, output_state_start=save_idx)
        for inst in all_infer_output:
            builder.add_np_item(inst)
        builder.finalize()
        print("Inference Saved to", save_path)