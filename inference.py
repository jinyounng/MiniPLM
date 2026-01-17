import time
import os

import torch
import torch.distributed as dist

import json
from arguments import get_args

from utils import print_args, initialize
from utils import save_rank

from pretrain.inferer import PretrainLMInferer, PretrainGenInferer


torch.set_num_threads(16)


def grouped_infer(args, ds_config, device, inferer_cls, start, end, time_stamp):
    base_ckpt_path = args.model_path
    ckpt_paths = [os.path.join(
        base_ckpt_path, f"{s}") for s in range(start, end, 5000)]
    
    args.model_path = ckpt_paths[0]
    inferer = None
    base_save_path = args.save
    for i, ckpt_path in enumerate(ckpt_paths):
        args.model_path = ckpt_path
        args.save = os.path.join(base_save_path, os.path.basename(ckpt_path))
        os.makedirs(args.save, exist_ok=True)
        save_rank(time_stamp, os.path.join(args.save, "log.txt"))

        if i == 0:
            inferer = inferer_cls(args, ds_config, device)
        else:
            inferer.setup_model_and_optimizer(args, set_optim=False)
            
        inferer.inference()


def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    args.time_stamp = cur_time
    time_stamp = "\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30
    save_rank(time_stamp, os.path.join(args.save, "log.txt"))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["zero_optimization"]["stage"] = 0
    
    args.deepspeed_config = None
        
    if args.type == "pt_lm_infer":
        inferer_cls = PretrainLMInferer
    elif args.type == "pt_gen_infer":
        inferer_cls = PretrainGenInferer
    else:
        raise ValueError(f"Invalid type: {args.type}")     
    
    if args.grouped_infer:
        grouped_infer(args, ds_config, device, inferer_cls, args.ckpt_start, args.ckpt_end, time_stamp)
    else:
        inferer = inferer_cls(args, ds_config, device)
        inferer.inference()

    
if __name__ == "__main__":
    main()
