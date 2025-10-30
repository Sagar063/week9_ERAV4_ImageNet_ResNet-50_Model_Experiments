
# utils/dist_utils.py
import os
import torch.distributed as dist
import torch

def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def get_rank():
    return dist.get_rank() if is_dist() else 0

def is_main_process():
    return get_rank() == 0

def setup_ddp(backend="nccl"):
    # initialize from torchrun env if present
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

def cleanup_ddp():
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()
