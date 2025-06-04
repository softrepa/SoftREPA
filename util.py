import random

import numpy as np
import torch

import os
import sys
import torch
import torch.distributed as dist

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = rank  # Now matches the index inside CUDA_VISIBLE_DEVICES
    torch.cuda.set_device(local_rank)  # Set correct GPU
    setup_for_distributed(local_rank == 0)
    return local_rank, dist.get_world_size()

def cleanup():
    dist.destroy_process_group()

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def gather_from_all_gpus(data):
    if not is_dist_avail_and_initialized():
        return data
    world_size = dist.get_world_size()
    if world_size == 1:
        return data
    data_list = [torch.zeros_like(data) for _ in range(world_size)]
    dist.all_gather(data_list, data.contiguous())
    data_gathered = torch.cat(data_list, dim=0)
    return data_gathered