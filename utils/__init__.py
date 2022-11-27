import yaml
from torch_lib.util import Count
import os
import torch
import subprocess
import torch.distributed as dist
import argparse


def parse_yaml(path):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


default_config_path = './checkpoint/load.yml'


def ToCuda(item):
    return item.cuda()
    # return item.cpu()


class QuestionIdGen:
    q_id = Count()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--use_slurm', type=str2bool, default=False)
    parser.add_argument('--master_address', type=str, default=None)
    parser.add_argument('--master_port', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--gen_selected_imgs', action='store_true')
    return parser.parse_args()


def setup_slurm(backend='nccl', address=None, port=None, verbose=False):
    '''Initialize slurm distributed training environment.
    '''
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    # specify master address
    if address is not None:
        os.environ['MASTER_ADDR'] = address
    elif 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)

    # The following code is the same as the setup_DDP() code in single-machine-and-multi-GPU-DistributedDataParallel-launch.py
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # If the OS is Windows or macOS, use gloo instead of nccl
    dist.init_process_group(backend=backend)
    # set distributed device
    device = torch.device('cuda:{}'.format(local_rank))
    if verbose:
        print('Using device: {}'.format(device))
        print(f'local rank: {local_rank}, global rank: {rank}, world size: {world_size}')
    return rank, local_rank, world_size, device


def set_DDP(args):
    # parallel
    if args.use_slurm is False:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
    else:
        print('use slurm to run distributed pytorch program.')
        _, rank, _, _ = setup_slurm(address=args.master_address, port=args.master_port)
    
    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def str2bool(b):
    if b.lower() in ["false"]:
        return False
    elif b.lower() in ["true"]:
        return True
    elif b is None:
        return None
    else:
        raise Exception("Invalid Bool Value")
