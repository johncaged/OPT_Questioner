from torch_lib import Proxy
from model.model import BaseQuestioner, QuestionerWithCaption
import os
from data.dataset import Tokenizer, build_dataloader, build_coco_dataset
from utils.loss import Loss, MixLoss
from utils import ToCuda, parse_yaml, default_config_path
from torch.optim import Adam
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from apex import amp
from torch_lib.log.directory import set_base_path, set_namespace
import argparse
from utils.callbacks import MyCallback
from utils.handlers import set_handler
from torch.optim.lr_scheduler import LambdaLR
import warnings
from torch_lib.util import NOTHING
import subprocess

warnings.filterwarnings('ignore')

config = parse_yaml(default_config_path)
set_base_path(config['base_path'])
set_namespace(config['namespace'])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--use_slurm', type=str2bool, default=False)
    parser.add_argument('--master_address', type=str, default=None)
    parser.add_argument('--master_port', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    return parser.parse_args()


def setup_DDP(backend='nccl', address=None, port=None, verbose=False):
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


def main():
    args = get_args()
    
    # parallel
    if args.use_slurm is False:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
    else:
        print('use slurm to run distributed pytorch program.')
        _, rank, _, _ = setup_DDP(address=args.master_address, port=args.master_port)

    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # tokenizer
    tokenizer = Tokenizer()
    # build and load model
    model = BaseQuestioner(tokenizer, QuestionerWithCaption())
    model.load_pretrained_weights()
    # optimizer
    optimizer = Adam(model.parameters(), lr=1e-5)
    
    model, optimizer = amp.initialize(ToCuda(model), optimizer, enabled=args.fp16, opt_level='O2')
    model = DDP(model)
    
    # resume
    # checkpoint = torch.load('./log/answer_type_224_512/checkpoint/checkpoint_19.pth', map_location='cpu')
    # model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # del checkpoint
    # build dataset
    # train_dataset = NOTHING
    train_data = build_coco_dataset('caption', 'train', tokenizer, text_encoder=model.module.clip)
    train2_data = build_coco_dataset('caption', 'train2', tokenizer, text_encoder=model.module.clip)
    train_dataset, train_sampler = build_dataloader(train_data + train2_data, 512)
    del train_data, train2_data
    val_dataset, _ = build_dataloader(build_coco_dataset('caption', 'val', tokenizer, text_encoder=model.module.clip), 512)
    # torch-lib pipeline
    proxy = Proxy(model)
    proxy.custom.train_sampler = train_sampler
    set_handler(proxy)
    proxy.count_params('M')
    total_epoch = 50
    proxy.build(
        loss=Loss(label_smoothing=0.0),
        optimizer=optimizer,
        lr_decay=LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: (1 - epoch / total_epoch) ** 0.9)
    )
    proxy.train(
        train_dataset,
        total_epoch,
        val_dataset,
        callbacks=MyCallback()
    )


def str2bool(b):
    if b.lower() in ["false"]:
        return False
    elif b.lower() in ["true"]:
        return True
    elif b is None:
        return None
    else:
        raise Exception("Invalid Bool Value")


if __name__ == '__main__':
    main()
