from torch_lib import Proxy
from model.model import BaseQuestioner, QuestionerWithCaption
import os
from data.dataset import Tokenizer, build_dataloader, build_dataset
from utils.loss import Loss, MixLoss
from utils import parse_yaml, default_config_path
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

warnings.filterwarnings("ignore")

config = parse_yaml(default_config_path)
set_base_path(config['base_path'])
set_namespace(config['namespace'])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--fp16', action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    
    # parallel
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # tokenizer
    tokenizer = Tokenizer()
    # build and load model
    model = BaseQuestioner(tokenizer, QuestionerWithCaption())
    model.load_pretrained_weights()
    # optimizer
    optimizer = Adam(model.parameters(), lr=3e-6)
    
    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    model, optimizer = amp.initialize(model.to(device_id), optimizer, enabled=args.fp16, opt_level='O2')
    model = DDP(model)
    
    # resume
    # checkpoint = torch.load('./log/test/checkpoint/checkpoint.pt', map_location='cpu')
    # model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # del checkpoint
    # build dataset
    train_dataset = build_dataloader(build_dataset('caption', 'train', tokenizer, text_encoder=model.module.clip), 256)
    val_dataset = build_dataloader(build_dataset('caption', 'val', tokenizer, text_encoder=model.module.clip), 256)
    # torch-lib pipeline
    proxy = Proxy(model)
    set_handler(proxy)
    proxy.count_params('M')
    total_epoch = 20
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


if __name__ == '__main__':
    main()
