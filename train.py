from torch_lib import Proxy
from model.model import BaseQuestioner, QuestionerWithCaption, VGAdapter
import os
from data.dataset import Tokenizer, build_dataloader, build_coco_dataset, build_vg_dataset
from utils.loss import Loss, MixLoss
from utils import ToCuda, parse_yaml, default_config_path
from torch.optim import Adam
from utils.metrics import MultiTaskMetric
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from apex import amp
from torch_lib.log.directory import set_base_path, set_namespace
from utils.callbacks import MyCallback
from utils.handlers import set_handler
from torch.optim.lr_scheduler import LambdaLR
import warnings
from torch_lib.util import NOTHING
from utils import get_args, set_DDP

warnings.filterwarnings('ignore')

config = parse_yaml(default_config_path)
set_base_path(config['base_path'])
set_namespace(config['namespace'])


def main():
    args = get_args()
    set_DDP(args)

    # tokenizer
    tokenizer = Tokenizer()
    # build and load model
    # model = BaseQuestioner(tokenizer, QuestionerWithCaption())
    model = BaseQuestioner(tokenizer, VGAdapter())
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
    
    # train_data = build_coco_dataset('caption', 'train', tokenizer, text_encoder=model.module.clip)
    # train2_data = build_coco_dataset('caption', 'train2', tokenizer, text_encoder=model.module.clip)
    # train_dataset, train_sampler = build_dataloader(train_data + train2_data, 512)
    # del train_data, train2_data
    # val_dataset, _ = build_dataloader(build_coco_dataset('caption', 'val', tokenizer, text_encoder=model.module.clip), 512)
    
    train_dataset, train_sampler = build_dataloader(build_vg_dataset('train', tokenizer), 512)
    val_dataset, _ = build_dataloader(build_vg_dataset('val', tokenizer), 512)
    # torch-lib pipeline
    proxy = Proxy(model)
    proxy.custom.train_sampler = train_sampler
    set_handler(proxy)
    proxy.count_params('M')
    total_epoch = 50
    proxy.build(
        loss=Loss(label_smoothing=0.0),
        optimizer=optimizer,
        lr_decay=LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: (1 - epoch / total_epoch) ** 0.9),
        metrics=MultiTaskMetric()
    )
    proxy.train(
        train_dataset,
        total_epoch,
        val_dataset,
        callbacks=MyCallback()
    )


if __name__ == '__main__':
    main()
