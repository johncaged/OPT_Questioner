from torch_lib import Proxy
from model.model import BaseQuestioner, MultiStageAdapter
from utils import ToCuda
from data.dataset import build_vg_dataset, Tokenizer, build_dataloader
from utils.metrics import MultiTaskMetric
import torch
from utils.loss import Loss
from utils import set_DDP, get_args
from utils.handlers import set_handler
from apex.parallel import DistributedDataParallel as DDP
from utils.callbacks import EvalCallback


def main():
    args = get_args()
    set_DDP(args)

    # tokenizer
    tokenizer = Tokenizer()
    # build and load model
    model = BaseQuestioner(tokenizer, MultiStageAdapter(), use_img_region_embedding=True)
    model = DDP(model)
    
    # resume
    checkpoint = torch.load('./log//checkpoint/checkpoint_.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    del checkpoint
    # build dataset
    val_dataset, _ = build_dataloader(build_vg_dataset('val', tokenizer), 512)
    # torch-lib pipeline
    proxy = Proxy(model)
    set_handler(proxy)
    proxy.count_params('M')
    proxy.build(
        loss=Loss(label_smoothing=0.0),
        metrics=MultiTaskMetric()
    )
    
    proxy.custom.epoch_metrics = []
    proxy.custom.epoch_losses = []
    
    iteration = 10
    
    for _ in range(iteration):
        proxy.eval(
            val_dataset,
            callbacks=EvalCallback()
        )
    
    metrics = {}
    sum_loss = 0
    for item_metrics, item_loss in zip(proxy.custom.epoch_metrics, proxy.custom.epoch_losses):
        for key, value in item_metrics.items():
            metrics.setdefault(key, 0.0)
            metrics[key] = metrics[key] + float(value)
        sum_loss += float(item_loss)
    
    for key in metrics.keys():
        metrics[key] /= iteration
    sum_loss /= iteration
    print(metrics)
    print(sum_loss)


if __name__ == '__main__':
    main()
