from data.dataset import build_cc3m_dataset, Tokenizer, build_dataloader, CC3MDataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
import torch.distributed as dist
import torch
from utils import get_args, set_DDP


def main():
    args = get_args()
    set_DDP(args)
    
    dataset, _ = build_dataloader(
        build_cc3m_dataset(Tokenizer()),
        batch_size=10,
        collate_fn=CC3MDataset.collate_fn
    )
    model = fasterrcnn_resnet50_fpn_v2(pretrained=True)


if __name__ == '__main__':
    main()
