from data.dataset import build_cc3m_dataset, Tokenizer, build_dataloader, CC3MDataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch.distributed as dist
import torch
from utils import get_args, set_DDP
import os
from utils import ToCuda
import json
import tqdm
import glob


@torch.no_grad()
def main():
    args = get_args()
    set_DDP(args)
    os.environ['TORCH_HOME'] = './checkpoint/pretrained/'
    
    dataset, _ = build_dataloader(
        build_cc3m_dataset(Tokenizer(), raw_resolution=True),
        batch_size=10,
        collate_fn=CC3MDataset.raw_resolution_collate_fn
    )
    model = ToCuda(fasterrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=0)).eval()
    
    _predictions = {}
    for batch, img_ids in tqdm.tqdm(dataset):
        imgs = [ToCuda(img.squeeze(0)) for img in batch['imgs']]
        results = model(imgs)
        for result, img_id in zip(results, img_ids):
            boxes = filter_boxes(result)
            _predictions[str(img_id)] = boxes
    with open('custom_dataset/CC3M/objects_{}.json'.format(dist.get_rank()), 'w') as f:
        json.dump(_predictions, f, indent=4)
    
    tensor_list = [ToCuda(torch.zeros(1)) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, ToCuda(torch.tensor([0.0])))
    if dist.get_rank() == 0:
        datas = glob.glob('custom_dataset/CC3M/objects_[0-9]*.json')
        full_data = {}
        for data in datas:
            with open(data) as f:
                full_data.update(json.load(f))
        with open('custom_dataset/CC3M/objects.json', 'w') as f:
            json.dump(full_data, f, indent=4)


def filter_boxes(result):
    return result['boxes'].detach().clone().cpu().tolist()


if __name__ == '__main__':
    main()
