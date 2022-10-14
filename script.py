from vqa_utils.vqaTools.vqa import VQA
from utils import parse_yaml, default_config_path
from data.dataset import CC3MDataset, Tokenizer, build_cc3m_dataset, CaptionProcessor
from torch.utils.data import DataLoader
import torch


def main():
    print((torch.ones(1) / torch.zeros(1)).tolist())
    a = torch.tensor([1.0])
    print(a, torch.zeros(1).size())
    # dataset = build_cc3m_dataset(Tokenizer())
    # dataset = DataLoader(dataset, shuffle=True, batch_size=64, collate_fn=CC3MDataset.collate_fn)
    
    # # TODO: IndexParser
    # # TODO: proxy.predict
    # # TODO: callback
    # for i in dataset:
    #     print(i['imgs'].size())
    #     print(i['tips'].size())
    #     return

if __name__ == '__main__':
    main()
