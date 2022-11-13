from torch_lib import Proxy
from model.model import QuestionerGenerator, ModelWrapper
from torch_lib import Proxy
from utils import ToCuda
from data.dataset import build_coco_dataset, Tokenizer
from torch.utils.data import DataLoader
from utils.metrics import MyMetric
import torch
from utils.loss import Loss


def main():
    # tokenizer
    tokenizer = Tokenizer()
    # build and load model
    model = QuestionerGenerator(tokenizer=tokenizer)
    model = ModelWrapper(model)
    checkpoint = torch.load('./log/test/checkpoint/checkpoint.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    del checkpoint
    model: QuestionerGenerator = ToCuda(model.module)
    val_dataset = DataLoader(build_coco_dataset('answer', 'val', tokenizer, True, 'once'), batch_size=10, shuffle=True)
    proxy = Proxy(model)
    proxy.build(
        # loss=Loss(),
        metrics=MyMetric()
    )
    proxy.eval(
        val_dataset
    )


if __name__ == '__main__':
    main()
