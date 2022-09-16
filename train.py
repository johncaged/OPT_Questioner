from torch_lib import Proxy
from model.model import Questioner
import os
from data.dataset import Tokenizer, build_dataloader, build_dataset
from utils.loss import Loss
from torch.optim import Adam

# tokenizer
tokenizer = Tokenizer()
# build and load model
model = Questioner(tokenizer)
model.load_pretrained_weights()
# build dataset
train_dataset = build_dataloader(build_dataset('answer', 'train', tokenizer), 64)
val_dataset = build_dataloader(build_dataset('answer', 'val', tokenizer), 64)
# torch-lib pipeline
proxy = Proxy(model, 'cuda:0')
proxy.build(
    loss=Loss(),
    optimizer=Adam(model.parameters(), lr=1e-6)
)
proxy.train(
    train_dataset,
    100,
    val_dataset
)
