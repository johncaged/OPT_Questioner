import torch.distributed as dist
from torch_lib.callback import Callback, CallbackContainer
from torch_lib.callback.common import SaveMetrics, SaveCheckpoint


class MyCallback(Callback):
    
    def __init__(self):
        super().__init__()
        self.rank = dist.get_rank()
        if self.rank == 0:
            self.callback = CallbackContainer([
                SaveMetrics(),
                SaveCheckpoint(1, 'checkpoint.pt', save_optimizer=True, save_epoch=True)
            ])
    
    def epoch_end(self, ctx):
        if self.rank == 0:
            self.callback.epoch_end(ctx)
