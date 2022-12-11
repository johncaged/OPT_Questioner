import torch.distributed as dist
from torch_lib.callback import Callback, CallbackContainer
from torch_lib.callback.common import SaveMetrics, SaveCheckpoint
import os


class MyCallback(Callback):
    
    def __init__(self):
        super().__init__()
        self.rank = dist.get_rank()
        if self.rank == 0:
            self.callback = CallbackContainer([
                SaveMetrics(),
                SaveCheckpoint(1, save_optimizer=True, save_epoch=True)
            ])
    
    def epoch_end(self, ctx):
        if self.rank == 0:
            self.callback.epoch_end(ctx)

    def epoch_begin(self, ctx):
        # set random seed
        ctx.ctx_check('custom.train_sampler', silent=False)
        ctx.custom.train_sampler.set_epoch(ctx.epoch.current)

    def step_end(self, ctx):
        # print new line in slurm mode to avoid output could not be seen.
        if 'SLURM_PROCID' in os.environ and self.rank == 0:
            print()


class EvalCallback(Callback):
    
    def __init__(self):
        super().__init__()
        self.rank = dist.get_rank()
    
    def end(self, ctx):
        ctx.custom.epoch_metrics.append(ctx.epoch.eval_metrics)
        ctx.custom.epoch_losses.append(ctx.epoch.eval_loss)

    def step_end(self, ctx):
        # print new line in slurm mode to avoid output could not be seen.
        if 'SLURM_PROCID' in os.environ and self.rank == 0:
            print()
