from torch_lib.core.handler import BackwardHandler, HandlerContainer, LossHandler
from torch_lib.core.handler import Handler
from apex import amp
import torch.distributed as dist
import torch
from utils import ToCuda
from torch_lib.util import is_nothing


class AMPBackward(BackwardHandler):
    
    def handle(self, ctx):
        with amp.scale_loss(ctx.step.loss, ctx.run.optimizer) as scaled_loss:
            scaled_loss.backward()


class CustomLossHandler(LossHandler):
    
    def handle(self, ctx):
        super().handle(ctx)
        ctx.custom.loss.append(float(ctx.step.loss.clone().cpu().detach()))
        ctx.step.loss = ctx.step.loss / 3


class CoverLossHandler(Handler):
    
    def handle(self, ctx):
        ctx.step.loss = ctx.custom.loss
        
        tensor_list = [ToCuda(torch.zeros(4)) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, ToCuda(torch.tensor([sum(ctx.custom.loss) / 3, ctx.custom.loss[0], ctx.custom.loss[1], ctx.custom.loss[2]])))
        
        gathered_data = torch.cat(tensor_list, dim=0)
        ctx.step.loss = float(gathered_data[0].mean().clone().cpu().detach())
        ctx.custom.metrics = {
            'question_loss': float(gathered_data[1].mean().clone().cpu().detach()),
            'answer_loss': float(gathered_data[2].mean().clone().cpu().detach()),
            'caption_loss': float(gathered_data[3].mean().clone().cpu().detach())
        }
        ctx.custom.loss = []


class SetModelHandler(Handler):
    
    def __init__(self, _forward):
        # four modes.
        assert _forward in ['question', 'answer', 'caption']
        self._forward = _forward
    
    def handle(self, ctx):
        ctx.model.module._forward = self._forward


def set_handler(proxy):
    proxy.custom.loss = []
    proxy.handler.Loss = CustomLossHandler
    proxy.handler.Backward = AMPBackward
    proxy.build_train()
    proxy.run.train[1][4].insert(1, SetModelHandler('question'))
    proxy.run.train[1][4].insert(4, HandlerContainer([
        proxy.handler.Backward(),
        SetModelHandler('answer'),
        proxy.handler.Forward(),
        proxy.handler.Loss(),
        proxy.handler.Backward(),
        SetModelHandler('caption'),
        proxy.handler.Forward(),
        proxy.handler.Loss()
    ]))
    proxy.run.train[1][4].insert(6, CoverLossHandler())
    proxy.run.train[1][9].insert(0, HandlerContainer([
        SetModelHandler('question'),
        proxy.handler.Forward(),
        proxy.handler.Loss(),
        SetModelHandler('answer'),
        proxy.handler.Forward(),
        proxy.handler.Loss(),
        SetModelHandler('caption')
    ]))
    proxy.run.train[1][9].insert(3, CoverLossHandler())

    proxy.build_eval()
    proxy.run.eval[4].insert(1, HandlerContainer([
        SetModelHandler('question'),
        proxy.handler.Forward(),
        proxy.handler.Loss(),
        SetModelHandler('answer'),
        proxy.handler.Forward(),
        proxy.handler.Loss(),
        SetModelHandler('caption')
    ]))
    proxy.run.eval[4].insert(4, CoverLossHandler())
