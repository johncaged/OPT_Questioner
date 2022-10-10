from torch_lib.core.handler import BackwardHandler, HandlerContainer, LossHandler
from torch_lib.core.handler import Handler
from apex import amp


class AMPBackward(BackwardHandler):
    
    def handle(self, ctx):
        with amp.scale_loss(ctx.step.loss, ctx.run.optimizer) as scaled_loss:
            scaled_loss.backward()


class CustomLossHandler(LossHandler):
    
    def handle(self, ctx):
        super().handle(ctx)
        ctx.step.loss = ctx.step.loss / 2
        if ctx.custom.loss == -1:
            ctx.custom.loss = float(ctx.step.loss.clone().cpu().detach())
        else:
            ctx.custom.loss += float(ctx.step.loss.clone().cpu().detach())


class CoverLossHandler(Handler):
    
    def handle(self, ctx):
        ctx.step.loss = ctx.custom.loss
        ctx.custom.loss = -1


class SetModelHandler(Handler):
    
    def __init__(self, _forward_answer):
        self._forward_answer = _forward_answer
    
    def handle(self, ctx):
        ctx.model.module._forward_answer = self._forward_answer


def set_handler(proxy):
    proxy.custom.loss = -1
    proxy.handler.Loss = CustomLossHandler
    proxy.handler.Backward = AMPBackward
    proxy.build_train()
    proxy.run.train[1][4].insert(1, SetModelHandler(False))
    proxy.run.train[1][4].insert(4, HandlerContainer([
        proxy.handler.Backward(),
        SetModelHandler(True),
        proxy.handler.Forward(),
        proxy.handler.Loss()
    ]))
    proxy.run.train[1][4].insert(6, CoverLossHandler())
