from torch_lib.core.handler import BackwardHandler
from apex import amp


class AMPBackward(BackwardHandler):
    
    def handle(self, ctx):
        with amp.scale_loss(ctx.step.loss, ctx.run.optimizer) as scaled_loss:
            scaled_loss.backward()


def set_handler(proxy):
    proxy.handler.Backward = AMPBackward
    proxy.build_train()
