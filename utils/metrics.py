from torch_lib.metric import Metric
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from data.dataset import Tokenizer
import torch


class MyMetric(Metric):
    
    def __init__(self):
        super().__init__()
        self.cider = Cider()
        self.spice = Spice()
        self.tokenizer = Tokenizer()
        self.predictions = {}
        self.targets = {}
    
    def get(self, ctx):
        def convert_items(items, tokenizer: Tokenizer):
            results = []
            for item in items:
                results.append(' '.join(tokenizer.tokenizer.convert_ids_to_tokens(item[(0 if item[0] != 101 else 1):item.index(102)])).replace(' ##', ''))
            return results
        
        detach = lambda x: x.detach().cpu().tolist()
        
        batch = ctx.step.batch
        target = torch.flatten(batch[1], start_dim=0, end_dim=1)
        prediction = ctx.step.y_pred
        
        target = convert_items(detach(target), self.tokenizer)
        prediction = convert_items(detach(prediction), self.tokenizer)
        
        pre, tar = {}, {}
        for t, p, i in zip(target, prediction, list(batch[2][0])):
            self.predictions[i] = [p]
            self.targets[i] = [t]
            pre[i] = [p]
            tar[i] = [t]
        score_s, _ = self.spice.compute_score(tar, pre)
        result = {
            'SPICE': score_s
        }
        
        if (ctx.step.current + 1) % 20 == 0:
            score_c, _ = self.cider.compute_score(self.targets, self.predictions)
            self.targets = {}
            self.predictions = {}
            result['CIDEr'] = score_c
        return result


class MultiTaskMetric(Metric):
    
    def get(self, ctx):
        return ctx.custom.metrics
