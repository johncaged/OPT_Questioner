import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, y_pred, y_true):
        # y_true is not used.
        prediction_scores, labels = y_pred
        loss = F.cross_entropy(prediction_scores,
                               labels,
                               reduction='none')
        if self.label_smoothing > 0:
            smooth_loss = -F.log_softmax(prediction_scores, dim=-1).mean(dim=-1)
            loss = (1 - self.label_smoothing) * qa_loss_tv + self.label_smoothing * smooth_loss

        return loss.mean()
