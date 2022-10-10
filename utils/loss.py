import torch.nn as nn
import torch.nn.functional as F
import torch


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
            loss = (1 - self.label_smoothing) * loss + self.label_smoothing * smooth_loss

        return loss.mean()


class MixLoss(nn.Module):
    
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.question_loss = Loss(label_smoothing=label_smoothing)
        self.answer_loss = Loss(label_smoothing=label_smoothing)
    
    def forward(self, y_pred, y_true):
        question_loss = self.question_loss((y_pred[0], y_pred[1]), None)
        answer_loss = self.answer_loss((y_pred[2], y_pred[3]), None)
        return (question_loss + answer_loss) / 2
