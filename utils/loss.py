import torch
import torch.nn as nn


def BCELoss():
    loss_fnc = nn.BCEWithLogitsLoss(reduction='mean',pos_weight=torch.cuda.FloatTensor([1]))
    return loss_fnc

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=3, logits=False, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.size_average = size_average
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.criterion(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.size_average:
            return F_loss.mean()
        else:
            return F_loss.sum()


def get_lossfunc(loss_name):
    try:
        if loss_name == "BCE":
            return BCELoss()
        if loss_name == "Focal":
            return BinaryFocalLoss()
    except Exception as e:
        print(str(e))
