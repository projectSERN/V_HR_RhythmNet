import torch.nn as nn
import torch
from config import config
from utils.custom_loss import MyLoss

class RhythmNetLoss(nn.Module):
    def __init__(self, weight=100.0):
        super(RhythmNetLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.lambd = weight
        self.gru_preds_considered = None
        self.custom_loss = MyLoss()

    def forward(self, resnet_preds, gru_preds, targets):
        l1_loss = self.l1_loss(resnet_preds, targets)
        smooth_loss_component = self.smooth_loss(gru_preds)

        loss = l1_loss + self.lambd * smooth_loss_component
        return loss

    # Need to write backward pass for this loss function
    def smooth_loss(self, gru_preds):
        smooth_loss = torch.zeros(1).to(config.DEVICE)
        self.gru_preds_considered = gru_preds.flatten()
        for hr_t in self.gru_preds_considered:
            smooth_loss = smooth_loss + self.custom_loss.apply(torch.autograd.Variable(hr_t, requires_grad=True),
                                                               self.gru_preds_considered,
                                                               self.gru_preds_considered.shape[0])
        return smooth_loss / self.gru_preds_considered.shape[0]
