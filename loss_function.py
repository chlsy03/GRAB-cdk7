import torch
from torch import nn

class Soft_NLL_Loss(nn.Module):
    def __init__(self, dim=-1):
        super(Soft_NLL_Loss, self).__init__()
        self.dim = dim

    def forward(self, pred, target):
        return torch.mean(torch.sum(-target * torch.log(pred), dim=self.dim))