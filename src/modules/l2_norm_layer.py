import torch
import torch.nn as nn
import torch.nn.functional as F

class L2NormLayer(nn.Module):
    def __init__(self, dim=1, p=2):
        super().__init__()
        self.dim = dim
        self.p = p

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=self.dim)