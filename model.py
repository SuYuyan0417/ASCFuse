import torch.nn as nn
from ASFFuse import INet


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = INet(in_channels=3, out_channels=3)

    def forward(self, x, rev=False):
        if not rev:
            out = self.model(x)
        else:
            out = self.model(x, rev=True)
        return out
