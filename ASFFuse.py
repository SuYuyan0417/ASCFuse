import torch
import torch.nn as nn
from modules.shared_encoder import SharedEncoder
from modules.cmdsf import CMDSF
from modules.reconstruction_decoder import ReconstructionDecoder
from modules.wave_fusion_block import WDFB


class INet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=64,
                 enc_blocks=4, num_inv=1):
        super(INet, self).__init__()
        self.in_c1 = in_channels
        self.in_c2 = in_channels
        self.dim = dim

        self.shared_encoder = SharedEncoder(
            in_channels=in_channels, dim=dim, num_blocks=enc_blocks
        )
        self.cmdsf1 = CMDSF(dim=dim, shifted=True)
        self.wave_fusion = WDFB(dim=dim, num_inv=num_inv)
        self.cmdsf2 = CMDSF(dim=dim, shifted=True)
        self.decoder = ReconstructionDecoder(
            dim=dim, out_channels=out_channels
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, rev=False):
        x1 = x.narrow(1, 0, self.in_c1)
        x2 = x.narrow(1, self.in_c1, self.in_c2)

        feat_v = self.shared_encoder(x1)
        feat_r = self.shared_encoder(x2)

        feat_cat, feat_v, feat_r = self.cmdsf1(feat_v, feat_r)
        feat_v = feat_v + feat_cat
        feat_r = feat_r + feat_cat

        feat_v, feat_r = self.wave_fusion(feat_v, feat_r)

        feat_cat2, feat_v, feat_r = self.cmdsf2(feat_v, feat_r)
        feat_v = feat_v + feat_cat2
        feat_r = feat_r + feat_cat2

        out = self.decoder(feat_v, feat_r)
        out = self.bn(out)
        return out
