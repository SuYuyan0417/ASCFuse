import torch.nn as nn
import torch.nn.functional as F
from modules.module_util import initialize_weights


class ReconstructionDecoder(nn.Module):

    def __init__(self, dim=64, out_channels=3):
        super().__init__()
        # 红外/红路分支: 1×1 卷积生成调制信息
        self.red_conv1 = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)

        # 可见光/绿路分支: 3×3 卷积主重建流
        self.green_conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.green_conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.green_conv3 = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        initialize_weights(
            [self.red_conv1, self.green_conv1, self.green_conv2, self.green_conv3], 0.1
        )

    def _layer_norm(self, x):
        """
        LayerNorm作用在通道维度 (C)。
        x: [B, C, H, W] -> [B, H, W, C] -> norm -> [B, C, H, W]
        """
        b, c, h, w = x.size()
        y = x.permute(0, 2, 3, 1).contiguous()
        y = F.layer_norm(y, y.shape[-1:])  # 对最后一维(C)做LN
        y = y.permute(0, 3, 1, 2).contiguous()
        return y

    def forward(self, feat_v, feat_r):
        # 分别LayerNorm
        feat_v = self._layer_norm(feat_v)
        feat_r = self._layer_norm(feat_r)

        # 红路分支生成调制信息
        red_mod = self.red_conv1(feat_r)

        # 绿路主重建 + 红路调制信息注入
        x = self.green_conv1(feat_v)
        x = x + red_mod  # 红外信息注入可见光

        x = self.lrelu(self.green_conv2(x))
        out = self.lrelu(self.green_conv3(x))

        return out
