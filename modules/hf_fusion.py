import torch
import torch.nn as nn
import torch.nn.functional as F


class HFFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 方向统计 -> 权重的 ConvBlock
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, 2, 1, 1, 0, bias=True),   # 输出 2 通道 (w1, w2)
        )
        self.sigmoid = nn.Sigmoid()

        # HH 单独过卷积
        self.hh_conv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)

    def forward(self, lh, hl, hh):

        b, c, H, W = lh.shape

        # 1) 方向感知全局平均池化
        gap_h = lh.mean(dim=2, keepdim=True)   # [B, C, 1, W]  沿 H 方向
        gap_w = hl.mean(dim=3, keepdim=True)   # [B, C, H, 1]  沿 W 方向

        # 2) 统一为 [B, C, 1, L] 形状, L = max(H, W)
        gap_w = gap_w.permute(0, 1, 3, 2)                # [B, C, 1, H]

        if gap_h.size(3) != gap_w.size(3):
            target = max(gap_h.size(3), gap_w.size(3))
            gap_h = F.adaptive_avg_pool1d(gap_h.squeeze(2), target).unsqueeze(2)
            gap_w = F.adaptive_avg_pool1d(gap_w.squeeze(2), target).unsqueeze(2)

        # 3) 沿通道拼接, 过 ConvBlock 生成两个权重
        cat = torch.cat([gap_h, gap_w], dim=1)   # [B, 2*C, 1, L]
        w = self.conv_block(cat)                  # [B, 2, 1, L]

        w1 = w[:, 0:1, :, :]                      # [B, 1, 1, L]
        w2 = w[:, 1:2, :, :]                      # [B, 1, 1, L]

        # 广播到 [B, 1, H, W]
        w1 = F.interpolate(w1, size=(int(H), int(W)), mode='nearest')
        w2 = F.interpolate(w2, size=(int(H), int(W)), mode='nearest')
        w1 = self.sigmoid(w1)
        w2 = self.sigmoid(w2)

        # 4) 交叉残差融合
        hl_out = hl + w1 * lh
        lh_out = lh + w2 * hl

        # 5) HH 单独过卷积
        hh_out = self.hh_conv(hh)

        return lh_out, hl_out, hh_out
