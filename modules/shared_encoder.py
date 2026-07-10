import torch.nn as nn
from modules.module_util import initialize_weights


class AttentionBlock(nn.Module):

    def __init__(self, dim, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, max(dim // reduction, 4), bias=False),
            nn.GELU(),
            nn.Linear(max(dim // reduction, 4), dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class RestormBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        # 深度可分离 3x3 卷积 (分组=通道数)
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True)
        self.norm1 = nn.LayerNorm(dim)
        self.pwconv1 = nn.Conv2d(dim, dim * 2, 1, 1, 0, bias=True)   # 升维
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * 2, dim, 1, 1, 0, bias=True)   # 降维
        self.attn = AttentionBlock(dim)

        # 前馈网络 (MLP)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn_conv1 = nn.Conv2d(dim, dim * 2, 1, 1, 0, bias=True)
        self.ffn_act = nn.GELU()
        self.ffn_conv2 = nn.Conv2d(dim * 2, dim, 1, 1, 0, bias=True)

        initialize_weights(
            [self.dwconv, self.pwconv1, self.pwconv2,
             self.ffn_conv1, self.ffn_conv2], 0.1
        )

    def forward(self, x):
        # --- 注意力分支 ---
        residual = x
        y = self.dwconv(x)
        # LayerNorm 作用在通道维: [B,C,H,W] -> [B,H,W,C] -> norm -> 回去
        y = y.permute(0, 2, 3, 1)
        y = self.norm1(y)
        y = y.permute(0, 3, 1, 2)
        y = self.pwconv2(self.act(self.pwconv1(y)))
        y = self.attn(y)
        x = residual + y

        # --- FFN 分支 ---
        residual = x
        y = x.permute(0, 2, 3, 1)
        y = self.norm2(y)
        y = y.permute(0, 3, 1, 2)
        y = self.ffn_conv2(self.ffn_act(self.ffn_conv1(y)))
        x = residual + y
        return x


class SharedEncoder(nn.Module):

    def __init__(self, in_channels=3, dim=64, num_blocks=4):
        super().__init__()
        self.dim = dim

        # 浅层特征提取
        self.shallow = nn.Conv2d(in_channels, dim, 3, 1, 1, bias=True)

        # 主体: Restormer 块堆叠
        self.body = nn.Sequential(*[RestormBlock(dim) for _ in range(num_blocks)])

        # 特征整合
        self.tail = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)

        initialize_weights([self.shallow, self.tail], 0.1)

    def forward(self, x):
        feat = self.shallow(x)
        feat = self.body(feat)
        feat = self.tail(feat)
        return feat
