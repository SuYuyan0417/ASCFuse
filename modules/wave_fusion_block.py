import torch
import torch.nn as nn
from invblock import INV_block
from modules.hf_fusion import HFFusion


def dwt_init(x):
    """不可学习 2D Haar 小波分解, 返回 (LL, HL, LH, HH)。"""
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, x_HL, x_LH, x_HH


def iwt_init(x):
    """不可学习 2D Haar 小波逆变换, 输入 4*C 通道拼接张量。"""
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width],
                    device=x.device, dtype=x.dtype)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class WDFB(nn.Module):
    """
    小波融合块: DWT → INN(LL耦合) + HFFusion(高频处理) → IWT。
    """

    def __init__(self, dim, num_inv=2):
        super().__init__()
        # 小波变换 (不可学习)
        self.dwt = DWT()
        self.iwt = IWT()

        # INN 可逆块: 红绿共享同一个 INN, 对 LL 做跨模态耦合
        self.inv_blocks = nn.ModuleList([
            INV_block(in_1=dim, in_2=dim) for _ in range(num_inv)
        ])

        # 高频融合: 红绿各自独立处理 (权重不共享)
        self.hf_fusion_v = HFFusion(dim)
        self.hf_fusion_r = HFFusion(dim)

    def forward(self, feat_v, feat_r):
        # 1) DWT 分解
        LL_v, HL_v, LH_v, HH_v = self.dwt(feat_v)   # [B, dim, H/2, W/2]
        LL_r, HL_r, LH_r, HH_r = self.dwt(feat_r)

        # 2) INN 低频耦合: LL_v 与 LL_r 共享同一个 INN (直接传两路,不拼接)
        for blk in self.inv_blocks:
            LL_v, LL_r = blk(LL_v, LL_r)

        # 3) 高频融合: 各模态内部方向交叉门控 (LH/HL) + HH 卷积
        LH_v, HL_v, HH_v = self.hf_fusion_v(LH_v, HL_v, HH_v)
        LH_r, HL_r, HH_r = self.hf_fusion_r(LH_r, HL_r, HH_r)

        # 4) IWT 重建
        feat_v = self.iwt(torch.cat([LL_v, HL_v, LH_v, HH_v], dim=1))  # [B, dim, H, W]
        feat_r = self.iwt(torch.cat([LL_r, HL_r, LH_r, HH_r], dim=1))

        return feat_v, feat_r
