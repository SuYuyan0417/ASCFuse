"""
CMDSF: 跨模态双注意力空域融合模块 (Cross-Modal Dual-attention Spatial Fusion)。

3分支结构:
    feat_v (绿路)  [B, dim, H, W]
    feat_r (红路)  [B, dim, H, W]
              │
              ├──► 分支1: cat(feat_v, feat_r) → CMSA (跨模态双注意力协同模块)
              │                            │
              │                            ▼
              │                         feat_cat  [B, dim, H, W]  (融合先验)
              │
              ├──► 分支2: feat_v ─┐
              │                   ├──► MAC (跨模态交互) ──► feat_v'  (含红路信息)
              └──► 分支3: feat_r ─┘                     └──► feat_r'  (含绿路信息)

包含两个创新子模块:
    1. DARM: 跨模态通道注意力 (CMCA) + 方向感知多尺度空间注意力 (DASA)
    2. MAC:  双向门控跨模态交互 (权重共享)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.module_util import initialize_weights


class CrossModalChannelAttention(nn.Module):
    """
    跨模态通道注意力 (Cross-Modal Channel Attention, CMCA)。

    创新点：
        - 输入是拼接的红绿特征 [B, 2*dim, H, W]
        - 同时利用全局平均池化和全局最大池化
        - 用红绿两路的联合统计生成单路输出特征的通道注意力
    """

    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden_dim = max(dim // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x_cat, rb_out):
        """
        Args:
            x_cat:   [B, 2*dim, H, W]  拼接的红绿特征
            rb_out:  [B, dim, H, W]   RB输出
        Returns:
            [B, dim, H, W]
        """
        b, c, _, _ = rb_out.size()  # c = dim
        avg = F.adaptive_avg_pool2d(x_cat, 1).view(b, 2 * c)
        max_ = F.adaptive_max_pool2d(x_cat, 1).view(b, 2 * c)
        att = self.mlp(avg + max_).view(b, c, 1, 1)
        return rb_out * att


class DirectionalSpatialAttention(nn.Module):
    """
    方向感知多尺度空间注意力 (Directional-Aware Spatial Attention, DASA)。

    创新点：
        - 空间注意力分水平和垂直两个方向
        - 每个方向用多尺度 1D 卷积 (K=3,5,7) 捕获不同尺度纹理
        - 水平方向 (1×K) 捕获横向纹理
        - 垂直方向 (K×1) 捕获纵向纹理
        - 通道注意力输出门控方向注意力，形成协同
    """

    def __init__(self, dim):
        super().__init__()
        # 通道门控
        self.gate_conv = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)

        # 水平方向多尺度 1D 卷积 (1×K)
        self.h_conv3 = nn.Conv2d(dim, dim, (1, 3), 1, (0, 1), bias=True)
        self.h_conv5 = nn.Conv2d(dim, dim, (1, 5), 1, (0, 2), bias=True)
        self.h_conv7 = nn.Conv2d(dim, dim, (1, 7), 1, (0, 3), bias=True)

        # 垂直方向多尺度 1D 卷积 (K×1)
        self.v_conv3 = nn.Conv2d(dim, dim, (3, 1), 1, (1, 0), bias=True)
        self.v_conv5 = nn.Conv2d(dim, dim, (5, 1), 1, (2, 0), bias=True)
        self.v_conv7 = nn.Conv2d(dim, dim, (7, 1), 1, (3, 0), bias=True)

        # 融合两个方向的注意力
        self.fuse = nn.Conv2d(dim * 2, 1, 3, 1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

        # 方向可学习权重 (平衡水平/垂直)
        self.wh = nn.Parameter(torch.tensor(0.5))
        self.wv = nn.Parameter(torch.tensor(0.5))

    def forward(self, rb_out, channel_out):
        """
        Args:
            rb_out:      [B, dim, H, W]  RB输出
            channel_out: [B, dim, H, W]  通道注意力输出 (作为门控)
        Returns:
            [B, dim, H, W]
        """
        # 通道门控
        gate = self.gate_conv(channel_out)
        feat = rb_out + gate

        # 水平方向多尺度 (1×K)
        h_att = self.h_conv3(feat) + self.h_conv5(feat) + self.h_conv7(feat)
        h_att = h_att.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # 垂直方向多尺度 (K×1)
        v_att = self.v_conv3(feat) + self.v_conv5(feat) + self.v_conv7(feat)
        v_att = v_att.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # 可学习加权方向融合
        att = self.wh * h_att + self.wv * v_att  # [B, 1, H, W]
        att = self.sigmoid(att)

        return rb_out * att


class CMSA(nn.Module):
    """
    跨模态双注意力协同模块 (Cross-Modal Synergistic Attention, CMSA)。

    创新点：
        1. 跨模态通道注意力 (CMCA)：用红绿拼接特征生成注意力，增强模态感知
        2. 方向感知多尺度空间注意力 (DASA)：通道输出门控方向注意力
        3. 可学习加权融合：自动平衡通道/空间分支贡献
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 残差块 RB
        self.rb = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_dim),
        )

        self.cmc_att = CrossModalChannelAttention(out_dim, reduction=4)
        self.dsa_att = DirectionalSpatialAttention(out_dim)

        # 可学习加权参数
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        # 投影 shortcut
        self.shortcut = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)
        self.final_relu = nn.ReLU(inplace=True)

        initialize_weights([self.rb[0], self.rb[3], self.shortcut], 0.1)

    def forward(self, x):
        """
        x: [B, in_dim, H, W]
        out: [B, out_dim, H, W]
        """
        residual = self.shortcut(x)
        rb_out = self.rb(x)

        # 跨模态通道注意力 (输入拼接特征也参与)
        channel_out = self.cmc_att(x, rb_out)
        # 方向感知多尺度空间注意力 (由通道输出门控)
        spatial_out = self.dsa_att(rb_out, channel_out)

        # 可学习加权融合 + 残差
        out = residual + self.alpha * channel_out + self.beta * spatial_out
        out = self.final_relu(out)
        return out


class MCA(nn.Module):
    """
    模态交叉注意力 (Modal Cross Attention, MCA)。

    基于窗口的多头交叉注意力 (Window-based Cross-Attention)。

    结构：
        - Query 来自当前分支 (feat_q)
        - Key/Value 来自另一分支 (feat_kv)
        - 在 H×W 的局部窗口内做 Scaled Dot-Product Attention
        - 可选 shifted window，增强跨窗口交互

    Args:
        dim: 输入通道数
        num_heads: 注意力头数
        window_size: 窗口边长 (像素数)，要求 H、W 能被 window_size 整除
        shifted: 是否使用 shifted window
    """

    def __init__(self, dim, num_heads=8, window_size=8, shifted=False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.shifted = shifted
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)
        self.kv_proj = nn.Conv2d(dim, dim * 2, 1, 1, 0, bias=True)
        self.out_proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        initialize_weights([self.q_proj, self.kv_proj, self.out_proj], 0.1)

    def window_partition(self, x, window_size):
        B, C, H, W = x.shape
        x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        windows = x.view(-1, window_size * window_size, C)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, -1, H, W)
        return x

    def forward(self, feat_q, feat_kv):
        """
        Args:
            feat_q:  [B, dim, H, W]  生成 Query 的分支
            feat_kv: [B, dim, H, W]  生成 Key/Value 的分支
        Returns:
            [B, dim, H, W]
        """
        B, C, H, W = feat_q.shape
        shift = self.window_size // 2 if self.shifted else 0

        q = self.q_proj(feat_q)
        kv = self.kv_proj(feat_kv)
        k, v = kv.chunk(2, dim=1)

        # Shifted window: cyclic shift
        if shift > 0:
            q = torch.roll(q, shifts=(-shift, -shift), dims=(2, 3))
            k = torch.roll(k, shifts=(-shift, -shift), dims=(2, 3))
            v = torch.roll(v, shifts=(-shift, -shift), dims=(2, 3))

        # Partition into windows
        q_win = self.window_partition(q, self.window_size)  # [N, Ws*Ws, C]
        k_win = self.window_partition(k, self.window_size)
        v_win = self.window_partition(v, self.window_size)

        N, L, _ = q_win.shape

        # Multi-head attention
        q_win = q_win.view(N, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_win = k_win.view(N, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_win = v_win.view(N, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q_win @ k_win.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = attn @ v_win
        out = out.permute(0, 2, 1, 3).contiguous().view(N, L, C)

        # Reverse windows
        out = self.window_reverse(out, self.window_size, H, W)

        # Reverse cyclic shift
        if shift > 0:
            out = torch.roll(out, shifts=(shift, shift), dims=(2, 3))

        out = self.out_proj(out)

        # Residual + LayerNorm + FFN
        out = feat_q + out
        out = self.norm1(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        ffn_out = self.ffn(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = out + ffn_out
        out = self.norm2(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return out


class MAC(nn.Module):
    """
    跨模态交互块 (Mutual Attention Cross, MAC)。

    基于窗口的 QKV 交叉注意力：
        - 绿路 feat_v 作为 Query，查询红外路 feat_r 的 Key/Value
        - 红路 feat_r 作为 Query，查询绿路 feat_v 的 Key/Value
        - 两个方向共享同一个 WindowCrossAttention 权重

    Args:
        dim: 输入特征通道数
        num_heads: 注意力头数
        window_size: 窗口边长
        shifted: 是否使用 shifted window
    """

    def __init__(self, dim, num_heads=8, window_size=8, shifted=False):
        super().__init__()
        self.attn = MCA(dim, num_heads, window_size, shifted)

    def forward(self, feat_v, feat_r):
        """
        Args:
            feat_v: [B, dim, H, W]  绿路特征
            feat_r: [B, dim, H, W]  红路特征
        Returns:
            feat_v_new, feat_r_new  (与输入同形状)
        """
        # 绿路查红路
        feat_v_new = self.attn(feat_v, feat_r)
        # 红路查绿路
        feat_r_new = self.attn(feat_r, feat_v)

        return feat_v_new, feat_r_new


class CMDSF(nn.Module):
    """
    跨模态双注意力空域融合模块 (Cross-Modal Dual-attention Spatial Fusion, CMDSF)。

    3分支结构:
        分支1: cat(红绿) → CMSA → feat_cat (融合先验)
        分支2&3: MAC 跨模态窗口 QKV 交叉注意力 → feat_v', feat_r'

    Args:
        dim: 输入特征通道数
        num_heads: MAC 窗口注意力头数
        window_size: MAC 窗口边长 (像素数)，要求 H、W 能被 window_size 整除
        shifted: MAC 是否使用 shifted window

    Input:
        feat_v, feat_r: [B, dim, H, W]

    Output:
        feat_cat:  [B, dim, H, W]  (拼接后过CMSA)
        feat_v:    [B, dim, H, W]  (绿路,经MAC跨模态交互)
        feat_r:    [B, dim, H, W]  (红路,经MAC跨模态交互)
    """

    def __init__(self, dim, num_heads=8, window_size=8, shifted=False):
        super().__init__()
        # 分支1：拼接后过跨模态双注意力协同模块 CMSA
        self.cmsa = CMSA(in_dim=dim * 2, out_dim=dim)
        # 分支2&3：红绿跨模态交互 MAC (基于窗口 QKV 交叉注意力)
        self.mac = MAC(dim, num_heads, window_size, shifted)

    def forward(self, feat_v, feat_r):
        """
        返回 (feat_cat, feat_v, feat_r)。
        """
        # 分支1: 拼接 → CMSA
        cat = torch.cat([feat_v, feat_r], dim=1)  # [B, 2*dim, H, W]
        feat_cat = self.cmsa(cat)                   # [B, dim, H, W]

        # 分支2&3: 红绿跨模态交互
        feat_v, feat_r = self.mac(feat_v, feat_r)

        return feat_cat, feat_v, feat_r
