# projects/lk_exp/layers.py   ← 直接覆盖掉原文件
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


# -------------------------- 辅助模块 -------------------------- #
class SEModule(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        w = self.avgpool(x)
        w = self.fc(w)
        return x * w


# --------------------- ① Fourier Attention ------------------- #
class FourierAttention(nn.Module):
    """简化版本用于调试 - 用卷积替代FFT操作"""
    def __init__(self, in_channels):
        super().__init__()
        # 用深度可分离卷积模拟频域操作
        self.conv = nn.Sequential(
            # 深度卷积 - 模拟频域变换
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # 点卷积 - 模拟频域融合
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.se = SEModule(in_channels, reduction=8)
        
        # 添加权重初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 简化版本：卷积 + SE注意力
        x_conv = self.conv(x)  # 用卷积替代FFT操作
        x_se = self.se(x_conv)  # SE注意力
        
        # 添加残差连接确保梯度传播
        return x + x_se



# --------------------- ② Spatial Attention ------------------- #
class SpatialAttention(nn.Module):
    """和原理图相同的 QK-based 空域注意力（可下采样 R）。"""
    def __init__(self, in_channels, heads=8, reduction=4, sr_ratio=1):
        super().__init__()
        self.heads = heads
        self.scale = (in_channels // heads) ** -0.5
        inner = in_channels // reduction

        self.q = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.kv = nn.Conv2d(in_channels, in_channels * 2, 1, bias=False)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(in_channels, in_channels, sr_ratio,
                                stride=sr_ratio, groups=in_channels, bias=False)
            self.norm = nn.LayerNorm(in_channels)

        self.proj = nn.Conv2d(in_channels, in_channels, 1, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, self.heads, C // self.heads, H * W).permute(0,1,3,2)  # B, h, HW, d

        # 可选下采样
        if self.sr_ratio > 1:
            x_ = self.sr(x)
            x_ = self.norm(x_.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, C, H//self.sr_ratio, W//self.sr_ratio)
        else:
            x_ = x
        k, v = self.kv(x_).chunk(2, dim=1)
        k = k.reshape(B, self.heads, C // self.heads, -1)      # B, h, d, HW'
        v = v.reshape(B, self.heads, C // self.heads, -1).permute(0,1,3,2)  # B, h, HW', d

        attn = (q @ k) * self.scale                             # B, h, HW, HW'
        attn = attn.softmax(dim=-1)

        out = attn @ v                                          # B, h, HW, d
        out = out.permute(0,1,3,2).reshape(B, C, H, W)
        out = self.proj(out)
        return out


# --------------- ③ FSDA：双域注意力主模块 ---------------- #
@MODELS.register_module()
class FSDAModule(nn.Module):
    """将 FourierAttention 与 SpatialAttention 并行 -> concat -> 1×1 fuse"""
    def __init__(self,
                 in_channels,
                 heads=8,
                 reduction=4,
                 sr_ratio=1,
                 fuse=True):
        super().__init__()
        self.fa = FourierAttention(in_channels)
        self.sa = SpatialAttention(in_channels, heads, reduction, sr_ratio)
        self.fuse = fuse
        if fuse:
            self.conv_fuse = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)
            self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        fa_out = self.fa(x)
        sa_out = self.sa(x)
        out = torch.cat([fa_out, sa_out], dim=1)   # (B, 2C, H, W)

        if self.fuse:
            out = self.bn(self.conv_fuse(out))
        return out

# --------------------- ④ 包装器模块 (新添加) ------------------- #
@MODELS.register_module()
class FSDAWrapperAttention(nn.Module):
    """
    一个包装器，用于将期望1D序列输入的Attention替换为期望2D特征图输入的FSDAModule。
    """
    def __init__(self,
                 embed_dims,
                 # 可以在这里添加FSDAModule需要的其他参数，比如heads, sr_ratio等
                 **kwargs): # 使用**kwargs接收MultiheadAttention的其他兼容性参数，但可能不会使用
        super().__init__()
        self.embed_dims = embed_dims
        
        # 在这里实例化你的FSDAModule
        self.fsda = FSDAModule(in_channels=embed_dims) # 假设heads等参数使用默认值

    def forward(self,
                query, # 输入形状: (B, Num_Queries, C)
                key=None,
                value=None,
                **kwargs):
        # 对于self-attention, query, key, value是相同的
        # 我们只使用query作为FSDAModule的输入
        
        B, N, C = query.shape
        
        # --- 关键步骤：1D序列 -> 2D特征图 ---
        # 由于Query不是天然的2D结构，我们需要给它一个假设的H和W
        # 最简单的方式是把它变成一个方形的特征图
        H = W = int(N ** 0.5)
        if H * W != N:
            # 如果Query数量不是完全平方数，这个简单的reshape会失败
            # 在这种情况下，可以采用补零等方式，但会增加复杂性
            # 对于Mask2Former的100个Query，我们可以假设一个10x10的结构
            assert H * W == N, f"The number of queries ({N}) must be a perfect square for simple reshaping."
        
        # (B, N, C) -> (B, C, N) -> (B, C, H, W)
        query_2d = query.permute(0, 2, 1).reshape(B, C, H, W)
        
        # --- 调用你的核心模块 ---
        output_2d = self.fsda(query_2d)
        
        # --- 关键步骤：2D特征图 -> 1D序列 ---
        # (B, C, H, W) -> (B, C, N) - > (B, N, C)
        output = output_2d.reshape(B, C, N).permute(0, 2, 1)
        
        # Transformer的Attention层通常需要返回两个值：(output, attention_weights)
        # 我们的模块没有显式的attention_weights，所以返回None
        return output