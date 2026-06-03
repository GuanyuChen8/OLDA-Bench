import torch
from torch import nn
from torch.nn import LayerNorm
import torch.nn.functional as F
from collections import OrderedDict
from typing import Callable
from torch.utils.checkpoint import checkpoint
from mmdet.registry import MODELS
from mmcv.cnn import build_activation_layer   

class Cluster_Attention(nn.Module):
    """O(N) 内存版本：只为每个 token 生成一个缩放系数。"""
    def __init__(self, dim, len_token,centers,  proj_drop=0.):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(dim, centers))
        # 只学一个标量因子，而非 N×N
        self.tran_vec = nn.Parameter(torch.randn(centers, 1))  
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape                            # x: (B,N,C)

        # 1) 计算 cluster attention (B,centers)
        q = F.normalize(x.mean(1), dim=-1)           # (B,C)
        k = F.normalize(self.centers, dim=0)         # (C,centers)
        attn = (q @ k).softmax(-1)                   # (B,centers)

        # 2) 得到 token 级 weight 向量 —— O(N)
        #    每个中心只有一个可学习系数，再用 attn 混合成 (B,N)
        token_weight = attn @ self.tran_vec          # (B,centers)·(centers,1) -> (B,1)
        token_weight = token_weight.expand(-1, N)    # (B,N)

        # 3) 逐 token 缩放，然后线性映射
        x = x * token_weight.unsqueeze(-1)           # (B,N,C)
        return self.proj_drop(self.proj(x))


class FR_Resblock(nn.Module):
    def __init__(self,
                 d_model: int,
                 len_token: int,
                 centers: int,
                 mlp_ratio: float = 4.0,
                 act_cfg: dict = dict(type='GELU')):     # <-- 接收 act_cfg
        super().__init__()
        act_layer = build_activation_layer(act_cfg)      # <-- 转成 nn.Module

        self.ln_1 = LayerNorm(d_model)
        self.attn = Cluster_Attention(d_model, len_token, centers)
        self.ln_2 = LayerNorm(d_model)

        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ('c_fc',   nn.Linear(d_model, mlp_width)),
            ('act',    act_layer),                       # 用转换后的激活
            ('c_proj', nn.Linear(mlp_width, d_model))
        ]))

    def forward(self, x: torch.Tensor):
        x = self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@MODELS.register_module()
class FRT(nn.Module):
    def __init__(self,
                 check_point: list,
                 width: int,
                 len_token: int,
                 centers: int,
                 dt_layers: int,
                 mlp_ratio: float = 4.0,
                 act_cfg: dict = dict(type='GELU')):   # <-- act_cfg
        super().__init__()
        self.check_point = check_point
        self.da_resblocks = nn.ModuleList([
            FR_Resblock(width, len_token, centers,
                        mlp_ratio=mlp_ratio, act_cfg=act_cfg)
            for _ in range(dt_layers)
        ])

    def forward(self, x: torch.Tensor):
        if self.check_point[0] and not torch.jit.is_scripting():
            for idx, dr in enumerate(self.da_resblocks):
                if idx<self.check_point[1]:
                    x = checkpoint(dr, x, use_reentrant=False)
                else:
                    x = dr(x)
        else:
            for dr in self.da_resblocks:
                x = dr(x)
        return x

# ----------------------------- neck.py -----------------------------
@MODELS.register_module()
class FRTNeck(nn.Module):
    """Wrap 多尺度 FRT，保持通道数不变."""
    def __init__(self,
                 in_channels,         # list[int]  e.g. [128, 256, 512, 1024]
                 len_tokens,          # 考虑显存原因，没用到，已经删了。 list[int]  e.g. [3136, 784, 196, 49]
                 centers=4,
                 dt_layers=3,
                 mlp_ratio=4.0,
                 act_cfg=dict(type='GELU'),
                 check_point=[False, 0]):
        super().__init__()
        assert len(in_channels) == len(len_tokens)
        self.processes = nn.ModuleList()
        for c, l in zip(in_channels, len_tokens):
            self.processes.append(
                FRT(check_point=check_point,
                    width=c,
                    len_token=l,
                    centers=centers,
                    dt_layers=dt_layers,
                    mlp_ratio=mlp_ratio,
                    act_cfg=act_cfg))   # act_cfg 版本
        self.out_channels = in_channels   # Pixel-Decoder 需要这个属性

    def forward(self, feats):
        # feats: list[Tensor]，4 个尺度 [B,Ci,Hi,Wi]
        outs = []
        for x, frt in zip(feats, self.processes):
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)     # B,H*W,C
            x = frt(x)                           # B,H*W,C
            x = x.transpose(1, 2).reshape(B, C, H, W)
            outs.append(x)
        return tuple(outs)


def test_frt_io():
    # ---- 超参数 ----
    B, N, C      = 2, 196, 256     # batch，token 数，通道数 (= width)
    centers      = 4               # 聚类中心个数
    dt_layers    = 3               # FR_Resblock 堆叠层数
    checkpoint_k = [False, 0]      # 不启用 gradient checkpoint（可随意改）

    # ---- 构建网络 ----
    frt = FRT(check_point=checkpoint_k,
              width=C,
              len_token=N,
              centers=centers,
              dt_layers=dt_layers).eval()     # 只做推断，eval 即可

    # ---- 随机生成输入 ----
    x = torch.randn(B, N, C)

    # ---- 前向传播 ----
    with torch.no_grad():
        y = frt(x)

    # ---- 打印 / 验证 ----
    print(f"Input  shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == x.shape, \
        f"Shape mismatch! in={x.shape}, out={y.shape}"
    print("✔ I/O shapes match – test passed.")

if __name__ == "__main__":
    test_frt_io()