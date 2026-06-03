import torch
from torch import nn
from torch.nn import LayerNorm
import torch.nn.functional as F
from collections import OrderedDict
from typing import Callable
from torch.utils.checkpoint import checkpoint
from mmdet.registry import MODELS
from mmcv.cnn import build_activation_layer   # MMEngine/MMCV 常用工具

class Cluster_Attention(nn.Module):
    r"""低秩近似版本 (O(r·N) 内存)

    Args:
        dim (int): token 通道维度 C
        centers (int): 聚类中心个数
        rank (int): 低秩阶数 r（≪ N）
        proj_drop (float): dropout 概率
    """
    def __init__(self, dim,centers, rank=32, proj_drop=0.):
        super().__init__()
        self.rank = rank                     # <<< 保存 r 以便 forward 使用
        self.centers = nn.Parameter(torch.randn(dim, centers))
        self.U = nn.Parameter(torch.randn(centers, rank, 1))
        self.V = nn.Parameter(torch.randn(centers, 1, rank))

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Args:
            x: Tensor (B, N, C)
        Returns:
            Tensor (B, N, C)
        """
        B, N, C = x.shape
        r = self.rank                      # <<< 取出 r

        # 1) 图级 cluster 权重 α (B, c)
        q = F.normalize(x.mean(1), dim=-1)           # (B, C)
        k = F.normalize(self.centers, dim=0)         # (C, c)
        attn = (q @ k).softmax(-1)                   # (B, c)

        # 2) 低秩矩阵 (B, r, r)
        base = torch.matmul(self.U, self.V)          # (c, r, r)
        tm   = (attn.unsqueeze(-1).unsqueeze(-1) * base).sum(1)  # (B, r, r)

        # 3) 将 token 维映射到 r 个“超 token”
        assert C % r == 0, \
            f"C ({C}) 必须能整除 rank ({r})，否则请改用线性投影方式"
        x_r = x.reshape(B, N, r, C // r).mean(-1)  # (B, N, r)
        out_r = torch.bmm(tm, x_r.transpose(1, 2))   # (B, r, N)
        out   = out_r.transpose(1, 2)                # (B, N, r)

        # 4) 线性映射回原通道维度
        out = self.proj_drop(self.proj(out))
        return out

# ---------------- Cluster_Attention 已经是低秩版本 ----------------
# 下面是“减→注意力→升” 的 FR_Resblock + FRT + FRTNeck

class FR_Resblock(nn.Module):
    def __init__(self,
                 d_model: int,
                 centers: int,
                 rank: int = 32,
                 mlp_ratio: float = 4.0,
                 act_cfg: dict = dict(type='GELU')):
        super().__init__()
        act_layer = build_activation_layer(act_cfg)

        # 1) 通道降维 C -> r
        self.reduce = nn.Linear(d_model, rank, bias=False)

        # 2) Cluster Attention 在 r 维上跑
        self.ln_1 = LayerNorm(rank)
        self.attn = Cluster_Attention(rank, centers, rank=rank)

        # 3) 通道升维 r -> C
        self.expand = nn.Linear(rank, d_model, bias=False)

        # 4) MLP
        self.ln_2 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ('c_fc',   nn.Linear(d_model, mlp_width)),
            ('act',    act_layer),
            ('c_proj', nn.Linear(mlp_width, d_model))
        ]))

    def forward(self, x):
        # x: (B,N,C)
        x_reduced = self.reduce(x)              # (B,N,r)
        x_att     = self.attn(self.ln_1(x_reduced))
        x_proj    = self.expand(x_att)          # (B,N,C)

        x = x_proj + self.mlp(self.ln_2(x_proj))
        return x

# ---------------- FRT ----------------
@MODELS.register_module()
class FRT(nn.Module):
    def __init__(self,
                 check_point: list,
                 width: int,
                 centers: int,
                 dt_layers: int,
                 rank: int = 32,
                 mlp_ratio: float = 4.0,
                 act_cfg: dict = dict(type='GELU')):
        super().__init__()
        self.check_point = check_point
        self.blocks = nn.ModuleList([
            FR_Resblock(width, centers, rank,
                        mlp_ratio=mlp_ratio, act_cfg=act_cfg)
            for _ in range(dt_layers)
        ])

    def forward(self, x):
        for idx, blk in enumerate(self.blocks):
            if self.check_point[0] and idx < self.check_point[1]:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x

# ---------------- FRTNeck ----------------
@MODELS.register_module()
class FRTNeck(nn.Module):
    def __init__(self,
                 in_channels,     # e.g. [512,1024]
                 centers=4,
                 dt_layers=3,
                 rank=32,
                 mlp_ratio=4.0,
                 act_cfg=dict(type='GELU'),
                 check_point=[False, 0]):
        super().__init__()
        self.processes = nn.ModuleList()
        for c in in_channels:
            self.processes.append(
                FRT(check_point, c, centers,
                    dt_layers, rank, mlp_ratio, act_cfg))
        self.out_channels = in_channels

    def forward(self, feats):          # feats: (stage2, stage3)
        outs = []
        for x, frt in zip(feats, self.processes):
            B,C,H,W = x.shape
            x = x.flatten(2).transpose(1,2)   # (B,N,C)
            x = frt(x)
            x = x.transpose(1,2).reshape(B,C,H,W)
            outs.append(x)
        return tuple(outs)


def test_frt_io():
    # ---- 手动配置一组超参数 ----
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