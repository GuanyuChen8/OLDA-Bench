
import types
from mmdet.models.backbones import SwinTransformer
from mmdet.registry import MODELS
from projects.lk_exp.layers import FSDAModule


_UNUSED_ATTRS = (
    # 旧注意力核心
    'qkv', 'q_proj', 'k_proj', 'v_proj',
    'attn_drop', 'softmax',
    # 相对位置编码
    'relative_position_bias_table',
    'relative_position_index',
    # 其它可能存在的权重 / buffer
    'logit_scale', 'dwconv'
)


@MODELS.register_module()
class SwinFSDA(SwinTransformer):
    """把 WindowMSA 完整替换为 FSDA，并删除未用权重"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        for stage in self.stages:                      # 4 个 stage
            for blk in stage.blocks:                   # SwinBlock
                msa = blk.attn.w_msa                   # WindowMSA
                C = msa.embed_dims

                # ---------- 注入 FSDA ----------
                msa.fsda = FSDAModule(
                    in_channels=C,
                    heads=msa.num_heads,
                    reduction=4,
                    sr_ratio=1)

                # ---------- 重写 forward ----------
                def fsda_forward(self, x, mask=None):
                    Bn, N, C = x.shape
                    H = W = int(N ** 0.5)

                    x2d = x.transpose(1, 2).reshape(Bn, C, H, W)
                    x_fsda = self.fsda(x2d)                          # FSDA
                    x_fsda = x_fsda.reshape(Bn, C, N).transpose(1, 2)

                    x_fsda = self.proj_drop(self.proj(x_fsda))       # proj+drop
                    return x + x_fsda                                # 残差

                msa.forward = types.MethodType(fsda_forward, msa)


