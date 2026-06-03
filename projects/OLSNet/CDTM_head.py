# projects/lk_exp/cdtm_head.py
from mmdet.registry import MODELS
from mmdet.models.dense_heads.mask2former_head import Mask2FormerHead
from projects.lk_exp.CDTM import FRTNeck  # FRTNeck（List[Tensor] -> List[Tensor]）

@MODELS.register_module()
class Mask2FormerHeadWithFRT(Mask2FormerHead):
    def __init__(self,
                 in_channels,                 # 必须显式接住，先保存
                 frt_cfg=None,                #  FRT/CDTM 超参
                 **kwargs):
        self._in_channels_for_frt = list(in_channels)  # super 之前缓存
        super().__init__(in_channels=in_channels, **kwargs)

        self.frt = None
        if frt_cfg is not None:
            # FRTNeck 要求与 head 接口一致：in_channels 的长度/顺序要与 backbone.out_indices 对齐
            self.frt = FRTNeck(in_channels=self._in_channels_for_frt, **frt_cfg)

    def forward(self, x, batch_data_samples):
        # x: List[Tensor], 每层(B,C,H,W)
        if self.frt is not None:
            x = self.frt(x)   # 形状保持 (B,C,H,W)，只做特征变换
        # 余下流程完全复用父类 —— pixel_decoder -> transformer_decoder -> head 计算
        return super().forward(x, batch_data_samples)
