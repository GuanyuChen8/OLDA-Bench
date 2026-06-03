# projects/lk_exp/fsda_grad_check_hook.py
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmdet.registry import HOOKS
from projects.lk_exp.layers import FSDAWrapperAttention
import torch

@HOOKS.register_module()
class FSDAGradCheckHook(Hook):

    # 一定要把 cfg 里可能传进来的参数都写在 __init__ 里
    def __init__(self,
                 check_interval: int = 50,
                 nan_threshold: float = 0.1,
                 zero_threshold: float = 0.8,
                 print_details: bool = False,
                 priority: str = 'NORMAL',
                 **kwargs):          # ← 多余参数兜底
        super().__init__()          # 基类不接收额外参数
        self.priority = priority    # 手动挂属性

        self.check_interval = check_interval
        self.nan_th = nan_threshold
        self.zero_th = zero_threshold
        self.print_details = print_details

    def after_backward(self, runner, loss):
        iter_num = runner.iter + 1
        if iter_num % self.check_interval:
            return
        for name, module in runner.model.named_modules():
            if isinstance(module, FSDAWrapperAttention):
                self._check_grad(name, module, iter_num)

    def _check_grad(self, mod_name, module, iter_num):
        total = nan = zero = 0
        norms = []
        for p_name, p in module.named_parameters(recurse=True):
            if p.grad is None:
                continue
            g = p.grad
            total += g.numel()
            nan_mask = torch.isnan(g)
            nan += nan_mask.sum().item()
            valid = g[~nan_mask]
            if valid.numel():
                zero += (valid.abs() < 1e-8).sum().item()
                norms.append(valid.norm().item())

        if total == 0:
            return
        nan_r, zero_r = nan / total, zero / total
        avg_norm = sum(norms) / len(norms) if norms else 0
        status = '✓'
        warn = []
        if nan_r > self.nan_th:
            status = '⚠️'; warn.append(f'NaN {nan_r:.2%}')
        if zero_r > self.zero_th:
            status = '⚠️'; warn.append(f'Zero {zero_r:.2%}')
        if avg_norm < 1e-6:
            status = '⚠️'; warn.append(f'Norm {avg_norm:.1e}')

        print_log(f'[Iter {iter_num}] {status} {mod_name}: '
                  f'NaN={nan_r:.2%}, Zero={zero_r:.2%}, AvgNorm={avg_norm:.2e}',
                  'current')
        for w in warn:
            print_log(f'[Iter {iter_num}] ⚠️ {mod_name}: {w}', 'current')
