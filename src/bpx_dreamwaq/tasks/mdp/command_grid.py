"""Grid-Adaptive 运动指令课程核心调度器（模块 B，纯 torch，可离线测试）。

参考《Rapid Locomotion via Reinforcement Learning》(Lee et al., 2020) 的
Grid-Adaptive Curriculum 思想：在前向速度 ``vx`` 与偏航角速度 ``wz`` 的联合
二维网格上维护每个指令组合（网格单元）的采样概率与跟踪表现统计；只有当某
单元的跟踪表现达到成功阈值且积累了足够样本时，才激活其相邻单元，从而以
耦合方式逐步扩展可学习的指令区域。

与论文原始概率更新的差异（论文未公开完整实现细节，此处为明确记录的近似规则）：

1. 每个单元维护采样次数、区间平均跟踪误差的 EMA 与区间成功率的 EMA；
2. 成功判定为「一个指令执行区间的平均 |Δvx| 与 |Δwz| 同时低于各自阈值」；
3. 单元达标（成功 EMA ≥ 阈值且采样次数 ≥ min_samples）时激活其 4 邻域内
   合法单元，激活单调递增（不撤销）；
4. 采样分布在激活单元上均匀分布，并与「全体合法单元均匀分布」按
   ``exploration_prob`` 混合，保留最低探索概率；
5. 前向速度与偏航角速度始终从联合分布（同一单元）采样，不独立组合。
"""

from __future__ import annotations

from typing import Tuple

import torch


class GridAdaptiveCommandScheduler:
    """(vx, wz) 联合指令网格的自适应课程调度器。"""

    def __init__(
        self,
        vx_resolution: float = 0.5,
        yaw_resolution: float = 0.5,
        initial_vx_range: Tuple[float, float] = (-0.5, 0.5),
        initial_yaw_range: Tuple[float, float] = (-0.5, 0.5),
        max_vx_range: Tuple[float, float] = (-1.0, 1.0),
        max_yaw_range: Tuple[float, float] = (-1.0, 1.0),
        success_threshold_vx: float = 0.25,
        success_threshold_yaw: float = 0.5,
        success_ratio_threshold: float = 0.9,
        min_samples: int = 5,
        ema_alpha: float = 0.7,
        exploration_prob: float = 0.1,
        command_jitter: float = 0.0,
        device: str = "cpu",
    ):
        if vx_resolution <= 0 or yaw_resolution <= 0:
            raise ValueError("网格分辨率必须为正")
        if not 0.0 <= exploration_prob < 1.0:
            raise ValueError("exploration_prob 必须在 [0, 1) 内")
        if min_samples < 1:
            raise ValueError("min_samples 至少为 1")
        if not 0.5 <= success_ratio_threshold <= 1.0:
            raise ValueError("success_ratio_threshold 必须在 [0.5, 1.0] 内")
        self.vx_resolution = vx_resolution
        self.yaw_resolution = yaw_resolution
        self.success_threshold_vx = success_threshold_vx
        self.success_threshold_yaw = success_threshold_yaw
        self.success_ratio_threshold = success_ratio_threshold
        self.min_samples = min_samples
        self.ema_alpha = ema_alpha
        self.exploration_prob = exploration_prob
        self.command_jitter = command_jitter
        self.device = device

        def centers(res: float, max_range: Tuple[float, float]) -> torch.Tensor:
            # 网格中心取分辨率的整数倍，限制在合法最大范围内。
            k_max = int(torch.floor(torch.tensor(max_range[1]) / res).item())
            k_min = -int(torch.floor(torch.tensor(abs(max_range[0])) / res).item())
            return torch.arange(k_min, k_max + 1, dtype=torch.float32, device=device) * res

        self.vx_centers = centers(vx_resolution, max_vx_range)
        self.yaw_centers = centers(yaw_resolution, max_yaw_range)
        self.num_vx = self.vx_centers.numel()
        self.num_yaw = self.yaw_centers.numel()
        self.num_cells = self.num_vx * self.num_yaw

        # 初始激活：中心落在初始范围内的单元。
        def initial_mask(centers: torch.Tensor, rng: Tuple[float, float]) -> torch.Tensor:
            return (centers >= rng[0] - 1e-6) & (centers <= rng[1] + 1e-6)

        vx_mask = initial_mask(self.vx_centers, initial_vx_range)
        yaw_mask = initial_mask(self.yaw_centers, initial_yaw_range)
        self.activated = vx_mask[:, None] & yaw_mask[None, :]

        # 逐单元统计。
        self.samples = torch.zeros(self.num_cells, dtype=torch.long, device=device)
        self.err_vx_ema = torch.zeros(self.num_cells, device=device)
        self.err_yaw_ema = torch.zeros(self.num_cells, device=device)
        self.success_ema = torch.zeros(self.num_cells, device=device)
        # 是否已达标（避免重复扩展计数）。
        self.graduated = torch.zeros(self.num_cells, dtype=torch.bool, device=device)
        self.expansion_count = 0

        self._update_probs()

    """
    属性
    """

    @property
    def probs(self) -> torch.Tensor:
        """展平后的采样概率分布（形状 num_cells，和为 1）。"""
        return self._probs

    @property
    def num_activated(self) -> int:
        return int(self.activated.sum().item())

    @property
    def active_vx_range(self) -> float:
        """激活单元覆盖的最大 |vx|（含半格宽）。"""
        idx = self.activated.any(dim=1).nonzero().flatten()
        return float(self.vx_centers[idx].abs().max().item() + self.vx_resolution / 2) if idx.numel() else 0.0

    @property
    def active_yaw_range(self) -> float:
        idx = self.activated.any(dim=0).nonzero().flatten()
        return float(self.yaw_centers[idx].abs().max().item() + self.yaw_resolution / 2) if idx.numel() else 0.0

    """
    采样与统计
    """

    def _update_probs(self):
        """归一化采样概率：激活单元均匀分布 + 全合法单元探索混合。"""
        activated = self.activated.flatten().float()
        num_active = activated.sum().clamp_min(1.0)
        num_legal = float(self.num_cells)
        active_uniform = activated / num_active
        legal_uniform = torch.full_like(active_uniform, 1.0 / num_legal)
        probs = (1.0 - self.exploration_prob) * active_uniform + self.exploration_prob * legal_uniform
        # 数值保护：空网格时退化为全合法均匀分布（激活集合至少包含初始单元，
        # 此分支仅在异常构造时触发）。
        if not torch.isfinite(probs).all() or probs.sum() <= 0:
            probs = legal_uniform
        self._probs = probs / probs.sum()

    def sample(self, num: int, generator: torch.Generator | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从联合分布采样 ``num`` 个指令组合。

        Returns:
            (cell_idx, vx, wz)：展平单元索引与对应指令值（可选加均匀抖动）。
        """
        cell_idx = torch.multinomial(self._probs, num, replacement=True, generator=generator)
        vx_idx = torch.div(cell_idx, self.num_yaw, rounding_mode="floor")
        yaw_idx = cell_idx % self.num_yaw
        vx = self.vx_centers[vx_idx].clone()
        wz = self.yaw_centers[yaw_idx].clone()
        if self.command_jitter > 0:
            vx += (torch.rand(num, device=self.device) * 2.0 - 1.0) * self.command_jitter
            wz += (torch.rand(num, device=self.device) * 2.0 - 1.0) * self.command_jitter
        return cell_idx, vx, wz

    def record_interval(
        self,
        cell_idx: torch.Tensor,
        mean_err_vx: torch.Tensor,
        mean_err_yaw: torch.Tensor,
    ) -> int:
        """记录一个指令执行区间的跟踪表现，返回本次触发的网格扩展次数。

        Args:
            cell_idx: (N,) 该区间使用的单元索引。
            mean_err_vx: (N,) 区间平均 |vx_cmd - vx|（m/s）。
            mean_err_yaw: (N,) 区间平均 |wz_cmd - wz|（rad/s）。
        """
        if cell_idx.numel() == 0:
            return 0
        err_vx = mean_err_vx.detach().to(self.device).reshape(-1)
        err_yaw = mean_err_yaw.detach().to(self.device).reshape(-1)
        idx = cell_idx.detach().to(self.device).reshape(-1).long()
        success = (err_vx < self.success_threshold_vx) & (err_yaw < self.success_threshold_yaw)

        alpha = self.ema_alpha
        num_cells = self.num_cells
        count = torch.bincount(idx, minlength=num_cells).float()
        sum_vx = torch.zeros(num_cells, device=self.device).index_add_(0, idx, err_vx)
        sum_yaw = torch.zeros(num_cells, device=self.device).index_add_(0, idx, err_yaw)
        sum_success = torch.zeros(num_cells, device=self.device).index_add_(0, idx, success.float())
        touched = count > 0
        batch_vx = torch.where(touched, sum_vx / count.clamp_min(1.0), self.err_vx_ema)
        batch_yaw = torch.where(touched, sum_yaw / count.clamp_min(1.0), self.err_yaw_ema)
        batch_success = torch.where(touched, sum_success / count.clamp_min(1.0), self.success_ema)
        self.samples += touched.long()
        self.err_vx_ema = torch.where(touched, alpha * self.err_vx_ema + (1 - alpha) * batch_vx, self.err_vx_ema)
        self.err_yaw_ema = torch.where(touched, alpha * self.err_yaw_ema + (1 - alpha) * batch_yaw, self.err_yaw_ema)
        self.success_ema = torch.where(
            touched, alpha * self.success_ema + (1 - alpha) * batch_success, self.success_ema
        )
        return self._maybe_expand(touched)

    def _maybe_expand(self, touched: torch.Tensor) -> int:
        """达标单元激活 4 邻域内合法但未激活的单元。"""
        ready = (
            touched
            & (self.samples >= self.min_samples)
            & (self.success_ema >= self.success_ratio_threshold)
            & ~self.graduated
        )
        if not ready.any():
            return 0
        self.graduated |= ready
        grid_ready = ready.reshape(self.num_vx, self.num_yaw)
        # 4 邻域扩张（通过移位实现）。
        expanded = self.activated.clone()
        expanded[1:, :] |= grid_ready[:-1, :]
        expanded[:-1, :] |= grid_ready[1:, :]
        expanded[:, 1:] |= grid_ready[:, :-1]
        expanded[:, :-1] |= grid_ready[:, 1:]
        newly = expanded & ~self.activated
        num_new = int(newly.sum().item())
        if num_new > 0:
            self.activated = expanded
            self.expansion_count += num_new
            self._update_probs()
        return num_new

    def cell_prob(self, cell_idx: torch.Tensor) -> torch.Tensor:
        """查询单元的当前采样概率（用于日志/导出）。"""
        return self._probs[cell_idx]
