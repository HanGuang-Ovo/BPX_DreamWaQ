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
4. 常规分布混合激活单元均匀分布与全合法单元探索，再与已激活区域内的
   零速附近分布混合；导出的概率包含这两类采样；
5. 前向速度与偏航角速度始终从同一单元内连续采样，裁到任务量程；
   低速分支只使用单元与低速区域的交集，统计仍归实际所属单元。
"""

from __future__ import annotations

import math
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
        continuous_sampling: bool = True,
        low_speed_sampling_prob: float = 0.3,
        low_speed_vx_range: Tuple[float, float] = (-0.25, 0.25),
        low_speed_yaw_range: Tuple[float, float] = (-0.25, 0.25),
    ):
        if not all(math.isfinite(r) and r > 0 for r in (vx_resolution, yaw_resolution)):
            raise ValueError("网格分辨率必须为有限正数")
        if not 0.0 <= exploration_prob < 1.0:
            raise ValueError("exploration_prob 必须在 [0, 1) 内")
        if min_samples < 1:
            raise ValueError("min_samples 至少为 1")
        if not 0.5 <= success_ratio_threshold <= 1.0:
            raise ValueError("success_ratio_threshold 必须在 [0.5, 1.0] 内")
        if not 0.0 <= low_speed_sampling_prob <= 1.0:
            raise ValueError("low_speed_sampling_prob 必须在 [0, 1] 内")
        if (
            not math.isfinite(command_jitter)
            or not 0 <= command_jitter <= min(vx_resolution, yaw_resolution) / 2
        ):
            raise ValueError("command_jitter 必须为有限非负数，且不超过任一轴的半格宽")
        for name, bounds in (
            ("max_vx_range", max_vx_range), ("max_yaw_range", max_yaw_range),
            ("initial_vx_range", initial_vx_range), ("initial_yaw_range", initial_yaw_range),
            ("low_speed_vx_range", low_speed_vx_range), ("low_speed_yaw_range", low_speed_yaw_range),
        ):
            if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or bounds[0] >= bounds[1]:
                raise ValueError(f"{name} 必须为有限且递增的区间")
        self.vx_resolution = vx_resolution
        self.yaw_resolution = yaw_resolution
        self.success_threshold_vx = success_threshold_vx
        self.success_threshold_yaw = success_threshold_yaw
        self.success_ratio_threshold = success_ratio_threshold
        self.min_samples = min_samples
        self.ema_alpha = ema_alpha
        self.exploration_prob = exploration_prob
        self.command_jitter = command_jitter
        self.continuous_sampling = continuous_sampling
        self.low_speed_sampling_prob = low_speed_sampling_prob
        self.low_speed_vx_range = low_speed_vx_range
        self.low_speed_yaw_range = low_speed_yaw_range
        self.device = device

        def centers(res: float, max_range: Tuple[float, float]) -> torch.Tensor:
            # 网格中心取分辨率的整数倍，限制在合法最大范围内。
            k_max = math.floor(max_range[1] / res)
            k_min = math.ceil(max_range[0] / res)
            if k_min > k_max:
                raise ValueError("任务量程内至少需要一个网格中心")
            return torch.arange(k_min, k_max + 1, dtype=torch.float32, device=device) * res

        self.vx_centers = centers(vx_resolution, max_vx_range)
        self.yaw_centers = centers(yaw_resolution, max_yaw_range)
        self.num_vx = self.vx_centers.numel()
        self.num_yaw = self.yaw_centers.numel()
        self.num_cells = self.num_vx * self.num_yaw

        def sampling_bounds(values, resolution, limits, low_speed_limits):
            half_width = resolution / 2 if continuous_sampling else command_jitter
            lower = (values - half_width).clamp_min(limits[0])
            upper = (values + half_width).clamp_max(limits[1])
            # 连续模式的两端单元覆盖非整格量程剩余部分，仍保持各格互不重叠。
            if continuous_sampling:
                lower[0], upper[-1] = limits[0], limits[1]
            low_lower = lower.clamp_min(low_speed_limits[0])
            low_upper = upper.clamp_max(low_speed_limits[1])
            overlap = (low_upper - low_lower).clamp_min(0)
            # 离散兼容模式中，中心点位于低速区内也可被加采样。
            overlap = torch.where(
                (lower == upper) & (low_lower <= low_upper), torch.ones_like(overlap), overlap
            )
            return lower, upper, low_lower, low_upper, overlap

        (self._vx_lower, self._vx_upper, self._low_vx_lower,
         self._low_vx_upper, vx_overlap) = sampling_bounds(
            self.vx_centers, vx_resolution, max_vx_range, low_speed_vx_range)
        (self._yaw_lower, self._yaw_upper, self._low_yaw_lower,
         self._low_yaw_upper, yaw_overlap) = sampling_bounds(
            self.yaw_centers, yaw_resolution, max_yaw_range, low_speed_yaw_range)
        self._low_speed_area = (vx_overlap[:, None] * yaw_overlap[None, :]).flatten()

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
        """激活单元可采样的最大 |vx|，已裁到任务量程。"""
        idx = self.activated.any(dim=1).nonzero().flatten()
        if not idx.numel():
            return 0.0
        return float(torch.maximum(self._vx_lower[idx].abs(), self._vx_upper[idx].abs()).max())

    @property
    def active_yaw_range(self) -> float:
        idx = self.activated.any(dim=0).nonzero().flatten()
        if not idx.numel():
            return 0.0
        return float(torch.maximum(self._yaw_lower[idx].abs(), self._yaw_upper[idx].abs()).max())

    """
    采样与统计
    """

    def _update_probs(self):
        """维护常规课程、已激活低速区域，以及导出用的混合边际概率。"""
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
        self._regular_probs = probs / probs.sum()
        low_weights = self._low_speed_area * activated
        low_total = low_weights.sum()
        self._low_speed_prob = self.low_speed_sampling_prob if low_total > 0 else 0.0
        self._low_speed_probs = low_weights / low_total if low_total > 0 else self._regular_probs
        self._probs = (
            (1.0 - self._low_speed_prob) * self._regular_probs
            + self._low_speed_prob * self._low_speed_probs
        )

    def sample(self, num: int, generator: torch.Generator | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从联合分布采样 ``num`` 个指令组合。

        Returns:
            (cell_idx, vx, wz)：实际所属的展平单元索引与格内指令。
        """
        cell_idx, vx, wz, _ = self.sample_with_low_speed_mask(num, generator=generator)
        return cell_idx, vx, wz

    def sample_with_low_speed_mask(
        self, num: int, generator: torch.Generator | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """额外返回低速分支掩码，供指令生成器同步收窄 vy 的采样范围。"""
        cell_idx = torch.multinomial(self._regular_probs, num, replacement=True, generator=generator)
        low_speed = torch.zeros(num, dtype=torch.bool, device=self.device)
        if self._low_speed_prob > 0:
            low_speed = torch.rand(num, device=self.device, generator=generator) < self._low_speed_prob
            low_cells = torch.multinomial(self._low_speed_probs, num, replacement=True, generator=generator)
            cell_idx = torch.where(low_speed, low_cells, cell_idx)
        vx_idx = torch.div(cell_idx, self.num_yaw, rounding_mode="floor")
        yaw_idx = cell_idx % self.num_yaw

        def within_cell(index, lower, upper, low_lower, low_upper):
            lo = torch.where(low_speed, low_lower[index], lower[index])
            hi = torch.where(low_speed, low_upper[index], upper[index])
            return lo + torch.rand(num, device=self.device, generator=generator) * (hi - lo)

        vx = within_cell(vx_idx, self._vx_lower, self._vx_upper, self._low_vx_lower, self._low_vx_upper)
        wz = within_cell(yaw_idx, self._yaw_lower, self._yaw_upper, self._low_yaw_lower, self._low_yaw_upper)
        return cell_idx, vx, wz, low_speed

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
