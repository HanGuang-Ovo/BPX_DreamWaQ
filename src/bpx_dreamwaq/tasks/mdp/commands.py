"""Grid-Adaptive 速度指令生成器（模块 B 的 Isaac Lab 接入层）。

在 ``UniformVelocityCommand`` 的语义上替换 ``vx``/``wz`` 的采样方式：

- ``vx`` 与 ``wz`` 从 ``GridAdaptiveCommandScheduler`` 的联合网格分布采样
  （同一单元），绝不独立采样后组合；
- ``vy`` 保持原有均匀采样，不受课程影响；
- 站立环境（``rel_standing_envs`` 概率）指令置零，与原实现一致；
- 每个指令执行区间（episode 内重采样周期或 episode 结束）结束时，把该区间
  的平均跟踪误差写回对应网格单元，驱动课程扩展——不会把整个 episode 的
  表现错误地归到最后一格。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .command_grid import GridAdaptiveCommandScheduler


class GridAdaptiveVelocityCommand(CommandTerm):
    """vx-wz 联合网格课程 + vy 均匀采样的速度指令生成器。"""

    cfg: GridAdaptiveVelocityCommandCfg

    def __init__(self, cfg: GridAdaptiveVelocityCommandCfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[self.cfg.asset_name]
        self.grid = GridAdaptiveCommandScheduler(
            vx_resolution=self.cfg.vx_resolution,
            yaw_resolution=self.cfg.yaw_resolution,
            initial_vx_range=self.cfg.initial_vx_range,
            initial_yaw_range=self.cfg.initial_yaw_range,
            max_vx_range=self.cfg.max_vx_range,
            max_yaw_range=self.cfg.max_yaw_range,
            success_threshold_vx=self.cfg.success_threshold_vx,
            success_threshold_yaw=self.cfg.success_threshold_yaw,
            success_ratio_threshold=self.cfg.success_ratio_threshold,
            min_samples=self.cfg.min_samples,
            ema_alpha=self.cfg.ema_alpha,
            exploration_prob=self.cfg.exploration_prob,
            command_jitter=self.cfg.command_jitter,
            device=self.device,
        )
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 每个环境当前使用的网格单元与区间累计误差。
        self.cell_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.interval_err_vx = torch.zeros(self.num_envs, device=self.device)
        self.interval_err_yaw = torch.zeros(self.num_envs, device=self.device)
        self.interval_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # 兼容 UniformVelocityCommand 的指标（per-env 累计，reset 时取均值清零）。
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """机体系速度指令 (num_envs, 3)：[vx, vy, wz]。"""
        return self.vel_command_b

    def _update_metrics(self):
        err_vx = torch.abs(self.vel_command_b[:, 0] - self.robot.data.root_lin_vel_b[:, 0])
        err_yaw = torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        # 与 UniformVelocityCommand 一致：按区间时长归一化的累计误差。
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["error_vel_xy"] += torch.norm(
            self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1
        ) / max_command_step
        self.metrics["error_vel_yaw"] += err_yaw / max_command_step
        # 课程区间累计（站立环境不参与课程统计：零指令无法判定跟踪能力）。
        active = ~self.is_standing_env
        self.interval_err_vx += err_vx * active
        self.interval_err_yaw += err_yaw * active
        self.interval_steps += active.long()

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            env_ids_t = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device)
        # 1) 结算刚结束的指令区间（episode 内重采样或 episode 结束）。
        if self.interval_steps.numel() > 0 and int(self.interval_steps.sum().item()) > 0:
            finished = env_ids_t[self.interval_steps[env_ids_t] > 0]
            if finished.numel() > 0:
                steps = self.interval_steps[finished].float()
                self.grid.record_interval(
                    cell_idx=self.cell_idx[finished],
                    mean_err_vx=self.interval_err_vx[finished] / steps,
                    mean_err_yaw=self.interval_err_yaw[finished] / steps,
                )
        # 2) 从联合网格分布采样新指令。
        cell_idx, vx, wz = self.grid.sample(env_ids_t.numel())
        self.cell_idx[env_ids_t] = cell_idx.to(self.device)
        r = torch.empty(env_ids_t.numel(), device=self.device)
        self.vel_command_b[env_ids_t, 0] = vx.to(self.device)
        self.vel_command_b[env_ids_t, 1] = r.uniform_(*self.cfg.lin_vel_y_range)
        self.vel_command_b[env_ids_t, 2] = wz.to(self.device)
        # 3) 重置区间累计。
        self.interval_err_vx[env_ids_t] = 0.0
        self.interval_err_yaw[env_ids_t] = 0.0
        self.interval_steps[env_ids_t] = 0
        # 4) 站立环境标记。
        self.is_standing_env[env_ids_t] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")


# 注意：配置类必须位于 CommandTerm 类之后定义（class_type 默认值引用该类）。


@configclass
class GridAdaptiveVelocityCommandCfg(CommandTermCfg):
    """Grid-Adaptive 指令课程配置。"""

    class_type: type = GridAdaptiveVelocityCommand  # type: ignore[assignment]

    asset_name: str = "robot"
    """场景中的机器人名称。"""

    rel_standing_envs: float = 0.02
    """站立（零指令）环境比例。"""

    lin_vel_y_range: tuple[float, float] = (-0.5, 0.5)
    """侧向速度指令范围（保持原有采样方式，不参与课程）。"""

    # ---- 课程网格参数 ----
    vx_resolution: float = 0.5
    yaw_resolution: float = 0.5
    initial_vx_range: tuple[float, float] = (-0.5, 0.5)
    initial_yaw_range: tuple[float, float] = (-0.5, 0.5)
    max_vx_range: tuple[float, float] = (-1.0, 1.0)
    max_yaw_range: tuple[float, float] = (-1.0, 1.0)
    success_threshold_vx: float = 0.25
    success_threshold_yaw: float = 0.5
    success_ratio_threshold: float = 0.9
    min_samples: int = 5
    ema_alpha: float = 0.7
    exploration_prob: float = 0.1
    command_jitter: float = 0.0
