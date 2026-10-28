"""按整回合指令执行质量和空间覆盖评价地形难度（仅依赖 torch）。

指令模块逐步累计移动任务及跟踪表现，课程判定前补采终止步：完成回合、
移动量和空间覆盖达标就升级；跟踪误差只用于未升级回合的降级判定。
零速、原地旋转和正常但覆盖不足的移动回合保持等级。
"""

from __future__ import annotations

import torch
from collections.abc import Sequence


class TerrainEpisodeMetrics:
    """各环境独立的回合统计；指令重采样不清零，episode reset 才清零。

    同一 episode 步号只采样一次。正常步由 CommandTerm 在重采样前采集，
    终止步由课程在机器人 reset 前采集，避免漏步、重计和跨回合瞬移。
    """

    def __init__(
        self, num_envs: int, device: str, min_command_speed: float = 0.1,
        lin_vel_tolerance: float = 0.25, yaw_tolerance: float = 0.5,
    ):
        if min_command_speed < 0 or lin_vel_tolerance <= 0 or yaw_tolerance <= 0:
            raise ValueError("指令速度下限必须非负，跟踪误差容限必须为正")
        self.min_command_speed = min_command_speed
        self.lin_vel_tolerance = lin_vel_tolerance
        self.yaw_tolerance = yaw_tolerance
        self._env_ids = torch.arange(num_envs, device=device)
        self.start_xy = torch.zeros(num_envs, 2, device=device)
        self.last_sample_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.command_distance = torch.zeros(num_envs, device=device)
        self.moving_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.good_tracking_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._step_dt = 0.0
        self.lin_error_integral = torch.zeros(num_envs, device=device)
        self.yaw_error_integral = torch.zeros(num_envs, device=device)
        self.max_excursion = torch.zeros(num_envs, device=device)
        self.max_terrain_distance = torch.zeros(num_envs, device=device)

    def reset(self, position_xy: torch.Tensor, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.start_xy[ids] = position_xy[ids]
        self.last_sample_step[ids] = 0
        for value in (
            self.command_distance, self.moving_steps, self.good_tracking_steps,
            self.lin_error_integral, self.yaw_error_integral,
            self.max_excursion, self.max_terrain_distance,
        ):
            value[ids] = 0.0

    def update(
        self, position_xy: torch.Tensor, terrain_origin_xy: torch.Tensor,
        lin_vel_xy: torch.Tensor, yaw_vel: torch.Tensor, command: torch.Tensor,
        episode_steps: torch.Tensor, dt: float, env_ids=None,
    ) -> None:
        self._step_dt = dt
        ids = self._env_ids if env_ids is None else self._env_ids[env_ids]
        fresh = episode_steps[ids] > self.last_sample_step[ids]
        cmd = command[ids]
        speed = torch.linalg.vector_norm(cmd[:, :2], dim=1)
        moving = fresh & torch.isfinite(cmd).all(dim=1) & (speed > self.min_command_speed)
        lin_error = torch.linalg.vector_norm(cmd[:, :2] - lin_vel_xy[ids], dim=1)
        yaw_error = torch.abs(cmd[:, 2] - yaw_vel[ids])
        good = moving & (lin_error <= self.lin_vel_tolerance) & (yaw_error <= self.yaw_tolerance)
        self.command_distance[ids] += torch.where(moving, speed * dt, 0.0)
        self.moving_steps[ids] += moving.long()
        self.good_tracking_steps[ids] += good.long()
        self.lin_error_integral[ids] += torch.where(
            moving, torch.nan_to_num(lin_error, nan=1e6, posinf=1e6) * dt, 0.0
        )
        self.yaw_error_integral[ids] += torch.where(
            moving, torch.nan_to_num(yaw_error, nan=1e6, posinf=1e6) * dt, 0.0
        )
        # 方形地形使用轴向最大距离，避免在中央方形平台的对角线内绕圈也被算作覆盖。
        valid_position = moving & torch.isfinite(position_xy[ids]).all(dim=1)
        excursion = (position_xy[ids] - self.start_xy[ids]).abs().amax(dim=1)
        terrain_distance = (position_xy[ids] - terrain_origin_xy[ids]).abs().amax(dim=1)
        self.max_excursion[ids] = torch.maximum(
            self.max_excursion[ids], torch.where(valid_position, excursion, 0.0)
        )
        self.max_terrain_distance[ids] = torch.maximum(
            self.max_terrain_distance[ids], torch.where(valid_position, terrain_distance, 0.0)
        )
        self.last_sample_step[ids] = episode_steps[ids]

    @property
    def moving_time(self) -> torch.Tensor:
        return self.moving_steps.float() * self._step_dt

    @property
    def good_tracking_fraction(self) -> torch.Tensor:
        return self.good_tracking_steps.float() / self.moving_steps.clamp_min(1)


def compute_terrain_level_moves(
    command_distance: torch.Tensor,
    moving_time: torch.Tensor,
    episode_time: torch.Tensor,
    good_tracking_fraction: torch.Tensor,
    max_excursion: torch.Tensor,
    max_terrain_distance: torch.Tensor,
    completed: torch.Tensor,
    failed: torch.Tensor,
    terrain_length: float,
    coverage_ratio: float = 0.25,
    min_command_distance: float = 1.0,
    min_tracking_duration_s: float = 2.0,
    min_moving_fraction: float = 0.5,
    failure_ratio: float = 0.5,
    allow_demotion: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """升级只要求完成、足够移动任务和实际空间覆盖，不检查跟踪误差。

    明确移动任务定义为：已执行的回合中至少 ``min_moving_fraction`` 的时间
    有平移指令。此时摔倒可以立即降级（不必等走满 1 m）；持续跟踪失败则还
    要有足够的指令路程和采样时长。升级优先于跟踪失败降级，避免误差容限
    间接成为升级门槛；失败终止仍禁止升级。未到评价门槛的正常回合保持等级。
    """
    if terrain_length <= 0 or coverage_ratio <= 0:
        raise ValueError("地形尺寸和覆盖比例必须为正")
    if min_command_distance <= 0 or min_tracking_duration_s <= 0:
        raise ValueError("最小指令路程和跟踪时长必须为正")
    if not 0 < min_moving_fraction <= 1:
        raise ValueError("移动时间占比必须在 (0, 1] 内")
    if not 0 < failure_ratio <= 1:
        raise ValueError("失败占比必须在 (0, 1] 内")
    lived = episode_time > 0
    moving_task = lived & (moving_time > 0) & (moving_time >= episode_time * min_moving_fraction)
    # 路程为浮点积分，容纳长回合累加的舍入误差；时间和跟踪占比由整数步数计算。
    enough_motion = (
        (command_distance >= min_command_distance * (1.0 - 1e-5))
        & (moving_time >= min_tracking_duration_s)
    )
    coverage_distance = terrain_length * coverage_ratio
    covered = (max_excursion >= coverage_distance) & (max_terrain_distance >= coverage_distance)
    move_up = lived & enough_motion & completed & ~failed & covered
    sustained_failure = enough_motion & (good_tracking_fraction <= 1.0 - failure_ratio)
    move_down = moving_task & (failed | sustained_failure) & allow_demotion & ~move_up
    return move_up, move_down


def terrain_levels_vel(
    env,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    coverage_ratio: float = 0.25,
    min_command_distance: float = 1.0,
    min_tracking_duration_s: float = 2.0,
    min_moving_fraction: float = 0.5,
    failure_ratio: float = 0.5,
    allow_demotion: bool = True,
    debug: bool = False,
) -> dict[str, float]:
    """地形课程项（CurriculumTermCfg.func）。

    依据本回合指令执行质量与覆盖更新 ``env_ids`` 中每个环境的地形等级，并维护
    逐等级的回合奖励/长度统计（读取 RewardManager 尚未清零的回合累计值，
    该课程在奖励管理器 reset 之前执行）。

    Returns:
        课程状态字典，由 CurriculumManager 记录为
        ``Curriculum/terrain_levels/<key>`` 标量。
    """
    terrain = env.scene.terrain
    if terrain.cfg.terrain_generator is None:
        raise ValueError("地形课程仅在 terrain_type='generator' 时可用")
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=terrain.device)[env_ids]
    else:
        env_ids = torch.as_tensor(env_ids, device=terrain.device, dtype=torch.long)

    invalid = getattr(env, "_numerical_invalid", None)
    if invalid is not None:
        env_ids = env_ids[~invalid[env_ids]]
    if env_ids.numel() == 0:
        return {}

    command_term = env.command_manager.get_term(command_name)
    metrics = command_term.terrain_metrics
    # ManagerBasedRLEnv 的 terminal step 在 CommandTerm.compute 前 reset，必须在这里补采。
    command_term.update_terrain_metrics(env_ids)
    episode_time = env.episode_length_buf[env_ids].float() * env.step_dt
    move_up, move_down = compute_terrain_level_moves(
        command_distance=metrics.command_distance[env_ids],
        moving_time=metrics.moving_time[env_ids],
        episode_time=episode_time,
        good_tracking_fraction=metrics.good_tracking_fraction[env_ids],
        max_excursion=metrics.max_excursion[env_ids],
        max_terrain_distance=metrics.max_terrain_distance[env_ids],
        completed=env.termination_manager.time_outs[env_ids],
        failed=env.termination_manager.terminated[env_ids],
        terrain_length=float(terrain.cfg.terrain_generator.size[0]),
        coverage_ratio=coverage_ratio,
        min_command_distance=min_command_distance,
        min_tracking_duration_s=min_tracking_duration_s,
        min_moving_fraction=min_moving_fraction,
        failure_ratio=failure_ratio,
        allow_demotion=allow_demotion,
    )
    # 首次内部 reset（环境构建期间）时机器人尚未经历任何仿真步，root 位置
    # 仍是默认值，距离无意义——跳过这些环境的课程更新。
    lived = env.episode_length_buf[env_ids] > 0
    move_up = move_up & lived
    move_down = move_down & lived
    # 回合发生在更新前的等级上：先记录再更新等级。
    levels_before = terrain.terrain_levels[env_ids].clone()
    # 等级 0 的降级请求不计入实际降级次数。
    move_down &= levels_before > 0
    # 更新地形等级（含边界裁剪与最高等级随机化，由 TerrainImporter 处理）。
    terrain.update_env_origins(env_ids, move_up, move_down)
    num_levels = int(terrain.max_terrain_level)
    stats = getattr(env, "terrain_curriculum_stats", None)
    if stats is None or stats["num_levels"] != num_levels:
        stats = {
            "num_levels": num_levels,
            "move_up_count": 0,
            "move_down_count": 0,
            "hold_count": 0,
            "reward_sum": torch.zeros(num_levels, device=terrain.device),
            "length_sum": torch.zeros(num_levels, device=terrain.device),
            "count": torch.zeros(num_levels, device=terrain.device),
        }
        env.terrain_curriculum_stats = stats
    stats["move_up_count"] += int(move_up.sum().item())
    stats["move_down_count"] += int(move_down.sum().item())
    hold = lived & ~(move_up | move_down)
    stats["hold_count"] += int(hold.sum().item())
    # 逐等级回合统计：课程先于奖励管理器 reset 执行，回合累计值仍然有效。
    episode_sums = getattr(env.reward_manager, "_episode_sums", None)
    if episode_sums is not None:
        episode_reward = torch.zeros_like(episode_time)
        for value in episode_sums.values():
            episode_reward += value[env_ids]
        stats["reward_sum"].index_add_(0, levels_before[lived].long(), episode_reward[lived])
        stats["length_sum"].index_add_(
            0, levels_before[lived].long(), env.episode_length_buf[env_ids][lived].float()
        )
        stats["count"].index_add_(0, levels_before[lived].long(), torch.ones_like(episode_time[lived]))

    result = {
        "mean_level": terrain.terrain_levels.float().mean().item(),
        "num_move_up": int(move_up.sum().item()),
        "num_move_down": int(move_down.sum().item()),
        "num_hold": int(hold.sum().item()),
        "num_reset_envs": int(env_ids.numel()),
    }
    if debug:
        # 逐等级环境计数与本次重置批次的回合统计。
        levels_now = terrain.terrain_levels
        max_level = int(terrain.max_terrain_level)
        for level in range(max_level):
            result[f"level_{level:02d}_envs"] = int((levels_now == level).sum().item())
        for name in ("command_distance", "moving_time", "max_excursion", "max_terrain_distance"):
            result[f"batch_mean_{name}"] = float(getattr(metrics, name)[env_ids].mean().item())
        result["batch_mean_tracking_fraction"] = float(metrics.good_tracking_fraction[env_ids].mean().item())
        moving_time = metrics.moving_time[env_ids].clamp_min(1e-6)
        result["batch_mean_lin_vel_error"] = float(
            (metrics.lin_error_integral[env_ids] / moving_time).mean().item()
        )
        result["batch_mean_yaw_error"] = float(
            (metrics.yaw_error_integral[env_ids] / moving_time).mean().item()
        )
    return result
