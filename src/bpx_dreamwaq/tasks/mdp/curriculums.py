"""游戏式地形难度课程（模块 A）。

按 DreamWaQ 采用的游戏式课程规则，在每个 episode 结束（机器人重置）时，
依据机器人在本回合的实际移动表现更新其地形难度：

- 升级：机器人移动距离超过当前地形块长度的一半（``move_up_ratio`` 可配）。
- 降级：移动距离低于指令所要求的预期移动距离的一半（``move_down_ratio`` 可配），
  且该回合速度指令足够大、足以判定跟踪表现（``min_command_speed`` 保护）。
- 升级与降级互斥；等级限制在合法范围；到达最高等级后由
  ``TerrainImporter.update_env_origins`` 随机分配等级以保留多难度训练经验
  （Isaac Lab 内置行为，与本任务要求一致）。
- 低速/零速指令的回合不参与降级判定，避免静止指令环境被持续降级。

地形课程统计挂在环境对象的 ``env.terrain_curriculum_stats`` 属性上，由训练
Runner 按迭代写入 TensorBoard；函数返回值经 CurriculumManager 记录为
``Curriculum/terrain_levels/*`` 标量。

本文件不导入 isaaclab（仅 torch），核心判定逻辑可离线单元测试。
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

# 说明：本模块保持仅依赖 torch，以便离线单元测试；isaaclab 类型仅在函数内
# 延迟导入（作为默认参数构造）。


def compute_terrain_level_moves(
    distance: torch.Tensor,
    command_xy: torch.Tensor,
    terrain_length: float,
    episode_length_s: float,
    up_distance_ratio: float = 0.5,
    down_distance_ratio: float = 0.5,
    min_command_speed: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """纯逻辑：根据移动距离判定每个环境的升级/降级。

    Args:
        distance: (N,) 机器人相对本回合环境原点的移动距离（世界系平面距离）。
        command_xy: (N, 2) 本回合速度指令（机体系 vx、vy）。
        terrain_length: 地形块边长（米），读取自实际地形配置。
        episode_length_s: 最大回合时长（秒）。
        up_distance_ratio: 升级距离 = 地形块长度 × 该比例。
        down_distance_ratio: 降级距离 = 预期指令距离 × 该比例。
        min_command_speed: 指令平面速度低于该值时不判定降级。

    Returns:
        (move_up, move_down): 两个 (N,) 布尔张量，互斥。
    """
    if up_distance_ratio <= 0:
        raise ValueError("up_distance_ratio 必须为正")
    if not 0.0 <= down_distance_ratio <= 1.0:
        raise ValueError("down_distance_ratio 必须在 [0, 1] 内（0 表示只升不降）")
    command_speed = torch.norm(command_xy, dim=-1)
    # 升级：走过了半块地形。
    move_up = distance > terrain_length * up_distance_ratio
    # 降级：连预期指令距离的一半都没走到；静止或极低速指令不判定。
    expected_distance = command_speed * episode_length_s
    move_down = distance < expected_distance * down_distance_ratio
    move_down = move_down & (command_speed > min_command_speed)
    # 互斥：一次更新中不允许同时升级和降级。
    move_down = move_down & ~move_up
    return move_up, move_down


def terrain_levels_vel(
    env,
    env_ids: Sequence[int],
    asset_cfg=None,
    command_name: str = "base_velocity",
    up_distance_ratio: float = 0.5,
    down_distance_ratio: float = 0.5,
    min_command_speed: float = 0.1,
    debug: bool = False,
) -> dict[str, float]:
    """地形课程项（CurriculumTermCfg.func）。

    依据本回合实际移动距离更新 ``env_ids`` 中每个环境的地形等级，并维护
    逐等级的回合奖励/长度统计（读取 RewardManager 尚未清零的回合累计值，
    该课程在奖励管理器 reset 之前执行）。

    Returns:
        课程状态字典，由 CurriculumManager 记录为
        ``Curriculum/terrain_levels/<key>`` 标量。
    """
    if asset_cfg is None:
        from isaaclab.managers import SceneEntityCfg

        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    if terrain.cfg.terrain_generator is None:
        raise ValueError("地形课程仅在 terrain_type='generator' 时可用")
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=terrain.device)[env_ids]
    else:
        env_ids = torch.as_tensor(env_ids, device=terrain.device)

    # 使用每个环境自己的机器人位置与本回合环境原点，避免全局偏移被误计为移动距离。
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    command = env.command_manager.get_command(command_name)

    move_up, move_down = compute_terrain_level_moves(
        distance=distance,
        command_xy=command[env_ids, :2],
        terrain_length=float(terrain.cfg.terrain_generator.size[0]),
        episode_length_s=env.max_episode_length_s,
        up_distance_ratio=up_distance_ratio,
        down_distance_ratio=down_distance_ratio,
        min_command_speed=min_command_speed,
    )
    # 首次内部 reset（环境构建期间）时机器人尚未经历任何仿真步，root 位置
    # 仍是默认值，距离无意义——跳过这些环境的课程更新。
    lived = env.episode_length_buf[env_ids] > 0
    move_up = move_up & lived
    move_down = move_down & lived
    # 回合发生在更新前的等级上：先记录再更新等级。
    levels_before = terrain.terrain_levels[env_ids].clone()
    # 更新地形等级（含边界裁剪与最高等级随机化，由 TerrainImporter 处理）。
    terrain.update_env_origins(env_ids, move_up, move_down)
    num_levels = int(terrain.max_terrain_level)
    stats = getattr(env, "terrain_curriculum_stats", None)
    if stats is None or stats["num_levels"] != num_levels:
        stats = {
            "num_levels": num_levels,
            "move_up_count": 0,
            "move_down_count": 0,
            "reward_sum": torch.zeros(num_levels, device=terrain.device),
            "length_sum": torch.zeros(num_levels, device=terrain.device),
            "count": torch.zeros(num_levels, device=terrain.device),
        }
        env.terrain_curriculum_stats = stats
    stats["move_up_count"] += int(move_up.sum().item())
    stats["move_down_count"] += int(move_down.sum().item())
    # 逐等级回合统计：课程先于奖励管理器 reset 执行，回合累计值仍然有效。
    episode_sums = getattr(env.reward_manager, "_episode_sums", None)
    if episode_sums is not None:
        episode_reward = torch.zeros_like(distance)
        for value in episode_sums.values():
            episode_reward += value[env_ids]
        stats["reward_sum"].index_add_(0, levels_before.long(), episode_reward)
        stats["length_sum"].index_add_(0, levels_before.long(), env.episode_length_buf[env_ids].float())
        stats["count"].index_add_(0, levels_before.long(), torch.ones_like(distance))

    result = {
        "mean_level": terrain.terrain_levels.float().mean().item(),
        "num_move_up": int(move_up.sum().item()),
        "num_move_down": int(move_down.sum().item()),
        "num_reset_envs": int(env_ids.numel()),
    }
    if debug:
        # 逐等级环境计数与本次重置批次的移动距离统计。
        levels_now = terrain.terrain_levels
        max_level = int(terrain.max_terrain_level)
        for level in range(max_level):
            result[f"level_{level:02d}_envs"] = int((levels_now == level).sum().item())
        result["batch_mean_distance"] = float(distance.mean().item())
        result["batch_max_distance"] = float(distance.max().item())
    return result
