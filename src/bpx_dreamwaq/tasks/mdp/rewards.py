"""BPX 任务的奖励函数。

通用奖励（速度跟踪、姿态、能耗、接触等）直接复用 Isaac Lab 内置实现
（``isaaclab.envs.mdp``，见 ``tasks/rough_env_cfg.py`` 的 RewardsCfg）。
机身高度、足端净空与二阶动作平滑项采用 DreamWaQ 风格的误差平方惩罚。
``feet_air_time`` 与上游
``isaaclab_tasks...velocity.mdp.rewards.feet_air_time`` 语义一致，
在本工程内实现以避免对 isaaclab_tasks 的源码耦合。
"""

from __future__ import annotations

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if False:  # pragma: no cover - 仅为类型提示
    from isaaclab.envs import ManagerBasedRLEnv

# 复用 Isaac Lab 内置奖励函数；引入此处仅为保持 mdp 命名空间完整。
import isaaclab.envs.mdp as _lab_mdp

track_lin_vel_xy_exp = _lab_mdp.track_lin_vel_xy_exp
track_ang_vel_z_exp = _lab_mdp.track_ang_vel_z_exp
lin_vel_z_l2 = _lab_mdp.lin_vel_z_l2
ang_vel_xy_l2 = _lab_mdp.ang_vel_xy_l2
joint_torques_l2 = _lab_mdp.joint_torques_l2
joint_acc_l2 = _lab_mdp.joint_acc_l2
action_rate_l2 = _lab_mdp.action_rate_l2
undesired_contacts = _lab_mdp.undesired_contacts
flat_orientation_l2 = _lab_mdp.flat_orientation_l2
joint_pos_limits = _lab_mdp.joint_pos_limits


def base_height_l2(
    env: "ManagerBasedRLEnv",
    target_height: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """惩罚机身相对扫描区域平均地面高度与目标高度的平方误差。

    与 Isaac Lab 的地形感知高度项一致，额外过滤无效射线；整个环境
    没有有效命中点时返回零。扫描均值是对坡面、台阶支撑面的近似，
    ``target_height`` 指机器人根坐标原点距该参考面的高度（米）。
    """
    asset = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    ray_hits = sensor.data.ray_hits_w
    valid_hits = torch.isfinite(ray_hits).all(dim=-1)
    valid_count = valid_hits.sum(dim=1)
    ground_height = torch.where(valid_hits, ray_hits[:, :, 2], 0.0).sum(dim=1)
    ground_height = ground_height / valid_count.clamp_min(1)
    height_error = asset.data.root_pos_w[:, 2] - ground_height - target_height
    return torch.where(valid_count > 0, torch.square(height_error), 0.0)


def feet_clearance(
    env: "ManagerBasedRLEnv",
    target_height: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    max_ray_distance: float = 0.15,
) -> torch.Tensor:
    """足端相对局部地形的高度误差平方，按世界系水平速度加权求和。

    每只脚使用最近的有效高度扫描命中点估计脚下地形，避免世界系高度
    在坡面和台阶上产生系统性误罚。``target_height`` 是 toe_link 原点
    距地形的目标高度（米），不是鞋底净空。扫描间距 0.1 m，台阶边缘
    存在离散误差；没有半径 ``max_ray_distance`` 内有效点的脚不计入。
    静止支撑脚的水平速度接近零，纯转向时移动的脚仍参与此项。
    """
    asset = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    foot_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    foot_vel = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]
    ray_hits = sensor.data.ray_hits_w

    valid_hits = torch.isfinite(ray_hits).all(dim=-1)
    safe_hits = torch.where(valid_hits.unsqueeze(-1), ray_hits, 0.0)
    distances_sq = torch.sum(
        torch.square(foot_pos[:, :, None, :2] - safe_hits[:, None, :, :2]), dim=-1
    )
    distances_sq = distances_sq.masked_fill(~valid_hits[:, None, :], float("inf"))
    nearest_distance_sq, nearest_ids = distances_sq.min(dim=-1)
    ground_height = safe_hits[:, :, 2].gather(1, nearest_ids)
    height_error = foot_pos[:, :, 2] - ground_height - target_height
    horizontal_speed = torch.linalg.vector_norm(foot_vel[:, :, :2], dim=-1)
    penalty = torch.square(height_error) * horizontal_speed
    return torch.where(nearest_distance_sq <= max_ray_distance**2, penalty, 0.0).sum(dim=1)


class action_smoothness_l2(ManagerTermBase):
    """惩罚原始策略动作的二阶差分 ``sum((a_t - 2*a_{t-1} + a_{t-2})**2)``。

    ActionManager 提供当前和上一步动作，此项保存再前一步动作。
    RewardManager 在回合重置时调用 ``reset``，各环境独立清空历史；
    首步使用与 ActionManager 一致的零填充，不跨回合使用历史。
    每次奖励计算调用一次，不除以策略时间步，也不乘动作 scale。
    """

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._prev_prev_action = torch.zeros_like(env.action_manager.action)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._prev_prev_action[env_ids] = 0.0

    def __call__(self, env: "ManagerBasedRLEnv") -> torch.Tensor:
        action = env.action_manager.action
        prev_action = env.action_manager.prev_action
        penalty = torch.sum(torch.square(action - 2.0 * prev_action + self._prev_prev_action), dim=1)
        self._prev_prev_action.copy_(prev_action)
        return penalty


def feet_air_time(
    env: "ManagerBasedRLEnv",
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """足端迈步奖励：触地瞬间按滞空时间超过阈值的部分给正奖励。

    指令平面速度低于 0.1 m/s（不需要迈步）时奖励为零。
    与 Isaac Lab velocity 任务模板的实现语义一致。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # 零指令时不奖励迈步。
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward
