"""BPX 任务的奖励函数。

通用奖励（速度跟踪、姿态、能耗、接触等）直接复用 Isaac Lab 内置实现
（``isaaclab.envs.mdp``，见 ``tasks/rough_env_cfg.py`` 的 RewardsCfg）。
任务特有的 ``feet_air_time`` 与上游
``isaaclab_tasks...velocity.mdp.rewards.feet_air_time`` 语义一致，
在本工程内实现以避免对 isaaclab_tasks 的源码耦合。
"""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

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
