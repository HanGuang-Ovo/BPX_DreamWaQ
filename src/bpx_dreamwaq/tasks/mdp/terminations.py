"""BPX 任务的补充终止条件。

机身过倾终止是本项目在机身非法接触终止之外补充的姿态保护。
训练不再按相对地形原点的高度终止，避免下坡、下台阶时提前重置。
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def base_tilt_above_maximum(
    env,
    maximum_tilt_rad: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机身相对重力方向的倾角超过阈值时终止。

    倾角由投影重力向量计算：tilt = acos(-g_z_body)，
    机身水平时为 0，倒立时为 π。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b
    tilt = torch.acos(torch.clamp(-projected_gravity[:, 2], -1.0, 1.0))
    return tilt > maximum_tilt_rad
