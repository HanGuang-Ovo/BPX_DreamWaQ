"""BPX 任务的补充终止条件。

机身过倾/过低终止与部署端 supervisor 的跌倒判定保持一致，作为机身
非法接触终止之外的姿态保护。高度终止使用相对本环境地形原点的高度，
适配粗糙地形（世界系高度在地形网格上无统一参考面）。
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


def base_height_below_minimum(
    env,
    minimum_height: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机身相对本环境地形原点的高度低于阈值时终止。"""
    asset: Articulation = env.scene[asset_cfg.name]
    relative_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return relative_height < minimum_height
