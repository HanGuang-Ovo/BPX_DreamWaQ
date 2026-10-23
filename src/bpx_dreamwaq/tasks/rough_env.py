"""BPX Manager-Based 环境运行时扩展。"""

from __future__ import annotations

import math

from isaaclab.envs import ManagerBasedRLEnv


class BPXManagerBasedRLEnv(ManagerBasedRLEnv):
    """在奖励项求和后，为每个环境的单步总奖励设置下限。"""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        minimum_reward = cfg.minimum_step_reward
        if not math.isfinite(minimum_reward):
            raise ValueError("minimum_step_reward 必须为有限数")
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def step(self, action):
        observations, reward, terminated, truncated, extras = super().step(action)
        # RewardManager 已完成所有奖励项的加权、乘 dt 和求和；这里不归一化，
        # 只把过小的总奖励直接抬到配置下限，正奖励完全不变。
        # NaN/Inf 仍交给训练器的有限值保护立即报错，不能用限幅掩盖数值故障。
        finite = reward.isfinite()
        reward[finite] = reward[finite].clamp_min(self.cfg.minimum_step_reward)
        return observations, reward, terminated, truncated, extras
