"""BPX Manager-Based 环境运行时扩展。"""

from __future__ import annotations

import math

from bpx_dreamwaq.tasks.mdp.numerical_guard import QuarantinedRewardTerm, QuarantinedRewardManager

from isaaclab.envs import ManagerBasedRLEnv


class BPXManagerBasedRLEnv(ManagerBasedRLEnv):
    """在奖励项求和后，为每个环境的单步总奖励设置下限。"""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        minimum_reward = cfg.minimum_step_reward
        if not math.isfinite(minimum_reward):
            raise ValueError("minimum_step_reward 必须为有限数")
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def load_managers(self):
        super().load_managers()
        # Wrap after construction, preserving stateful reward reset and diagnostic attributes.
        for name in self.reward_manager.active_terms:
            cfg = self.reward_manager.get_term_cfg(name)
            cfg.func = QuarantinedRewardTerm(cfg.func)
            self.reward_manager.set_term_cfg(name, cfg)
        self.reward_manager = QuarantinedRewardManager(self.reward_manager)

    def step(self, action):
        observations, reward, terminated, truncated, extras = super().step(action)
        invalid = getattr(self, "_numerical_invalid", None)
        if invalid is not None:
            extras["numerical_invalid"] = invalid.clone()
            # A numerical failure is never a time-limit bootstrap.
            truncated[invalid] = False
            extras.setdefault("log", {})["Numerical/invalid_envs"] = invalid.sum().float()
            extras["log"]["Numerical/total_invalid_envs"] = float(getattr(self, "_numerical_fault_count", 0))
        # RewardManager 已完成所有奖励项的加权、乘 dt 和求和；这里不归一化，
        # 只把过小的总奖励直接抬到配置下限，正奖励完全不变。
        # NaN/Inf 仍交给训练器的有限值保护立即报错，不能用限幅掩盖数值故障。
        finite = reward.isfinite()
        reward[finite] = reward[finite].clamp_min(self.cfg.minimum_step_reward)
        return observations, reward, terminated, truncated, extras
