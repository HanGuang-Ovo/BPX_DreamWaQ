"""Explicit delayed PD with independent per-motor gain and strength variation."""

import math

import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass


class RandomizedDelayedPDActuator(DelayedPDActuator):
    def __init__(self, cfg, *args, **kwargs):
        for bounds in (cfg.kp_factor_range, cfg.kd_factor_range, cfg.strength_range):
            if not all(math.isfinite(x) for x in bounds) or not 0 < bounds[0] <= bounds[1]:
                raise ValueError("Actuator factor ranges must be finite and positive")
        super().__init__(cfg, *args, **kwargs)
        self._nominal_stiffness = self.stiffness.clone()
        self._nominal_damping = self.damping.clone()
        self.strength = torch.ones_like(self.stiffness)

    def reset(self, env_ids):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        for target, nominal, bounds in (
            (self.stiffness, self._nominal_stiffness, self.cfg.kp_factor_range),
            (self.damping, self._nominal_damping, self.cfg.kd_factor_range),
        ):
            target[ids] = nominal[ids] * torch.empty_like(target[ids]).uniform_(*bounds)
        self.strength[ids] = torch.empty_like(self.strength[ids]).uniform_(*self.cfg.strength_range)

    def _clip_effort(self, effort):
        # Scale motor output before the unchanged physical torque limit.
        return super()._clip_effort(effort * self.strength)


@configclass
class RandomizedDelayedPDActuatorCfg(DelayedPDActuatorCfg):
    class_type: type = RandomizedDelayedPDActuator
    kp_factor_range: tuple = (0.9, 1.1)
    kd_factor_range: tuple = (0.9, 1.1)
    strength_range: tuple = (0.9, 1.1)
