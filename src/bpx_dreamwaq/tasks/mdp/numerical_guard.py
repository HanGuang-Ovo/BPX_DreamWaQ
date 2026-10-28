"""Quarantine physically implausible transitions before reward and auto-reset."""

from __future__ import annotations

import math

import torch


def check_robot_state(env, *, max_linear_speed: float = 20.0,
                      max_angular_speed: float = 50.0,
                      max_joint_speed: float = 100.0, fail_fast: bool = False) -> torch.Tensor:
    """Mark invalid environments for reset and loss exclusion; optionally fail fast.

    Diagnostic thresholds do not clip physical states or normal control actions.
    """
    limits = (max_linear_speed, max_angular_speed, max_joint_speed)
    if not all(math.isfinite(v) and v > 0 for v in limits):
        raise ValueError("numerical guard limits must be finite and positive")
    robot = env.scene["robot"]
    data = robot.data
    fields = {
        "root_pos_w": data.root_pos_w,
        "root_quat_w": data.root_quat_w,
        "root_lin_vel_w": data.root_lin_vel_w,
        "root_ang_vel_w": data.root_ang_vel_w,
        "joint_pos": data.joint_pos,
        "joint_vel": data.joint_vel,
        "applied_torque": data.applied_torque,
    }
    invalid = torch.zeros(env.num_envs, dtype=torch.bool, device=data.joint_vel.device)
    for tensor in fields.values():
        invalid |= ~torch.isfinite(tensor).all(dim=-1)
    invalid |= (data.root_lin_vel_w.abs() > max_linear_speed).any(dim=-1)
    invalid |= (data.root_ang_vel_w.abs() > max_angular_speed).any(dim=-1)
    invalid |= (data.joint_vel.abs() > max_joint_speed).any(dim=-1)
    env._numerical_invalid = invalid.clone()
    if invalid.any().item():
        ids = invalid.nonzero(as_tuple=False).flatten()
        env._numerical_fault_count = getattr(env, "_numerical_fault_count", 0) + ids.numel()
        env._numerical_fault_events = getattr(env, "_numerical_fault_events", 0) + 1
        # Bound diagnostic I/O while retaining a running counter.
        if not fail_fast and env._numerical_fault_events > 3 and env._numerical_fault_events % 100:
            return invalid
        details = []
        for idx in ids[:4].tolist():
            state = {name: tensor[idx].detach().cpu().tolist() for name, tensor in fields.items()}
            state["episode_step"] = int(env.episode_length_buf[idx].item())
            if hasattr(env.scene, "terrain") and hasattr(env.scene.terrain, "terrain_levels"):
                state["terrain_level"] = int(env.scene.terrain.terrain_levels[idx].item())
                state["terrain_type"] = int(env.scene.terrain.terrain_types[idx].item())
            if hasattr(env, "action_manager"):
                state["action"] = env.action_manager.action[idx].detach().cpu().tolist()
            if hasattr(env, "command_manager"):
                state["command"] = env.command_manager.get_command("base_velocity")[idx].detach().cpu().tolist()
            details.append(f"env_id={idx} {state}")
        message = (
            f"BPX pre-reset numerical guard: {ids.numel()} invalid envs; "
            f"limits |linear component|={max_linear_speed} m/s, |angular component|={max_angular_speed} rad/s, "
            f"|joint velocity|={max_joint_speed} rad/s; " + "; ".join(details)
        )
        if fail_fast:
            raise FloatingPointError(message)
        print(message, flush=True)
    return invalid


class QuarantinedRewardTerm:
    """Mask raw invalid per-env rewards before weighting and episode accumulation."""

    def __init__(self, func):
        self.func = func

    def __getattr__(self, name):
        return getattr(self.func, name)

    def __call__(self, env, **params):
        value = self.func(env, **params)
        invalid = getattr(env, "_numerical_invalid", None)
        return value if invalid is None else torch.where(invalid, torch.zeros_like(value), value)


class QuarantinedRewardManager:
    """Delegate reward computation, excluding aborted episodes from reward logs."""

    def __init__(self, manager):
        self.manager = manager

    def __getattr__(self, name):
        return getattr(self.manager, name)

    def reset(self, env_ids=None):
        invalid = getattr(self.manager._env, "_numerical_invalid", None)
        if invalid is None:
            return self.manager.reset(env_ids)
        ids = torch.arange(invalid.numel(), device=invalid.device)
        if env_ids is not None:
            ids = ids[env_ids]
        good = ids[~invalid[ids]]
        info = {}
        if good.numel():
            for key, values in self.manager._episode_sums.items():
                info["Episode_Reward/" + key] = values[good].mean() / self.manager._env.max_episode_length_s
        # Clear both healthy and faulty episode sums and all stateful reward terms.
        self.manager.reset(env_ids)
        return info
