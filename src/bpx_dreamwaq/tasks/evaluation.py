"""Per-environment acceptance metrics; simulator truth is used only for scoring."""

import numpy as np


class AcceptanceMetrics:
    def __init__(self, num_envs, command, linear_tolerance, yaw_tolerance):
        self.command = np.asarray(command, dtype=np.float64)
        self.linear_tolerance = linear_tolerance
        self.yaw_tolerance = yaw_tolerance
        self.linear_error_sq = np.zeros(num_envs)
        self.yaw_error_sq = np.zeros(num_envs)
        self.estimate_error_sq = np.zeros((num_envs, 3))
        self.terminated = np.zeros(num_envs, dtype=np.int64)
        self.truncated = np.zeros(num_envs, dtype=np.int64)
        self.max_tilt = 0.0
        self.samples = 0

    def update(self, linear_velocity, yaw_rate, estimate, tilt, terminated, truncated, *, score=True):
        values = [np.asarray(x) for x in (linear_velocity, yaw_rate, estimate, tilt)]
        if not all(np.isfinite(x).all() for x in values):
            raise FloatingPointError("验收状态或 CENet 输出出现 NaN/Inf")
        linear_velocity, yaw_rate, estimate, tilt = values
        # Count every termination, including the warm-up period and auto-resets.
        self.terminated += np.asarray(terminated, dtype=np.int64)
        self.truncated += np.asarray(truncated, dtype=np.int64)
        self.max_tilt = max(self.max_tilt, float(np.max(tilt)))
        if score:
            self.linear_error_sq += np.square(linear_velocity[:, :2] - self.command[:2]).sum(-1)
            self.yaw_error_sq += np.square(yaw_rate - self.command[2])
            self.estimate_error_sq += np.square(estimate - linear_velocity)
            self.samples += 1

    def result(self, *, completed):
        if not self.samples:
            raise ValueError("验收没有有效采样；duration 必须大于 warmup")
        linear_rmse = np.sqrt(self.linear_error_sq / self.samples)
        yaw_rmse = np.sqrt(self.yaw_error_sq / self.samples)
        passed = bool(
            completed and not self.terminated.any() and not self.truncated.any()
            and linear_rmse.max() <= self.linear_tolerance
            and yaw_rmse.max() <= self.yaw_tolerance
        )
        return {
            "passed": passed, "completed": completed,
            "command": self.command.tolist(), "scored_steps": self.samples,
            "terminated_count": int(self.terminated.sum()),
            "truncated_count": int(self.truncated.sum()),
            "failed_env_count": int(np.count_nonzero(self.terminated + self.truncated)),
            "linear_rmse_per_env": linear_rmse.tolist(),
            "yaw_rmse_per_env": yaw_rmse.tolist(),
            "max_linear_rmse": float(linear_rmse.max()),
            "max_yaw_rmse": float(yaw_rmse.max()),
            "cenet_velocity_rmse_xyz": np.sqrt(self.estimate_error_sq.mean(0) / self.samples).tolist(),
            "max_tilt_rad": self.max_tilt,
            "linear_tolerance": self.linear_tolerance, "yaw_tolerance": self.yaw_tolerance,
        }
