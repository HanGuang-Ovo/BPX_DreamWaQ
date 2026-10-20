"""Stand / Walk / STOPPING 路由；实机控制器与只读窗口共用。"""

import numpy as np

from .transition import SlewRateLimiter


class PolicyRouter:
    def __init__(self, control, period):
        self.control, self.period = control, period
        self.limiter = SlewRateLimiter(control["command_slew_rate"])
        self.mode = "stand"
        self.mode_time = self.condition_time = 0.0

    def step(self, command):
        command = np.asarray(command, dtype=float)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("command 必须为三个有限值")
        control, dt = self.control, self.period
        limits = np.asarray(control["command_limit"])
        command = np.clip(command, -limits, limits)
        activity = float(np.max(np.abs(command) / limits))
        self.mode_time += dt
        old_mode = self.mode
        if self.mode == "stand":
            self.condition_time = self.condition_time + dt if activity > control["walk_enter_threshold"] else 0.0
            if self.condition_time >= control["walk_enter_delay_s"]:
                self.mode = "walk"
        elif self.mode == "walk":
            stop = self.mode_time >= control["minimum_walk_time_s"] and activity < control["walk_exit_threshold"]
            self.condition_time = self.condition_time + dt if stop else 0.0
            if self.condition_time >= control["walk_exit_delay_s"]:
                self.mode = "stopping"
        elif activity > control["walk_enter_threshold"]:
            self.mode = "walk"
        elif self.mode_time >= control["stopping_duration_s"]:
            self.mode = "stand"
        if self.mode != old_mode:
            self.mode_time = self.condition_time = 0.0
        target = command if self.mode == "walk" else np.zeros(3)
        return self.limiter.update(target, dt)
