"""实机按时间起身及 Stand/Walk/STOPPING；无需高度或接地验收。"""

import numpy as np

from bpx_dreamwaq.runtime.init_controller import LegKinematics, quintic
from .config import vector
from .controller import ActorController


class HardwareFootTrajectory:
    """足端五次轨迹和 IK；不执行限位、限速、跟踪误差或接地门槛。"""

    def __init__(self, config, names):
        self.config = config
        self.kinematics = LegKinematics(names, [.11522, -.11522, .11522, -.11522], .23, .2315648)
        self.support_feet = np.column_stack((np.zeros(4), self.kinematics.offsets,
                                            np.full(4, -config.support_depth)))
        self.final_feet = self.support_feet.copy()
        self.final_feet[:, 2] = -config.standing_depth
        self.final_target = self.kinematics.inverse(self.final_feet)
        self.started_at = None
        self.progress = 0.0
        self.phase = "idle"
        self.target = np.zeros(12)

    def start(self, now, position):
        self.target = vector(position, 12, "Init initial position").copy()
        self.start_feet = self.kinematics.forward(self.target)
        self.started_at = float(now)
        self.progress = 0.0
        self.phase = "support"

    @property
    def complete(self):
        return self.progress + 1e-9 >= self.config.duration

    def step(self, now):
        self.progress = min(max(float(now) - self.started_at, 0.0), self.config.duration)
        if self.progress <= self.config.support_duration:
            alpha = quintic(self.progress / self.config.support_duration)
            feet = self.start_feet + alpha * (self.support_feet - self.start_feet)
            self.phase = "support"
        else:
            alpha = quintic((self.progress - self.config.support_duration)
                            / (self.config.duration - self.config.support_duration))
            feet = self.support_feet + alpha * (self.final_feet - self.support_feet)
            self.phase = "lift"
        self.target = vector(self.kinematics.inverse(feet), 12, "Init target")
        if self.complete:
            self.phase = "complete"
        return self.target.copy()


class HardwareController:
    def __init__(self, config, *, policies=None):
        self.config = config
        self.actor = ActorController(config, policies=policies)
        self.calibration = self.actor.calibration
        self.trajectory = HardwareFootTrajectory(config.init_controller, list(config.policy["joint_names"]))
        self.mode = "waiting_init"
        self.waiting_target = self.target = self.result = None
        self.kp = np.asarray(config.control["kp"], dtype=float)
        self.kd = np.asarray(config.control["kd"], dtype=float)
        self.gain_start = (self.kp.copy(), self.kd.copy())
        self.gain_started_at = None

    def initialize(self, sample):
        q = self.calibration.position_from_sdk(sample.q)
        self.waiting_target = self.target = q.copy()
        self.actor.commit_target(q)
        self.mode = "stand" if sample.telemetry.get("motion_state") == 6 else "waiting_init"

    def _gain_transition(self, now):
        self.gain_start = (self.kp.copy(), self.kd.copy())
        self.gain_started_at = now

    def step(self, sample, command, *, now, policy_due, init_requested=False):
        command = vector(command, 3, "command")
        q = self.calibration.position_from_sdk(sample.q)
        if policy_due and self.mode == "waiting_init" and init_requested:
            self.trajectory.start(now, q)
            self.mode = "init"
            self._gain_transition(now)
        if self.mode == "init":
            self.target = self.trajectory.step(now)
            if self.trajectory.complete and policy_due:
                self.mode = "stand"
                self.actor.started_at = None
                self.actor.stand_history = None
                self._gain_transition(now)
        if policy_due and self.mode in ("stand", "walk", "stopping"):
            self.result = self.actor.step(sample, command)
            self.mode = self.actor.mode
            self.target = np.asarray(self.result["target_model"])
        elif self.mode == "waiting_init":
            self.target = self.waiting_target.copy()
        if self.result is None or self.mode in ("waiting_init", "init"):
            self.result = {"mode": self.mode, "command": [0., 0., 0.]}
        init = self.config.init_controller
        desired_kp = np.full(12, init.stiffness) if self.mode == "init" else np.asarray(self.config.control["kp"])
        desired_kd = np.full(12, init.damping) if self.mode == "init" else np.asarray(self.config.control["kd"])
        alpha = (1. if self.gain_started_at is None else
                 quintic((now - self.gain_started_at) / self.config.control["action_blend_duration_s"]))
        self.kp = self.gain_start[0] + alpha * (desired_kp - self.gain_start[0])
        self.kd = self.gain_start[1] + alpha * (desired_kd - self.gain_start[1])
        self.result.update(mode=self.mode, target_model=vector(self.target, 12, "target").tolist(),
                           reason="waiting_for_RB" if self.mode == "waiting_init" else
                                  "timed_foot_trajectory" if self.mode == "init" else "policy_router",
                           init_phase=self.trajectory.phase if self.mode == "init" else "",
                           init_progress=self.trajectory.progress if self.mode == "init" else 0.,
                           init_requested=bool(init_requested))
        return self.result

    def commit_target(self, target):
        self.actor.commit_target(target)
