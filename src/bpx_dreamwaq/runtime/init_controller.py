"""仿真/实机共享的 BPX 解析 IK 与足端五次起身轨迹；不依赖 MuJoCo。"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class InitControllerConfig:
    """足端轨迹起身；depth 为髋到 toe link 原点的竖直距离。"""

    mode: str = "policy"
    duration: float = 4.0
    support_duration: float = 1.0
    support_depth: float = 0.19
    standing_depth: float = 0.38
    stiffness: float = 80.0
    # 5 ms 显式 PD 下，Kd=3 曾在趴姿静置后触发起身时产生高频振荡。
    damping: float = 2.0
    torque_limit: float = 30.0
    joint_speed_limit: float = 2.0
    tracking_error_limit: float = 0.35
    timeout: float = 8.0


def quintic(progress: float) -> float:
    """零起止速度、零起止加速度的五次时间标度。"""
    s = float(np.clip(progress, 0.0, 1.0))
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


class LegKinematics:
    def __init__(self, names, offsets, upper, lower, limits=None):
        self.indices = np.array([
            [names.index(f"{leg}_{joint}_joint") for joint in ("hip_roll", "hip_pitch", "knee")]
            for leg in ("fl", "fr", "hl", "hr")
        ])
        self.offsets = np.asarray(offsets, dtype=float)
        self.upper = np.broadcast_to(upper, (4,)).astype(float)
        self.lower = np.broadcast_to(lower, (4,)).astype(float)
        self.limits = None if limits is None else np.asarray(limits, dtype=float)
        if (self.offsets.shape != (4,)
                or not all(np.isfinite(x).all() for x in (self.offsets, self.upper, self.lower))
                or np.any(self.upper <= 0) or np.any(self.lower <= 0)
                or (self.limits is not None and (self.limits.shape != (12, 2)
                    or not np.isfinite(self.limits).all()
                    or np.any(self.limits[:, 0] >= self.limits[:, 1])))):
            raise ValueError("Init 几何与限位无效")

    def forward(self, joint_position: np.ndarray) -> np.ndarray:
        a, b, c = np.asarray(joint_position)[self.indices].T
        x = -self.upper * np.sin(b) - self.lower * np.sin(b + c)
        z = -self.upper * np.cos(b) - self.lower * np.cos(b + c)
        return np.column_stack((x, self.offsets * np.cos(a) - z * np.sin(a),
                                self.offsets * np.sin(a) + z * np.cos(a)))

    def inverse(self, feet: np.ndarray) -> np.ndarray:
        feet = np.asarray(feet, dtype=float)
        if feet.shape != (4, 3) or not np.all(np.isfinite(feet)):
            raise ValueError("Init 足端目标必须为 4x3 有限数组")
        x, y, z = feet.T
        radial_squared = y * y + z * z - self.offsets ** 2
        if np.any(radial_squared <= 0) or np.any(z >= 0):
            raise ValueError("Init 足端目标超出向下伸腿的可达空间")
        planar_z = -np.sqrt(radial_squared)
        roll = (np.arctan2(z, y) - np.arctan2(planar_z, self.offsets) + np.pi) % (2 * np.pi) - np.pi
        cosine = (x * x + planar_z ** 2 - self.upper ** 2 - self.lower ** 2) / (2 * self.upper * self.lower)
        if np.any(np.abs(cosine) > 1.0 + 1e-9):
            raise ValueError("Init 足端目标超出连杆长度允许的可达空间")
        knee = -np.arccos(np.clip(cosine, -1, 1))
        pitch = np.arctan2(-x, -planar_z) - np.arctan2(self.lower * np.sin(knee),
                                                    self.upper + self.lower * np.cos(knee))
        result = np.empty(12)
        result[self.indices] = np.column_stack((roll, pitch, knee))
        if self.limits is not None and (np.any(result < self.limits[:, 0] - 1e-7)
                                      or np.any(result > self.limits[:, 1] + 1e-7)):
            raise ValueError("Init IK 目标超过关节限位")
        return result


class FootTrajectory:
    """以物理步频率求轨迹；每次进入 INIT 捕获实际姿态并重新计时。"""

    label = "foot trajectory + PD (quintic)"
    behavior_name = "init_pd"

    def __init__(self, config, kinematics):
        self.config = config
        self.kinematics = kinematics
        self.progress = 0.0
        self.started_at: float | None = None
        self.last_time = 0.0
        self.phase = "idle"
        self.target = np.zeros(12)
        self.support_feet = np.column_stack((np.zeros(4), self.kinematics.offsets,
                                            np.full(4, -self.config.support_depth)))
        self.final_feet = self.support_feet.copy()
        self.final_feet[:, 2] = -self.config.standing_depth
        self.final_target = self.kinematics.inverse(self.final_feet)
        # Validate all nominal support/lift samples before connecting or simulating.
        for s in np.linspace(0, 1, 101):
            self.kinematics.inverse(self.support_feet + s * (self.final_feet - self.support_feet))

    def start(self, now, position) -> None:
        self.target = np.asarray(position, dtype=float).copy()
        if self.target.shape != (12,) or not np.isfinite(self.target).all() or not np.isfinite(now):
            raise ValueError("Init 初始角度/时间无效")
        self.start_feet = self.kinematics.forward(self.target)
        for s in np.linspace(0, 1, 101):
            self.kinematics.inverse(self.start_feet + s * (self.support_feet - self.start_feet))
        self.started_at = self.last_time = float(now)
        self.progress = 0.0
        self.phase = "support"

    @property
    def complete(self) -> bool:
        return self.progress >= self.config.duration and np.max(np.abs(self.target - self.final_target)) < 1e-5

    def timed_out(self, now) -> bool:
        return self.started_at is not None and now - self.started_at >= self.config.timeout

    def step(self, now, position, supported) -> np.ndarray:
        if self.started_at is None:
            raise RuntimeError("必须先 start() 再执行 Init")
        now = float(now)
        dt = max(0.0, now - self.last_time)
        self.last_time = now
        error = float(np.max(np.abs(self.target - np.asarray(position))))
        if error <= self.config.tracking_error_limit:
            next_progress = min(self.progress + dt, self.config.duration)
            # At the support waypoint, wait for four feet before lifting further.
            if not supported and next_progress > self.config.support_duration:
                next_progress = max(self.progress, self.config.support_duration)
            self.progress = next_progress
        if self.progress <= self.config.support_duration:
            alpha = quintic(self.progress / self.config.support_duration)
            feet = self.start_feet + alpha * (self.support_feet - self.start_feet)
            self.phase = "support"
        else:
            alpha = quintic((self.progress - self.config.support_duration)
                            / (self.config.duration - self.config.support_duration))
            feet = self.support_feet + alpha * (self.final_feet - self.support_feet)
            self.phase = "lift"
        desired = self.kinematics.inverse(feet)
        self.target += np.clip(desired - self.target, -self.config.joint_speed_limit * dt,
                               self.config.joint_speed_limit * dt)
        if self.complete:
            self.phase = "settle"
        elif error > self.config.tracking_error_limit or (self.progress >= self.config.support_duration and not supported):
            self.phase = "paused"
        return self.target.copy()
