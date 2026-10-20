"""实机仅检查数值有效性和四元数模长。"""

import numpy as np

from .config import vector


def require_ready(config, stage):
    reasons = config.blockers(stage)
    if reasons:
        raise RuntimeError("运行资源不完整：\n  - " + "\n  - ".join(reasons))


def quaternion_xyzw(values):
    q = vector(values, 4, "IMU quaternion xyzw")
    if not 0.5 <= np.linalg.norm(q) <= 1.5:
        raise ValueError("IMU 四元数模长不在 [0.5,1.5]")
    return q


class StateGuard:
    def __init__(self, config):
        self.config = config

    def check(self, sample, now=None, *, require_imu=True, hardware=False, require_odometry=False):
        if not np.isfinite(sample.host_time):
            raise ValueError("反馈主机时间必须为有限数")
        vector(sample.q, 12, "joint position")
        vector(sample.dq, 12, "joint velocity")
        if require_imu:
            quaternion_xyzw(sample.quat_xyzw)
            vector(sample.omega, 3, "IMU angular velocity")
        for key, size in (("torque", 12), ("acceleration", 3),
                          ("velocity_body", 3), ("odom_position", 3)):
            values = getattr(sample, key)
            if values is not None:
                vector(values, size, key)
        if require_odometry or self.config.policy["include_base_lin_vel"]:
            vector(sample.velocity_body, 3, "body velocity")
