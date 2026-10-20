"""单帧 Actor 契约，历史展平由 ObservationHistory 完成。"""

import numpy as np


def actor_frame(angular_velocity, projected_gravity, command, joint_position,
                joint_velocity, last_action, default_position, linear_velocity=None):
    values = [angular_velocity, projected_gravity, command,
              np.asarray(joint_position) - np.asarray(default_position),
              joint_velocity, last_action]
    widths = [3, 3, 3, 12, 12, 12]
    if linear_velocity is not None:
        values.insert(0, linear_velocity)
        widths.insert(0, 3)
    arrays = [np.asarray(value, dtype=np.float32) for value in values]
    if any(a.shape != (width,) or not np.isfinite(a).all()
           for a, width in zip(arrays, widths)):
        raise ValueError("Actor 观测形状错误或存在非有限值")
    return np.concatenate(arrays)
