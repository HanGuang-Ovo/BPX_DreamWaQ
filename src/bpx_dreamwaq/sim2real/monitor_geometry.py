"""只读姿态图的 BPX 骨架几何；不参与控制或足端接触判断。"""

import numpy as np


def leg_points(q):
    """官方 URDF 尺寸的示意骨架，不表示足端接触或实际距地高度。"""

    def rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    legs = []
    for i, (front, side) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        anchor = np.array([front * 0.2386, side * 0.056, 0.0])
        hip = anchor + rx(q[i]) @ np.array([0.0, side * 0.11522, 0.0])
        knee = hip + rx(q[i]) @ ry(q[i + 4]) @ np.array([0.0, 0.0, -0.23])
        foot = knee + rx(q[i]) @ ry(q[i + 4] + q[i + 8]) @ np.array([0.0, 0.0, -0.2315648])
        legs.append(np.array([anchor, hip, knee, foot]))
    return legs
