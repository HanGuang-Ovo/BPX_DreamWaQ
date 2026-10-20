"""机身到世界的 wxyz 四元数与机体系重力方向。"""

import numpy as np


def rotation_body_to_world(quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("四元数必须为非零有限 wxyz 数组")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
    ])


def rotate_world_to_body(quaternion_wxyz, vector_world):
    vector = np.asarray(vector_world, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("世界系向量必须为三个有限数")
    return rotation_body_to_world(quaternion_wxyz).T @ vector
