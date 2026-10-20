"""训练关节顺序与官方 SDK 顺序；默认沿用 SDK 坐标；实机接入时核对与训练模型一致。"""

LEGS = ("fl", "fr", "hl", "hr")
JOINT_TYPES = ("hip_roll", "hip_pitch", "knee")
POLICY_JOINT_NAMES = tuple(f"{leg}_{kind}_joint" for kind in JOINT_TYPES for leg in LEGS)
SDK_JOINT_NAMES = tuple(f"{leg}_{kind}_joint" for leg in LEGS for kind in JOINT_TYPES)
DEFAULT_JOINT_POSITION = (0.0,) * 4 + (0.7,) * 4 + (-1.4,) * 4
