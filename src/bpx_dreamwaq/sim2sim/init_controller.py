"""水平趴卧起身：髋坐标系足端五次插值、解析 IK 和关节 PD。

四腿先同步收脚到低位支撑，再沿竖直方向伸展。toe link 原点是运动学端点，
并非足底接触点；standing_depth 不等于机身离地高度。此轨迹面向平地趴卧，
不包含跨台阶落足规划或翻身恢复。所有几何尺寸、轴向和限位从 MJCF 读取。
"""

from __future__ import annotations

import mujoco
import numpy as np

from .robot import BpxMujocoRobot
from bpx_dreamwaq.runtime.init_controller import LegKinematics as SharedKinematics, FootTrajectory, quintic


class LegKinematics(SharedKinematics):
    """BPX 的 x 轴髋横滚 + y 轴髋俯仰/膝结构，固定膝向后弯曲分支。"""

    def __init__(self, robot: BpxMujocoRobot):
        model = robot.model
        names = robot.config.joint_names
        self.indices = np.array([
            [names.index(f"{leg}_{joint}_joint") for joint in ("hip_roll", "hip_pitch", "knee")]
            for leg in ("fl", "fr", "hl", "hr")
        ])
        offsets, lengths = [], []
        for leg in ("fl", "fr", "hl", "hr"):
            bodies = [model.body(f"{leg}_{part}_link").id for part in ("hip", "thigh", "calf", "toe")]
            hip, thigh, calf, toe = bodies
            if list(model.body_parentid[bodies]) != [robot.base_body_id, hip, thigh, calf]:
                raise ValueError("Init IK 不支持该腿部父子结构")
            if not np.allclose(model.body_quat[bodies], [1, 0, 0, 0]):
                raise ValueError("Init IK 要求腿部零位坐标系方向一致")
            for part, body, axis in zip(("hip_roll", "hip_pitch", "knee"), bodies, np.eye(3)[[0, 1, 1]]):
                joint = model.joint(f"{leg}_{part}_joint").id
                if (model.jnt_bodyid[joint] != body or not np.allclose(model.jnt_axis[joint], axis)
                        or not np.allclose(model.jnt_pos[joint], 0)
                        or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE):
                    raise ValueError("Init IK 不支持该关节轴或关节原点")
            offset = model.body_pos[thigh]
            upper, lower = model.body_pos[calf], model.body_pos[toe]
            if (not np.allclose(offset[[0, 2]], 0) or not np.allclose(upper[:2], 0)
                    or not np.allclose(lower[:2], 0) or upper[2] >= 0 or lower[2] >= 0):
                raise ValueError("Init IK 不支持该腿部连杆偏移")
            offsets.append(offset[1])
            lengths.append([-upper[2], -lower[2]])
        self.offsets = np.asarray(offsets)
        self.upper, self.lower = np.asarray(lengths).T
        self.limits = model.jnt_range[robot.joint_ids].copy()



class FootTrajectoryInit(FootTrajectory):
    """MuJoCo 反馈适配；轨迹实现与实机完全共用。"""

    def __init__(self, robot: BpxMujocoRobot):
        self.robot = robot
        super().__init__(robot.config.init_controller, LegKinematics(robot))
        self.target = robot.default_joint_position.copy()

    def start(self):
        super().start(float(self.robot.data.time), self.robot.joint_position())

    @property
    def timed_out(self):
        return super().timed_out(float(self.robot.data.time))

    def __call__(self, observation=None):
        target = self.step(float(self.robot.data.time), self.robot.joint_position(),
                           self.robot.feet_in_contact() == 4)
        return ((target - self.robot.default_joint_position) / self.robot.config.control.action_scale).astype(np.float32)
