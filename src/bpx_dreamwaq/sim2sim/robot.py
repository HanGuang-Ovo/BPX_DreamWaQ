"""MuJoCo 机器人状态、观测和 PD 控制接口。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np
import numpy.typing as npt

from .config import Sim2SimConfig
from bpx_dreamwaq.runtime.math import rotate_world_to_body

FloatArray = npt.NDArray[np.floating]


@dataclass(frozen=True)
class RobotState:
    base_linear_velocity: FloatArray
    base_angular_velocity: FloatArray
    projected_gravity: FloatArray
    joint_position: FloatArray
    joint_velocity: FloatArray


class BpxMujocoRobot:
    """通过名称而不是隐含索引访问 BPX 模型。"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: Sim2SimConfig):
        self.model = model
        self.data = data
        self.config = config
        self._configure_simulation()
        self.joint_ids = np.array(
            [self._required_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in config.joint_names], dtype=np.int32
        )
        self.joint_qpos_addresses = model.jnt_qposadr[self.joint_ids].copy()
        self.joint_dof_addresses = model.jnt_dofadr[self.joint_ids].copy()
        self.base_joint_id = self._required_id(mujoco.mjtObj.mjOBJ_JOINT, config.base_joint_name)
        self.base_qpos_address = int(model.jnt_qposadr[self.base_joint_id])
        self.base_dof_address = int(model.jnt_dofadr[self.base_joint_id])
        self.base_body_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, config.base_body_name)
        self.floor_geom_id = self._required_id(mujoco.mjtObj.mjOBJ_GEOM, config.floor_geom_name)
        self.ground_geom_ids = {self.floor_geom_id} | {
            gid for gid in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").startswith("terrain_ground_")
        }
        # Reserve visual group 5 for ground ray queries; no robot or decoration is included.
        for gid in self.ground_geom_ids:
            model.geom_group[gid] = 5
        self.toe_body_ids = {
            body_id
            for body_id in range(model.nbody)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "").endswith("_toe_link")
        }
        self.actuator_ids = self._find_actuators()
        self.default_joint_position = np.asarray(config.initial_state.default_joint_position, dtype=np.float64)
        self._validate_model()
        self._configure_actuators()

    def _configure_simulation(self) -> None:
        integrators = {
            "euler": mujoco.mjtIntegrator.mjINT_EULER,
            "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
            "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
            "rk4": mujoco.mjtIntegrator.mjINT_RK4,
        }
        self.model.opt.timestep = self.config.simulation.timestep
        self.model.opt.integrator = integrators[self.config.simulation.integrator]

    def _required_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo 模型中找不到 {object_type.name}: {name}")
        return object_id

    def _find_actuators(self) -> npt.NDArray[np.int32]:
        actuator_ids: list[int] = []
        for joint_id, joint_name in zip(self.joint_ids, self.config.joint_names, strict=True):
            matches = np.flatnonzero(self.model.actuator_trnid[:, 0] == joint_id)
            if len(matches) != 1:
                raise ValueError(f"关节 {joint_name} 应恰好对应一个执行器，实际为 {len(matches)} 个")
            actuator_ids.append(int(matches[0]))
        return np.asarray(actuator_ids, dtype=np.int32)

    def _validate_model(self) -> None:
        if self.model.jnt_type[self.base_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"{self.config.base_joint_name} 必须是 free joint")
        if len(set(self.joint_qpos_addresses.tolist())) != len(self.joint_ids):
            raise ValueError("关节 qpos 地址存在重复")
        if self.model.nu < len(self.joint_ids):
            raise ValueError("执行器数量小于受控关节数量")
        if len(self.toe_body_ids) != 4:
            raise ValueError(f"MuJoCo 模型应包含 4 个 toe body，实际为 {len(self.toe_body_ids)} 个")

    def _configure_actuators(self) -> None:
        """把已有 torque motor 配置为显式力矩或 MuJoCo 内置 PD 伺服。"""

        if self.config.control.mode == "explicit":
            return
        for actuator_id in self.actuator_ids:
            self.model.actuator_gaintype[actuator_id] = mujoco.mjtGain.mjGAIN_FIXED
            self.model.actuator_biastype[actuator_id] = mujoco.mjtBias.mjBIAS_AFFINE
            self.model.actuator_gainprm[actuator_id] = 0.0
            self.model.actuator_biasprm[actuator_id] = 0.0
            self.model.actuator_gainprm[actuator_id, 0] = self.config.control.stiffness
            self.model.actuator_biasprm[actuator_id, 1] = -self.config.control.stiffness
            self.model.actuator_biasprm[actuator_id, 2] = -self.config.control.damping
            self.model.actuator_ctrllimited[actuator_id] = 0
            self.model.actuator_forcelimited[actuator_id] = 1
            self.model.actuator_forcerange[actuator_id] = (
                -self.config.control.torque_limit,
                self.config.control.torque_limit,
            )

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        address = self.base_qpos_address
        self.data.qpos[address : address + 3] = self.config.initial_state.base_position
        self.data.qpos[address + 3 : address + 7] = self.config.initial_state.base_quaternion_wxyz
        self.data.qpos[self.joint_qpos_addresses] = self.config.initial_state.joint_position
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def joint_position(self) -> FloatArray:
        return self.data.qpos[self.joint_qpos_addresses].copy()

    def joint_velocity(self) -> FloatArray:
        return self.data.qvel[self.joint_dof_addresses].copy()

    def base_velocity_body(self) -> tuple[FloatArray, FloatArray]:
        address = self.base_dof_address
        # MuJoCo free joint 的前三项 qvel 是世界系线速度，后三项是机体系角速度。
        # 平移速度位于根 link 原点，而 Isaac Lab 的 root_lin_vel_b 使用根 link 质心速度，
        # 因此需要加上 omega × r_com。直接读取 qvel 还能避免 mj_step 后 cvel 落后一个积分步。
        linear_velocity_world = self.data.qvel[address : address + 3]
        angular_velocity_body = self.data.qvel[address + 3 : address + 6]
        quaternion_wxyz = self.data.qpos[self.base_qpos_address + 3 : self.base_qpos_address + 7]
        linear_velocity_body_at_origin = rotate_world_to_body(quaternion_wxyz, linear_velocity_world)
        center_of_mass_position_body = self.model.body_ipos[self.base_body_id]
        linear_velocity_body = linear_velocity_body_at_origin + np.cross(
            angular_velocity_body, center_of_mass_position_body
        )
        return linear_velocity_body, angular_velocity_body.copy()

    def projected_gravity(self) -> FloatArray:
        gravity_world = np.asarray(self.model.opt.gravity, dtype=np.float64)
        gravity_norm = np.linalg.norm(gravity_world)
        if gravity_norm == 0.0:
            raise ValueError("MuJoCo 重力不能为零，否则无法计算投影重力")
        quaternion_wxyz = self.data.xquat[self.base_body_id]
        return rotate_world_to_body(quaternion_wxyz, gravity_world / gravity_norm)

    def state(self) -> RobotState:
        linear_velocity, angular_velocity = self.base_velocity_body()
        return RobotState(
            base_linear_velocity=linear_velocity,
            base_angular_velocity=angular_velocity,
            projected_gravity=self.projected_gravity(),
            joint_position=self.joint_position(),
            joint_velocity=self.joint_velocity(),
        )

    def observation(self, command: FloatArray, last_action: FloatArray) -> npt.NDArray[np.float32]:
        """按 Isaac Lab PolicyCfg 中的项顺序构造48维观测。"""

        state = self.state()
        observation = np.concatenate(
            (
                state.base_linear_velocity,
                state.base_angular_velocity,
                state.projected_gravity,
                np.asarray(command, dtype=np.float64),
                state.joint_position - self.default_joint_position,
                state.joint_velocity,
                np.asarray(last_action, dtype=np.float64),
            )
        ).astype(np.float32)
        # Preserve full frames for Stand/Init; the runner filters locomotion input.
        full_dimension = 12 + 3 * len(self.config.joint_names)
        if observation.shape != (full_dimension,):
            raise RuntimeError(
                f"单帧观测维度不匹配：期望 {full_dimension}，实际 {observation.shape}"
            )
        if not np.all(np.isfinite(observation)):
            raise FloatingPointError("观测中出现 NaN 或 Inf")
        return observation

    def desired_joint_position(self, action: FloatArray) -> FloatArray:
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != (self.config.observation.action_dimension,):
            raise ValueError(
                f"动作维度不匹配：期望 {self.config.observation.action_dimension}，实际 {action_array.shape}"
            )
        return self.default_joint_position + self.config.control.action_scale * action_array

    def apply_pd(
        self, action: FloatArray, *, stiffness: float | None = None,
        damping: float | None = None, torque_limit: float | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        desired_position = self.desired_joint_position(action)
        kp = self.config.control.stiffness if stiffness is None else stiffness
        kd = self.config.control.damping if damping is None else damping
        limit = self.config.control.torque_limit if torque_limit is None else min(torque_limit, self.config.control.torque_limit)
        torque = kp * (desired_position - self.joint_position())
        torque -= kd * self.joint_velocity()
        torque = np.clip(torque, -limit, limit)
        if self.config.control.mode == "implicit":
            self.data.ctrl[self.actuator_ids] = desired_position
        else:
            self.data.ctrl[self.actuator_ids] = torque
        return desired_position, torque

    def ground_height(self) -> float:
        """Surface directly below the base, for monitoring/supervision only."""
        position = self.data.xpos[self.base_body_id].copy()
        position[2] = max(100.0, position[2] + 1.0)
        geom_id = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            self.model, self.data, position, np.array([0., 0., -1.]),
            np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8), 1, -1, geom_id,
        )
        if distance < 0 or geom_id[0] not in self.ground_geom_ids:
            raise RuntimeError("No ground below robot; outside the supported test area")
        return float(position[2] - distance)

    def base_height(self) -> float:
        """Vertical clearance above local ground (world Z is logged separately)."""
        return float(self.data.xpos[self.base_body_id, 2]) - self.ground_height()

    def terrain_region(self) -> str:
        if self.config.terrain.kind == "flat":
            return "FLAT"
        from .terrain import region_at
        return region_at(*self.data.xpos[self.base_body_id, :2])

    def tilt_angle(self) -> float:
        projected_gravity = self.projected_gravity()
        return math.acos(float(np.clip(-projected_gravity[2], -1.0, 1.0)))

    def torso_touches_floor(self) -> bool:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 not in self.ground_geom_ids and geom2 not in self.ground_geom_ids:
                continue
            other_geom = geom2 if geom1 in self.ground_geom_ids else geom1
            if int(self.model.geom_bodyid[other_geom]) == self.base_body_id:
                return True
        return False

    def feet_in_contact(self) -> int:
        """返回当前与地面接触的不同足端数量。"""

        contacting_feet: set[int] = set()
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 not in self.ground_geom_ids and geom2 not in self.ground_geom_ids:
                continue
            other_geom = geom2 if geom1 in self.ground_geom_ids else geom1
            other_body = int(self.model.geom_bodyid[other_geom])
            if other_body in self.toe_body_ids:
                contacting_feet.add(other_body)
        return len(contacting_feet)
