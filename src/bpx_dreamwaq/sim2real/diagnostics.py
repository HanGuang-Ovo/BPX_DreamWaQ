"""只读诊断数据与策略影子推理；不包含任何运动或标零接口。"""

import math

import numpy as np

from bpx_dreamwaq.runtime.history import ObservationHistory
from bpx_dreamwaq.runtime.joints import SDK_JOINT_NAMES, POLICY_JOINT_NAMES
from bpx_dreamwaq.runtime.math import rotation_body_to_world
from bpx_dreamwaq.runtime.observation import actor_frame
from bpx_dreamwaq.runtime.policy import OnnxPolicy
from bpx_dreamwaq.runtime.routing import PolicyRouter
from .calibration import Calibration
from .config import vector
from .safety import StateGuard, quaternion_xyzw
from .state import Sample


def rotation_to_rpy(rotation):
    """ZYX 欧拉角；俯仰接近 ±90° 时 roll/yaw 并非唯一。"""
    r = np.asarray(rotation)
    pitch = math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0]))
    if math.hypot(r[0, 0], r[1, 0]) < 1e-8:
        return np.array([0.0, pitch, math.atan2(-r[0, 1], r[1, 1])])
    return np.array([math.atan2(r[2, 1], r[2, 2]), pitch, math.atan2(r[1, 0], r[0, 0])])


def attitude(config, sample):
    q = quaternion_xyzw(sample.quat_xyzw)
    norm = float(np.linalg.norm(q))
    raw = rotation_body_to_world(q[[3, 0, 1, 2]])
    settings = config.calibration["imu"]
    extrinsic = np.asarray(settings["sensor_to_body"]).reshape(3, 3)
    body = raw @ extrinsic.T if settings["quaternion_frame"] == "sensor_to_world" else raw
    result = {
        "quat_xyzw": q.tolist(),
        "norm": norm,
        "sdk_rotation": raw.tolist(),
        "body_rotation": body.tolist(),
        "sdk_rpy_deg": np.degrees(rotation_to_rpy(raw)).tolist(),
        "body_rpy_deg": np.degrees(rotation_to_rpy(body)).tolist(),
        "gravity_body": (body.T @ np.array([0.0, 0.0, -1.0])).tolist(),
    }
    if sample.omega is not None:
        result["omega_body"] = (extrinsic @ (vector(sample.omega, 3, "IMU omega") - settings["gyro_bias"])).tolist()
    return result


MOTION_NAMES = {
    0: "LyingDown / 已趴卧",
    1: "StandingUp / 起身",
    2: "Passive / 阻尼",
    3: "SitDown / 趴下中",
    6: "Motion / 运动模式",
}
GAIT_NAMES = {0: "Walk", 3: "Bipedal", 4: "Flip", 6: "WalkPhase", 7: "PoseTracking", 8: "Running", 12: "Jump"}


def feedback_control(config, sample):
    """反馈的固件状态和电脑手柄输入；不声称已知固件内部神经网络。"""
    telemetry = sample.telemetry
    motion, gait = telemetry.get("motion_state"), telemetry.get("gait")
    result = {
        "mode": "waiting",
        "policy": None,
        "motion": MOTION_NAMES.get(motion, f"未知 ({motion})" if motion is not None else "未提供"),
        "gait": GAIT_NAMES.get(gait, f"未知 ({gait})" if gait is not None else "未提供"),
        "control_source": (
            {0: "Unknown", 1: "RemoteControl / APP或遥控器", 2: "Navigator / 导航"}.get(
                telemetry.get("control_mode"), "未提供"
            )
        ),
        "remote_command": None,
        "input_source": telemetry.get("remote_command_source", "录制未包含手柄输入"),
        "reason": "等待机器人反馈和电脑手柄",
    }
    if telemetry.get("remote_command") is not None:
        try:
            result["remote_command"] = vector(telemetry["remote_command"], 3, "电脑手柄输入").tolist()
        except ValueError as exc:
            result["reason"] = str(exc)
            return result
    if not sample.connected:
        result["reason"] = "机器人连接已断开"
    elif motion is None:
        result["reason"] = "SDK/录制没有运动状态反馈"
    elif motion != 6:
        result["reason"] = result["motion"] + "；不应用策略预览"
    elif gait != 0:
        result["reason"] = "当前步态不属于本工程策略对应范围"
    elif result["remote_command"] is None:
        result["reason"] = telemetry.get("remote_command_error") or "没有有效电脑手柄输入"
    else:
        result["eligible"] = True
        result["reason"] = "由实时反馈和电脑手柄生成策略指令"
    return result


class PolicyPreview:
    """与部署共用指令路由和单模型接口；previous_action 来自预览输出。

    预览没有接管混合、力矩执行和闭环跟踪，不能视为真机正在执行策略。
    """

    def __init__(self, config, *, policy=None, enabled=True):
        self.config = config
        self.cal = Calibration(config)
        self.policy = None
        self.errors = {}
        if enabled:
            try:
                model = policy if policy is not None else OnnxPolicy(
                    config.model_path(), config.policy["observation_dimension"], 12
                )
                session = getattr(model, "session", None)
                if session is not None:
                    ins, outs = session.get_inputs(), session.get_outputs()
                    if (
                        len(ins) != 1
                        or len(outs) != 1
                        or ins[0].shape != [1, config.policy["observation_dimension"]]
                        or outs[0].shape != [1, 12]
                    ):
                        raise ValueError("模型必须匹配配置的输入维度及 12 维输出")
                self.policy = model
            except Exception as exc:
                self.errors["policy"] = str(exc)
        else:
            self.errors["policy"] = "策略预览已关闭"
        self.reset()

    def reset_history(self):
        self.history = ObservationHistory(
            self.config.policy["history_length"], include_base_lin_vel=self.config.policy["include_base_lin_vel"]
        )
        self.last_action = np.zeros(12, dtype=np.float32)

    def reset(self):
        self.guard = StateGuard(self.config)
        self.router = PolicyRouter(self.config.control, self.config.period)
        self.control = {}
        self.reset_history()

    def step(self, sample):
        control = feedback_control(self.config, sample)
        if not control.get("eligible"):
            self.reset()
            self.control = control
            return {}
        self.control = control
        self.guard.check(sample)
        q = self.cal.position_from_sdk(sample.q)
        dq = self.cal.velocity_from_sdk(sample.dq)
        gravity, angular = self.cal.imu(sample.quat_xyzw, sample.omega)
        command = self.router.step(control["remote_command"])
        if self.router.mode == "stand":
            command = np.zeros(3, dtype=np.float32)
        control.update(mode=self.router.mode, policy="policy", model_command=command.tolist())
        if self.policy is None:
            return {}
        frame = actor_frame(
            angular,
            gravity,
            command,
            q,
            dq,
            self.last_action,
            self.cal.default,
            sample.velocity_body if self.config.policy["include_base_lin_vel"] else None,
        )
        observation = self.history.append(frame)
        try:
            action = vector(self.policy(observation), 12, "policy action").astype(np.float32)
            self.last_action = action.copy()
            target = self.cal.default + self.config.policy["action_scale"] * action
            outside = [
                j.name
                for j, v in zip(self.config.joints, target)
                if (j.minimum is not None and v < j.minimum) or (j.maximum is not None and v > j.maximum)
            ]
            self.errors.pop("policy", None)
            return {"policy": {
                "action": action.tolist(),
                "target_model": target.tolist(),
                "target_sdk": self.cal.position_to_sdk(target).tolist(),
                "command": command.tolist(),
                "outside_limits": outside,
            }}
        except Exception as exc:
            self.errors["policy"] = str(exc)
            self.reset_history()
            return {}


def build_snapshot(config, sample, probe=None, *, ready=True):
    """独立保留缺失的列，不把旧值或无效数据伪装成零。"""
    cal = Calibration(config)
    snapshot = {
        "sample": sample,
        "rows": [],
        "policies": {},
        "policy_errors": {},
        "attitude": None,
        "warnings": [],
        "control": feedback_control(config, sample),
    }

    def optional(values, size, name):
        if values is None:
            return None
        try:
            return vector(values, size, name)
        except ValueError as exc:
            snapshot["warnings"].append(str(exc))
            return None

    raw_q = optional(sample.q, 12, "SDK q")
    raw_dq = optional(sample.dq, 12, "SDK dq")
    raw_tau = optional(sample.torque, 12, "SDK torque")
    q = cal.position_from_sdk(raw_q) if raw_q is not None else None
    dq = cal.velocity_from_sdk(raw_dq) if raw_dq is not None else None
    for i, j in enumerate(config.joints):
        nominal_index = SDK_JOINT_NAMES.index(j.name)
        snapshot["rows"].append({
            "name": j.name,
            "policy_index": i,
            "sdk_index": j.sdk_index,
            "official_index": nominal_index,
            "sign": j.sign,
            "offset": j.offset,
            "q_sdk": float(raw_q[j.sdk_index]) if raw_q is not None else None,
            "q_official": float(raw_q[nominal_index]) if raw_q is not None else None,
            "q_model": float(q[i]) if q is not None else None,
            "dq_model": float(dq[i]) if dq is not None else None,
            "torque_sdk": float(raw_tau[j.sdk_index]) if raw_tau is not None else None,
            "default": float(cal.default[i]),
            "minimum": j.minimum,
            "maximum": j.maximum,
        })
    if sample.quat_xyzw is not None:
        try:
            snapshot["attitude"] = attitude(config, sample)
        except ValueError as exc:
            snapshot["warnings"].append(str(exc))
    if probe is not None:
        try:
            if not ready:
                raise RuntimeError("等待两次 joint/IMU 更新；尚未开始预览")
            snapshot["policies"] = probe.step(sample)
            snapshot["control"] = dict(probe.control)
        except (ValueError, RuntimeError) as exc:
            probe.reset()
            snapshot["warnings"].append(str(exc))
            snapshot["control"].update(mode="paused", policy=None, reason=str(exc))
        snapshot["policy_errors"] = dict(probe.errors)
    if not snapshot["control"].get("eligible"):
        reason = snapshot["control"]["reason"]
        if reason not in snapshot["warnings"]:
            snapshot["warnings"].append(reason)
    return snapshot


def demo_sample(config, elapsed, host_time, token):
    """明确标记的模拟反馈；策略仍使用配置中的真实模型。"""
    roll, pitch, yaw = 0.18 * math.sin(elapsed * 0.6), 0.14 * math.sin(elapsed * 0.4), 0.3 * math.sin(elapsed * 0.25)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    quat = [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]
    # 模拟原始 SDK 坐标，不使用用户 sign/offset 反向转换，避免把偏差隐藏掉。
    q = np.asarray(config.policy["default_joint_position"]) + 0.035 * np.sin(elapsed + np.arange(12) * 0.4)
    dq = 0.035 * np.cos(elapsed + np.arange(12) * 0.4)
    indices = [SDK_JOINT_NAMES.index(name) for name in POLICY_JOINT_NAMES]
    raw_q, raw_dq = np.empty(12), np.empty(12)
    raw_q[indices], raw_dq[indices] = q, dq
    return Sample(
        host_time,
        session=1,
        source="demo",
        q=raw_q.tolist(),
        dq=raw_dq.tolist(),
        torque=(0.5 * np.sin(elapsed + np.arange(12))).tolist(),
        quat_xyzw=quat,
        omega=[0.108 * math.cos(elapsed * 0.6), 0.056 * math.cos(elapsed * 0.4), 0.075 * math.cos(elapsed * 0.25)],
        acceleration=[0.0, 0.0, 9.81],
        velocity_body=[0.0, 0.0, 0.0],
        joint_token=token,
        imu_token=token,
        odom_token=token,
        joint_age_s=0.0,
        imu_age_s=0.0,
        odom_age_s=0.0,
        identity={"robot_model": "SIMULATED BPX", "serial_number": "DEMO", "sdk_version": "无 SDK 连接"},
        telemetry={
            "battery_level": 80,
            "motor_temperature": [25.0] * 12,
            "motion_state": 6,
            "gait": 0,
            "control_mode": 1,
            "motion_age_s": 0.0,
            "remote_command": [0.35, 0.0, 0.0] if 6 <= elapsed % 18 < 12 else [0.0, 0.0, 0.0],
            "remote_command_source": "DEMO · 模拟电脑手柄",
            "remote_command_error": "",
        },
    )
