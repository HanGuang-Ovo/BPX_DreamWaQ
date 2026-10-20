"""SDK 坐标和策略参数；不要求身份、固件保护或实机验证确认表。"""

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import tomllib

import numpy as np

from bpx_dreamwaq.runtime.joints import POLICY_JOINT_NAMES, SDK_JOINT_NAMES, DEFAULT_JOINT_POSITION

# 仅用于窗口中的官方模型范围对照，不限制实机接管或发送目标。
OFFICIAL_JOINT_LIMITS = ((-.8727, .8727),) * 4 + ((-.7854, 2.60661),) * 4 + ((-2.7531, -.5539),) * 4


def vector(value, size, name, *, positive=False):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} 要求 {size} 个有限数")
    if positive and np.any(array <= 0):
        raise ValueError(f"{name} 必须全部大于零")
    return array


def positive(value, name):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} 必须为有限正数")
    return result


@dataclass(frozen=True)
class JointCalibration:
    name: str
    sdk_index: int
    sign: int
    offset: float
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True)
class HardwareInitConfig:
    mode: str = "pd"
    duration: float = 4.0
    support_duration: float = 1.0
    support_depth: float = 0.19
    standing_depth: float = 0.38
    stiffness: float = 80.0
    damping: float = 2.0


@dataclass(frozen=True)
class Config:
    path: Path
    calibration_path: Path
    runtime: dict
    calibration: dict
    joints: tuple[JointCalibration, ...]

    @property
    def network(self):
        return self.runtime["network"]

    @property
    def policy(self):
        return self.runtime["policy"]

    @property
    def control(self):
        return self.runtime["control"]

    @property
    def monitor(self):
        return self.runtime.get("monitor", {})

    @property
    def init_controller(self):
        return HardwareInitConfig(**self.runtime["init_controller"])

    @property
    def period(self):
        return 1.0 / self.control["policy_rate_hz"]

    def resolve(self, value):
        p = Path(value).expanduser()
        return p.resolve() if p.is_absolute() else (self.path.parent / p).resolve()

    def model_path(self, mode):
        return self.resolve(self.policy[f"{mode}_model"])

    @property
    def alignment_verified(self):
        return self.calibration.get("alignment", {}).get("verified", False)

    def blockers(self, stage="deploy"):
        if stage == "observe":
            return []
        return [f"policy.{mode}_model 不存在" for mode in ("walk", "stand")
                if not self.model_path(mode).is_file()]


def load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    calibration_path = Path(raw["calibration_file"]).expanduser()
    if not calibration_path.is_absolute():
        calibration_path = path.parent / calibration_path
    with calibration_path.open("rb") as stream:
        calibration = tomllib.load(stream)
    calibration.setdefault("identity", {"serial_number": "", "robot_model": "", "firmware_version": ""})
    calibration.setdefault("alignment", {"verified": False})
    imu = calibration.setdefault("imu", {})
    for key, value in {"sensor_to_body": [1,0,0,0,1,0,0,0,1], "gyro_bias": [0,0,0],
                       "quaternion_frame": "body_to_world"}.items():
        imu.setdefault(key, value)
    # 旧文件仍可读取，但这些已经移除的保护项不参与运行。
    for key in ("safety", "firmware", "supervisor"):
        raw.pop(key, None)
    calibration.pop("odometry", None)
    net = raw["network"]
    if not isinstance(net["robot_ip"], str) or not net["robot_ip"]:
        raise ValueError("network.robot_ip 不能为空")
    for key in ("state_port", "joint_state_port", "tcp_local_port"):
        value = net[key]
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 65535:
            raise ValueError(f"network.{key} 端口无效")
    if not net["state_port"] or not net["joint_state_port"] or net["state_port"] == net["joint_state_port"]:
        raise ValueError("两个 UDP 端口必须非零且不同")
    if not isinstance(net["state_rate_hz"], int) or not 1 <= net["state_rate_hz"] <= 200:
        raise ValueError("network.state_rate_hz 要求 1..200")
    positive(net["connect_timeout_s"], "network.connect_timeout_s")
    policy = raw["policy"]
    if policy["history_length"] not in (1, 10) or not isinstance(policy["include_base_lin_vel"], bool):
        raise ValueError("支持 H=1/10 的 45/48 维单帧 Actor")
    dimension = (48 if policy["include_base_lin_vel"] else 45) * policy["history_length"]
    if policy["observation_dimension"] != dimension or tuple(policy["joint_names"]) != POLICY_JOINT_NAMES:
        raise ValueError("策略观测维度或关节顺序与模型契约不一致")
    if not np.array_equal(vector(policy["default_joint_position"], 12, "default_joint_position"), DEFAULT_JOINT_POSITION):
        raise ValueError("当前模型默认关节角必须为 roll=0/pitch=0.7/knee=-1.4")
    if policy["action_scale"] != 0.5:
        raise ValueError("当前模型的 action_scale 必须为 0.5")
    control = raw["control"]
    for key in ("gains_verified", "standing_handover_verified", "prone_handover_verified"):
        control.pop(key, None)
    control.pop("prone_joint_position", None)
    defaults = asdict(HardwareInitConfig())
    supplied = raw.get("init_controller", {})
    raw["init_controller"] = {key: supplied.get(key, value) for key, value in defaults.items()}
    init = raw["init_controller"]
    if init["mode"] != "pd":
        raise ValueError("实机起身使用 mode=pd")
    for key in defaults:
        if key != "mode":
            if key in ("stiffness", "damping"):
                if not math.isfinite(init[key]):
                    raise ValueError(f"init_controller.{key} 必须为有限数")
            else:
                positive(init[key], f"init_controller.{key}")
    if not init["support_duration"] < init["duration"]:
        raise ValueError("support_duration 必须小于 duration")
    if control["policy_rate_hz"] != 50:
        raise ValueError("当前策略及历史必须按 50 Hz 更新")
    positive(control["send_rate_hz"], "control.send_rate_hz")
    for key in ("kp", "kd"):
        vector(control[key], 12, key)
    for key in ("takeover_duration_s", "action_blend_duration_s", "walk_enter_delay_s",
                "walk_exit_delay_s", "minimum_walk_time_s", "stopping_duration_s"):
        positive(control[key], key)
    vector(control["command_limit"], 3, "command_limit", positive=True)
    vector(control["command_slew_rate"], 3, "command_slew_rate", positive=True)
    if not 0 <= control["walk_exit_threshold"] < control["walk_enter_threshold"]:
        raise ValueError("Walk 路由迟滞阈值无效")
    joints = []
    entries = calibration.get("joints", [])
    if any(j.get("name") not in POLICY_JOINT_NAMES for j in entries):
        raise ValueError("关节覆盖项包含未知名称")
    for i, name in enumerate(POLICY_JOINT_NAMES):
        matches = [j for j in entries if j["name"] == name]
        if len(matches) > 1:
            raise ValueError(f"重复关节 {name}")
        lo, hi = OFFICIAL_JOINT_LIMITS[i]
        item = {"sdk_index": SDK_JOINT_NAMES.index(name), "sign": 1, "offset": 0.,
                "minimum": lo, "maximum": hi, **(matches[0] if matches else {})}
        index = item["sdk_index"]
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 12:
            raise ValueError("sdk_index 要求 0..11")
        if item["sign"] not in (-1, 1) or isinstance(item["sign"], bool):
            raise ValueError("sign 必须为 +1/-1")
        offset = float(item["offset"])
        lo, hi = item.get("minimum"), item.get("maximum")
        if not math.isfinite(offset) or any(v is not None and not math.isfinite(v) for v in (lo, hi)):
            raise ValueError("关节坐标与显示范围必须为有限数")
        if (lo is None) != (hi is None) or (lo is not None and lo >= hi):
            raise ValueError("显示范围要求 minimum < maximum")
        joints.append(JointCalibration(name, index, item["sign"], offset, lo, hi))
    if sorted(j.sdk_index for j in joints) != list(range(12)):
        raise ValueError("SDK 索引必须为 0..11 的一一映射")
    matrix = vector(imu["sensor_to_body"], 9, "imu.sensor_to_body").reshape(3, 3)
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(matrix), 1):
        raise ValueError("imu.sensor_to_body 必须为旋转矩阵，det=+1")
    vector(imu["gyro_bias"], 3, "imu.gyro_bias")
    if imu["quaternion_frame"] not in ("sensor_to_world", "body_to_world"):
        raise ValueError("imu.quaternion_frame 要求 sensor_to_world/body_to_world")
    calibration["joints"] = [asdict(joint) for joint in joints]
    return Config(path, calibration_path.resolve(), raw, calibration, tuple(joints))
