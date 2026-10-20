"""sim2sim 配置读取与校验。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib

from bpx_dreamwaq.runtime.supervisor import SupervisorConfig
from bpx_dreamwaq.runtime.init_controller import InitControllerConfig
from bpx_dreamwaq.paths import PROJECT_ROOT


@dataclass(frozen=True)
class PathConfig:
    """MuJoCo 模型和统一策略的路径；策略格式由推理后端决定。"""

    mjcf: Path
    policy: Path


@dataclass(frozen=True)
class SimulationConfig:
    timestep: float
    decimation: int
    duration: float
    realtime: bool
    integrator: str

    @property
    def policy_dt(self) -> float:
        return self.timestep * self.decimation


@dataclass(frozen=True)
class ControlConfig:
    mode: str
    action_scale: float
    stiffness: float
    damping: float
    torque_limit: float
    velocity_limit: float


@dataclass(frozen=True)
class InitialStateConfig:
    base_position: tuple[float, float, float]
    base_quaternion_wxyz: tuple[float, float, float, float]
    joint_position: tuple[float, ...]
    default_joint_position: tuple[float, ...]


@dataclass(frozen=True)
class ObservationConfig:
    dimension: int
    action_dimension: int
    history_length: int = 1
    include_base_lin_vel: bool = True

    @property
    def frame_dimension(self) -> int:
        return self.dimension // self.history_length


@dataclass(frozen=True)
class CommandConfig:
    linear_x: float
    linear_y: float
    angular_z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.linear_x, self.linear_y, self.angular_z)


@dataclass(frozen=True)
class TerminationConfig:
    minimum_base_height: float
    maximum_tilt: float


@dataclass(frozen=True)
class TerrainConfig:
    kind: str = "flat"
    seed: int = 20260918


@dataclass(frozen=True)
class Sim2SimConfig:
    repository_root: Path
    paths: PathConfig
    simulation: SimulationConfig
    control: ControlConfig
    initial_state: InitialStateConfig
    observation: ObservationConfig
    command: CommandConfig
    termination: TerminationConfig
    supervisor: SupervisorConfig
    joint_names: tuple[str, ...]
    base_body_name: str
    base_joint_name: str
    floor_geom_name: str
    terrain: TerrainConfig = TerrainConfig()
    init_controller: InitControllerConfig = InitControllerConfig()


def _tuple_of_floats(value: object, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} 必须包含 {length} 个数值")
    return tuple(float(item) for item in value)


def _resolve_path(repository_root: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} 必须是非空路径")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (repository_root / path).resolve()


def default_repository_root() -> Path:
    """根据本文件位置返回仓库根目录。"""

    return PROJECT_ROOT


def load_config(path: str | Path, repository_root: str | Path | None = None) -> Sim2SimConfig:
    """读取 TOML 配置，并把资源路径解析为绝对路径。"""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)

    root = Path(repository_root).expanduser().resolve() if repository_root else default_repository_root()
    paths = raw["paths"]
    simulation = raw["simulation"]
    control = raw["control"]
    initial = raw["initial_state"]
    observation = raw["observation"]
    command = raw["command"]
    termination = raw["termination"]
    supervisor = raw["supervisor"]
    robot = raw["robot"]

    joint_names = tuple(str(name) for name in robot["joint_names"])
    default_joint_position = _tuple_of_floats(
        initial["default_joint_position"], len(joint_names), "initial_state.default_joint_position"
    )
    # joint_position 只决定 MuJoCo reset 姿态；default_joint_position 仍是训练时
    # JointPositionActionCfg 的动作零点。旧 TOML 没有该键时保持原有站姿复位行为。
    joint_position = _tuple_of_floats(
        initial.get("joint_position", initial["default_joint_position"]),
        len(joint_names),
        "initial_state.joint_position",
    )
    cfg = Sim2SimConfig(
        repository_root=root,
        paths=PathConfig(
            mjcf=_resolve_path(root, paths["mjcf"], "paths.mjcf"),
            policy=_resolve_path(root, paths.get("policy"), "paths.policy"),
        ),
        simulation=SimulationConfig(
            timestep=float(simulation["timestep"]),
            decimation=int(simulation["decimation"]),
            duration=float(simulation["duration"]),
            realtime=bool(simulation["realtime"]),
            integrator=str(simulation["integrator"]),
        ),
        control=ControlConfig(
            mode=str(control["mode"]),
            action_scale=float(control["action_scale"]),
            stiffness=float(control["stiffness"]),
            damping=float(control["damping"]),
            torque_limit=float(control["torque_limit"]),
            velocity_limit=float(control["velocity_limit"]),
        ),
        initial_state=InitialStateConfig(
            base_position=_tuple_of_floats(initial["base_position"], 3, "initial_state.base_position"),
            base_quaternion_wxyz=_tuple_of_floats(
                initial["base_quaternion_wxyz"], 4, "initial_state.base_quaternion_wxyz"
            ),
            joint_position=joint_position,
            default_joint_position=default_joint_position,
        ),
        observation=ObservationConfig(
            dimension=int(observation["dimension"]),
            action_dimension=int(observation["action_dimension"]),
            history_length=int(observation.get("history_length", 1)),
            include_base_lin_vel=observation.get("include_base_lin_vel", True),
        ),
        command=CommandConfig(
            linear_x=float(command["linear_x"]),
            linear_y=float(command["linear_y"]),
            angular_z=float(command["angular_z"]),
        ),
        termination=TerminationConfig(
            minimum_base_height=float(termination["minimum_base_height"]),
            maximum_tilt=float(termination["maximum_tilt"]),
        ),
        supervisor=SupervisorConfig(
            command_scale=_tuple_of_floats(supervisor["command_scale"], 3, "supervisor.command_scale"),
            command_slew_rate=_tuple_of_floats(
                supervisor["command_slew_rate"], 3, "supervisor.command_slew_rate"
            ),
            walk_enter_threshold=float(supervisor["walk_enter_threshold"]),
            walk_exit_threshold=float(supervisor["walk_exit_threshold"]),
            walk_enter_delay=float(supervisor["walk_enter_delay"]),
            walk_exit_delay=float(supervisor["walk_exit_delay"]),
            minimum_walk_time=float(supervisor["minimum_walk_time"]),
            stop_linear_speed=float(supervisor["stop_linear_speed"]),
            stop_angular_speed=float(supervisor["stop_angular_speed"]),
            stand_confirm_time=float(supervisor["stand_confirm_time"]),
            stopping_timeout=float(supervisor["stopping_timeout"]),
            fall_base_height=float(supervisor["fall_base_height"]),
            fall_tilt=float(supervisor["fall_tilt"]),
            init_start_max_tilt=float(supervisor["init_start_max_tilt"]),
            init_abort_tilt=float(supervisor.get("init_abort_tilt", supervisor["fall_tilt"])),
            init_base_height=float(
                supervisor.get("init_base_height", supervisor.get("recovery_base_height"))
            ),
            init_tilt=float(supervisor.get("init_tilt", supervisor.get("recovery_tilt"))),
            init_confirm_time=float(
                supervisor.get("init_confirm_time", supervisor.get("recovery_confirm_time"))
            ),
            action_blend_duration=float(supervisor["action_blend_duration"]),
        ),
        joint_names=joint_names,
        base_body_name=str(robot["base_body_name"]),
        base_joint_name=str(robot["base_joint_name"]),
        floor_geom_name=str(robot["floor_geom_name"]),
        terrain=TerrainConfig(**raw.get("terrain", {})),
        init_controller=InitControllerConfig(**raw.get("init_controller", {})),
    )

    init = cfg.init_controller
    if init.mode != "pd":
        raise ValueError("init_controller.mode 必须为 pd；起身使用足端轨迹 + PD")
    for name in ("duration", "support_duration", "support_depth", "standing_depth", "stiffness",
                 "damping", "torque_limit", "joint_speed_limit", "tracking_error_limit", "timeout"):
        value = getattr(init, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"init_controller.{name} 必须为有限正数")
    if not init.support_duration < init.duration < init.timeout:
        raise ValueError("Init 要求 support_duration < duration < timeout")
    if init.support_depth >= init.standing_depth:
        raise ValueError("Init support_depth 必须小于 standing_depth")
    if init.mode == "pd" and cfg.control.mode != "explicit":
        raise ValueError("足端轨迹 Init 需要 control.mode = explicit")

    if cfg.terrain.kind not in ("flat", "three_level"):
        raise ValueError("terrain.kind must be flat or three_level")
    if not isinstance(cfg.terrain.seed, int) or cfg.terrain.seed < 0:
        raise ValueError("terrain.seed must be a nonnegative integer")
    if cfg.simulation.timestep <= 0.0 or cfg.simulation.decimation <= 0:
        raise ValueError("仿真步长和 decimation 必须大于零")
    if cfg.simulation.integrator not in ("euler", "implicit", "implicitfast", "rk4"):
        raise ValueError(f"不支持的 MuJoCo 积分器：{cfg.simulation.integrator}")
    if cfg.control.mode not in ("explicit", "implicit"):
        raise ValueError("control.mode 必须是 explicit 或 implicit")
    if not math.isfinite(cfg.control.action_scale) or cfg.control.action_scale <= 0.0:
        raise ValueError("control.action_scale 必须是有限正数")
    if any(not math.isfinite(value) or value <= 0.0 for value in cfg.supervisor.command_scale):
        raise ValueError("supervisor.command_scale 必须全部大于零")
    if any(not math.isfinite(value) or value <= 0.0 for value in cfg.supervisor.command_slew_rate):
        raise ValueError("supervisor.command_slew_rate 必须全部大于零")
    if not 0.0 <= cfg.supervisor.walk_exit_threshold < cfg.supervisor.walk_enter_threshold:
        raise ValueError("supervisor 的退出阈值必须非负且小于进入阈值")
    positive_supervisor_times = (
        cfg.supervisor.walk_enter_delay,
        cfg.supervisor.walk_exit_delay,
        cfg.supervisor.stand_confirm_time,
        cfg.supervisor.stopping_timeout,
        cfg.supervisor.init_confirm_time,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive_supervisor_times):
        raise ValueError("supervisor 的判定时间参数必须大于零")
    nonnegative_supervisor_times = (
        cfg.supervisor.minimum_walk_time,
        cfg.supervisor.action_blend_duration,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in nonnegative_supervisor_times):
        raise ValueError("supervisor 的驻留和混合时间不能小于零")
    if cfg.supervisor.stopping_timeout < cfg.supervisor.stand_confirm_time:
        raise ValueError("supervisor.stopping_timeout 不能小于 stand_confirm_time")
    supervisor_thresholds = (
        cfg.supervisor.walk_enter_threshold,
        cfg.supervisor.walk_exit_threshold,
        cfg.supervisor.stop_linear_speed,
        cfg.supervisor.stop_angular_speed,
        cfg.supervisor.fall_base_height,
        cfg.supervisor.fall_tilt,
        cfg.supervisor.init_start_max_tilt,
        cfg.supervisor.init_abort_tilt,
        cfg.supervisor.init_base_height,
        cfg.supervisor.init_tilt,
    )
    if not all(math.isfinite(value) for value in supervisor_thresholds):
        raise ValueError("supervisor 阈值必须为有限数")
    if cfg.supervisor.stop_linear_speed < 0.0 or cfg.supervisor.stop_angular_speed < 0.0:
        raise ValueError("supervisor 的停止速度阈值不能小于零")
    if cfg.supervisor.fall_base_height <= 0.0 or cfg.supervisor.init_base_height <= 0.0:
        raise ValueError("supervisor 的高度阈值必须大于零")
    if cfg.supervisor.fall_tilt <= 0.0 or cfg.supervisor.init_tilt < 0.0:
        raise ValueError("supervisor 的倾角阈值无效")
    if not 0.0 <= cfg.supervisor.init_start_max_tilt < cfg.supervisor.fall_tilt:
        raise ValueError("supervisor.init_start_max_tilt 必须非负且小于 fall_tilt")
    if not cfg.supervisor.init_start_max_tilt < cfg.supervisor.init_abort_tilt <= cfg.supervisor.fall_tilt:
        raise ValueError(
            "supervisor.init_abort_tilt 必须大于 init_start_max_tilt 且不大于 fall_tilt"
        )
    if cfg.supervisor.init_base_height <= cfg.supervisor.fall_base_height:
        raise ValueError("supervisor.init_base_height 必须大于 fall_base_height")
    if cfg.supervisor.init_tilt >= cfg.supervisor.fall_tilt:
        raise ValueError("supervisor.init_tilt 必须小于 fall_tilt")
    if cfg.observation.history_length < 1:
        raise ValueError("observation.history_length 必须为正整数")
    if not isinstance(cfg.observation.include_base_lin_vel, bool):
        raise ValueError("observation.include_base_lin_vel 必须为布尔值")
    base_width = 12 if cfg.observation.include_base_lin_vel else 9
    expected_observation_dimension = (base_width + 3 * len(cfg.joint_names)) * cfg.observation.history_length
    if cfg.observation.dimension != expected_observation_dimension:
        raise ValueError(
            f"观测维度应为 ({base_width} + 3 * 关节数) * history_length = {expected_observation_dimension}，"
            f"实际配置为 {cfg.observation.dimension}"
        )
    if cfg.observation.action_dimension != len(cfg.joint_names):
        raise ValueError("动作维度必须等于受控关节数")
    return cfg
