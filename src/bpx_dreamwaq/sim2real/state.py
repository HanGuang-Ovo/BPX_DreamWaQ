"""原始 SDK 状态与逐数据组新鲜度。host_time 使用主机单调时钟。"""

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Sample:
    host_time: float
    session: int = 0
    source: str = "ordinary"
    connected: bool = True
    q: list | None = None
    dq: list | None = None
    torque: list | None = None
    quat_xyzw: list | None = None
    omega: list | None = None
    acceleration: list | None = None
    velocity_body: list | None = None
    joint_token: int | float | None = None
    imu_token: int | float | None = None
    odom_token: int | float | None = None
    joint_age_s: float | None = None
    imu_age_s: float | None = None
    odom_age_s: float | None = None
    telemetry: dict = field(default_factory=dict)
    identity: dict = field(default_factory=dict)
    odom_position: list | None = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})
