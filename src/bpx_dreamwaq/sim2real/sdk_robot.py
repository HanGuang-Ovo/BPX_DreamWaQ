"""官方 SDK 适配。观察对象没有关节发送方法，运动对象单独创建。"""

import time

from .config import vector
from .safety import quaternion_xyzw
from .state import Sample


class SDKSource:
    def __init__(self, config, *, joint_control=False, sdk=None, clock=time.monotonic):
        self.config = config
        self.joint_control = joint_control
        self.clock = clock
        if sdk is None:
            try:
                import bpx_sdk as sdk
            except ImportError as exc:
                raise RuntimeError("请先安装与固件匹配的官方 bpx_sdk wheel，参见 docs/SIM2REAL.md") from exc
        self.sdk = sdk
        self.robot = sdk.JointLevelControl() if joint_control else sdk.RequestRobotState()
        self.session = 0
        self.was_connected = False
        self.tokens = {}
        self.updated_at = {}
        self.identity = {}
        self.command_started = False
        self.last_position = None

    def __enter__(self):
        net = self.config.network
        try:
            self.robot.setRobotIp(net["robot_ip"])
            self.robot.setRobotStateUploadPort(net["state_port"])
            self.robot.setTcpLocalPort(net["tcp_local_port"])
            self.robot.setRobotStateUploadRate(net["state_rate_hz"])
            if self.joint_control:
                self.robot.setJointStateUploadPort(net["joint_state_port"])
            if not self.robot.connect():
                raise RuntimeError("SDK 未能启动连接")
            deadline = self.clock() + net["connect_timeout_s"]
            while not self.robot.isConnected():
                if self.clock() >= deadline:
                    raise TimeoutError("等待真实 TCP 连接超时")
                time.sleep(0.005)
            self._connection(True)
            return self
        except BaseException:
            self.robot.disconnect()
            raise

    def _connection(self, connected):
        if connected and not self.was_connected:
            self.session += 1
            self.tokens.clear()
            self.updated_at.clear()
            model = self.robot.getRobotModel()
            version = self.robot.getRobotVersion()
            self.identity = {
                "serial_number": self.robot.getRobotSerialNumber(),
                "robot_model": {1: "BPX", 2: "BPXPro", 3: "BPW"}.get(model, "Unknown") if model is not None else None,
                "firmware_version": ".".join(str(v) for v in version[:3]) if version else None,
                "sdk_version": getattr(self.sdk, "__version__", None),
            }
        if not connected:
            self.tokens.clear()
            self.updated_at.clear()
        self.was_connected = connected

    def _age(self, name, token, now):
        if token is None:
            return None
        if name not in self.tokens or token != self.tokens[name]:
            self.tokens[name] = token
            self.updated_at[name] = now
        return max(0.0, now - self.updated_at[name])

    def _group(self, timestamp_getter, getters):
        for _ in range(3):
            before = timestamp_getter()
            values = [getter() for getter in getters]
            after = timestamp_getter()
            if before is not None and before == after:
                return after, values
        return None, [None] * len(getters)

    def read(self):
        now = self.clock()
        connected = bool(self.robot.isConnected())
        self._connection(connected)
        if not connected:
            return Sample(now, self.session, connected=False, identity=self.identity)
        r = self.robot
        if self.joint_control:
            joint_token, values = self._group(
                r.getJointStateSeqHighRate,
                [
                    r.getJointPositionHighRate,
                    r.getJointVelocityHighRate,
                    r.getJointTorqueHighRate,
                    r.getImuQuatHighRate,
                    r.getImuOmegaHighRate,
                    r.getImuAccHighRate,
                    r.getJointStateTimestampHighRate,
                ],
            )
            q, dq, torque, quat, omega, acc, high_timestamp = values
            imu_token = joint_token
        else:
            joint_token, (q, dq, torque) = self._group(
                r.getJointStateTimestamp,
                [
                    r.getJointPosition,
                    r.getJointVelocity,
                    r.getJointTorque,
                ],
            )
            imu_token, (quat, omega, acc) = self._group(
                r.getImuTimestamp,
                [
                    r.getImuQuat,
                    r.getImuOmega,
                    r.getImuAcc,
                ],
            )
        odom_token, (odom,) = self._group(r.getOdometryTimestamp, [r.getLegOdom])
        velocity = odom.get("velocity_body") if odom else None
        odom_position = odom.get("position") if odom else None
        if odom_position is not None:
            odom_position = vector(odom_position, 3, "SDK odometry position").tolist()
        arrays = [(q, 12), (dq, 12), (torque, 12), (quat, 4), (omega, 3), (acc, 3), (velocity, 3)]
        q, dq, torque, quat, omega, acc, velocity = [
            vector(value, width, "SDK feedback").tolist() if value is not None else None for value, width in arrays
        ]
        if quat is not None:
            quaternion_xyzw(quat)
        if q is not None:
            self.last_position = q.copy()
        # host_time 是本次读缓存时间，新鲜度是主机首次观察到 token 变化的时间。
        now = self.clock()
        joint_age = self._age("joint", joint_token, now) if q is not None and dq is not None else None
        imu_age = self._age("imu", imu_token, now) if quat is not None and omega is not None else None
        odom_age = self._age("odom", odom_token, now) if velocity is not None else None
        motion_token, (motion_state, gait) = self._group(
            r.getMotionStateTimestamp,
            [
                r.getCurrentMotionState,
                r.getCurrentGait,
            ],
        )
        telemetry = {
            "motion_state": motion_state,
            "gait": gait,
            "motion_age_s": self._age("motion", motion_token, now),
            "control_mode": r.getControlMode(),
            "battery_level": r.getBatteryLevel(),
            "motor_temperature": r.getMotorTemperature(),
            "driver_temperature": r.getDriverTemperature(),
        }
        for key in ("motor_temperature", "driver_temperature"):
            if telemetry[key] is not None:
                telemetry[key] = vector(telemetry[key], 12, key).tolist()
        if self.joint_control:
            telemetry["high_rate_timestamp"] = high_timestamp
        return Sample(
            now,
            self.session,
            "high_rate" if self.joint_control else "ordinary",
            True,
            q,
            dq,
            torque,
            quat,
            omega,
            acc,
            velocity,
            joint_token,
            imu_token,
            odom_token,
            joint_age,
            imu_age,
            odom_age,
            telemetry,
            dict(self.identity),
            odom_position=odom_position,
        )

    def send(self, kp, position, kd, velocity, feedforward):
        if not self.joint_control:
            raise RuntimeError("只读连接禁止发送关节命令")
        if not self.robot.isConnected():
            raise RuntimeError("发送前 TCP 已断开")
        arrays = [
            vector(v, 12, name).tolist()
            for name, v in zip(
                ("kp", "position", "kd", "velocity", "feedforward"), (kp, position, kd, velocity, feedforward)
            )
        ]
        # 发送可能已到达机器人，即使返回失败也需要清理。
        self.command_started = True
        if not self.robot.setJointCommand(*arrays):
            raise RuntimeError("SDK 完整关节命令发送失败")

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.joint_control and self.command_started:
                # best effort：退出动作失败不覆盖原始异常，也不自动重连。
                for _ in range(3):
                    try:
                        if not self.robot.isConnected():
                            break
                        pos = self.last_position if self.last_position is not None else [0.] * 12
                        self.robot.setJointCommand([0.] * 12, pos, [1.] * 12,
                                                   [0.] * 12, [0.] * 12)
                    except Exception:
                        pass
        finally:
            self.robot.disconnect()
        return False
