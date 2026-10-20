"""离线回归：使用模拟 SDK，绝不连接机器人。"""

from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from pathlib import Path
import copy
import io
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bpx_dreamwaq.runtime.history import ObservationHistory
from bpx_dreamwaq.runtime.joints import DEFAULT_JOINT_POSITION
from bpx_dreamwaq.runtime.supervisor import BehaviorMode, SupervisorDecision
from bpx_dreamwaq.sim2real.calibration import Calibration
from bpx_dreamwaq.sim2real.cli import main
from bpx_dreamwaq.sim2real.config import load_config
from bpx_dreamwaq.sim2real.controller import ActorController
from bpx_dreamwaq.sim2real.recording import JsonlWriter, load_samples, resample
from bpx_dreamwaq.sim2real.runner import replay, run_live
from bpx_dreamwaq.sim2real.safety import StateGuard
from bpx_dreamwaq.sim2real.sdk_robot import SDKSource
from bpx_dreamwaq.sim2real.state import Sample


def config(ready=False):
    return load_config(ROOT / "configs/sim2real/runtime.toml")


def sample(cfg=None, t=0.0, token=1, **changes):
    cfg = cfg or config()
    q = Calibration(cfg).position_to_sdk(DEFAULT_JOINT_POSITION).tolist()
    data = dict(
        host_time=t,
        session=1,
        source="high_rate",
        q=q,
        dq=[0.0] * 12,
        torque=[0.0] * 12,
        quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        omega=[0.0] * 3,
        acceleration=[0.0, 0.0, 9.81],
        joint_token=token,
        imu_token=token,
        joint_age_s=0.0,
        imu_age_s=0.0,
        odom_token=token,
        odom_age_s=0.0,
        velocity_body=[0., 0., 0.],
        odom_position=[0., 0., .37],
        identity={**cfg.calibration["identity"], "sdk_version": "1.0.9"},
        telemetry={"motion_state": 6},
    )
    return Sample(**{**data, **changes})


class Policy:
    def __init__(self, value=0.2):
        self.value = value
        self.observations = []

    def __call__(self, obs):
        self.observations.append(obs.copy())
        return np.full(12, self.value, dtype=np.float32)


class Clock:
    def __init__(self):
        self.value = 0.0

    def now(self):
        return self.value

    def sleep(self, delta):
        self.value += delta


class Source:
    def __init__(self, cfg, clock, disconnect_at=None):
        self.cfg, self.clock = cfg, clock
        self.q = sample(cfg).q
        self.commands = []
        self.reads = 0
        self.exited = False
        self.disconnect_at = disconnect_at

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.exited = True

    def read(self):
        self.reads += 1
        connected = self.disconnect_at is None or self.clock.now() < self.disconnect_at
        return sample(self.cfg, t=self.clock.now(), token=self.reads, q=list(self.q), connected=connected)

    def send(self, kp, target, kd, velocity, ff):
        if self.disconnect_at is not None and self.clock.now() >= self.disconnect_at:
            raise RuntimeError("SDK 连接已断开")
        self.commands.append(
            (self.clock.now(), np.asarray(kp).copy(), np.asarray(target).copy(), np.asarray(kd).copy())
        )
        self.q = list(target)


class RuntimeTest(unittest.TestCase):
    def test_missing_identity_and_confirmation_flags_do_not_block_live(self):
        cfg, clock = config(), Clock()
        self.assertEqual(cfg.calibration["identity"]["serial_number"], "")
        # 模型存在性与身份确认分开验证；实际推理由下方注入策略完成。
        with patch("bpx_dreamwaq.sim2real.config.Path.is_file", return_value=True):
            self.assertEqual(cfg.blockers(), [])
        source = Source(cfg, clock)
        with tempfile.TemporaryDirectory() as tmp:
            run_live(cfg, Path(tmp)/"log.jsonl", (0,0,0), .03,
                     policies={"stand": Policy(0), "walk": Policy(0)},
                     source_factory=lambda *a, **k: source, clock=clock.now, sleep=clock.sleep)
        self.assertTrue(source.commands)

    def test_preflight_never_construct_sdk(self):
        with patch(
            "bpx_dreamwaq.sim2real.sdk_robot.SDKSource.__init__", side_effect=AssertionError("hardware")
        ), redirect_stdout(io.StringIO()):
            self.assertEqual(main(["check"]), 0)
            self.assertEqual(main(["run"]), 0)

    def test_mapping_sign_offset_round_trip(self):
        cfg = config()
        joints = tuple(replace(j, sign=(-1 if i % 2 else 1), offset=0.01 * i) for i, j in enumerate(cfg.joints))
        cal = Calibration(replace(cfg, joints=joints))
        expected = np.arange(12) * 0.1
        sdk = cal.position_to_sdk(expected)
        np.testing.assert_allclose(cal.position_from_sdk(sdk), expected)
        dq = np.arange(12) * 0.2
        sdk_dq = np.empty(12)
        sdk_dq[cal.indices] = cal.sign * dq
        np.testing.assert_allclose(cal.velocity_from_sdk(sdk_dq), dq)
        # 增益只重排，不随方向符号变成负数。
        np.testing.assert_allclose(cal.gains_to_sdk(np.arange(12))[cal.indices], np.arange(12))

    def test_imu_rotation_and_bias(self):
        cfg = config()
        cal = copy.deepcopy(cfg.calibration)
        cal["imu"].update(
            sensor_to_body=[0, -1, 0, 1, 0, 0, 0, 0, 1], gyro_bias=[0.1, 0, 0], quaternion_frame="sensor_to_world"
        )
        gravity, omega = Calibration(replace(cfg, calibration=cal)).imu([0, 0, 0, 1], [0.3, 0, 0])
        np.testing.assert_allclose(gravity, [0, 0, -1])
        np.testing.assert_allclose(omega, [0, 0.2, 0], atol=1e-8)

    def test_state_guard_rejects_nonfinite_data_and_invalid_quaternion_only(self):
        cfg = config()
        guard = StateGuard(cfg)
        good = sample(cfg)
        for changes in ({"q": [float("nan")]*12}, {"dq": [float("inf")]*12},
                        {"omega": [float("nan")]*3}, {"torque": [float("inf")]*12},
                        {"acceleration": [float("nan")]*3},
                        {"quat_xyzw": [0,0,0,.49]}, {"quat_xyzw": [0,0,0,1.51]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                guard.check(replace(good, **changes))
        for scale in (.5,.7,1.,1.3,1.5):
            altered = replace(good, quat_xyzw=[0,0,0,scale], q=[10.]*12, dq=[1000.]*12,
                              session=99, identity={}, source="ordinary", joint_age_s=100.,
                              imu_age_s=None, odom_token=None, odom_position=None,
                              velocity_body=None)
            guard.check(altered, now=1000., hardware=True)
            gravity, _ = Calibration(cfg).imu(altered.quat_xyzw, altered.omega)
            np.testing.assert_allclose(gravity, [0,0,-1])

    def test_large_finite_policy_target_is_not_joint_clipped_or_rate_limited(self):
        cfg = config()
        controller = ActorController(cfg, policies={"stand": Policy(100.), "walk": Policy(100.)})
        result = None
        for tick in range(120):
            result = controller.step(sample(cfg, t=tick*.02), (0,0,0))
            controller.commit_target(result["target_model"])
        expected = np.asarray(DEFAULT_JOINT_POSITION)+50.
        np.testing.assert_allclose(result["target_model"], expected)
        self.assertGreater(result["target_model"][8], cfg.joints[8].maximum)

    def test_nonfinite_policy_output_never_reaches_motor_command(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                cfg, clock = config(), Clock()
                source = Source(cfg, clock)
                policies = {"stand": Policy(value), "walk": Policy(value)}
                with self.assertRaisesRegex(RuntimeError, "Actor 输出无效"):
                    run_live(cfg, Path(tmp) / "log.jsonl", (0, 0, 0), .05,
                             policies=policies, source_factory=lambda *a, **k: source,
                             clock=clock.now, sleep=clock.sleep)
                self.assertEqual(source.commands, [])
                self.assertTrue(source.exited)

    def test_controller_first_command_holds_measured_pose(self):
        cfg = config()
        policy = Policy()
        controller = ActorController(cfg, policies={"walk": policy, "stand": policy})
        first = controller.step(sample(cfg), (0, 0, 0))
        np.testing.assert_allclose(first["target_model"], DEFAULT_JOINT_POSITION)
        self.assertEqual(len(first["observation"]), 450)
        self.assertEqual(first["takeover_alpha"], 0.0)

    def test_history_receives_processed_action(self):
        cfg = config()
        policy = Policy()
        controller = ActorController(cfg, policies={"walk": policy, "stand": policy})
        controller.step(sample(cfg), (0, 0, 0))
        target = np.array(DEFAULT_JOINT_POSITION) + 0.03
        controller.commit_target(target)
        controller.step(sample(cfg, t=0.02, token=2), (0, 0, 0))
        np.testing.assert_allclose(policy.observations[-1][-12:], 0.06, atol=1e-7)

    def test_actor_bounds_external_supervisor_command_before_observation(self):
        cfg = config()
        limits = np.asarray(cfg.control["command_limit"])
        for sign in (1., -1.):
            with self.subTest(sign=sign):
                policy = Policy(0.)
                controller = ActorController(cfg, policies={"walk": policy, "stand": policy})
                raw = sign * np.array([5., -2., 3.])
                decision = SupervisorDecision(BehaviorMode.WALK, raw, True, "test")
                result = controller.step(sample(cfg), raw, decision=decision)
                expected = np.clip(raw, -limits, limits)
                np.testing.assert_allclose(result["command"], expected)
                commands = policy.observations[-1][60:90].reshape(10, 3)
                np.testing.assert_allclose(commands, np.tile(expected, (10, 1)))

    def test_actor_rejects_invalid_supervisor_command_before_clipping(self):
        cfg = config()
        for command in ([np.inf, 0., 0.], [np.nan, 0., 0.], [1., 2.]):
            with self.subTest(command=command):
                policy = Policy(0.)
                controller = ActorController(cfg, policies={"walk": policy, "stand": policy})
                decision = SupervisorDecision(BehaviorMode.WALK, np.asarray(command), True, "test")
                with self.assertRaises(ValueError):
                    controller.step(sample(cfg), (0., 0., 0.), decision=decision)
                self.assertEqual(policy.observations, [])

    def test_replay_resampling_has_no_future_state_and_increases_age(self):
        cfg = config()
        samples = [sample(cfg, t=0, telemetry={"motion_age_s": 0.01}), sample(cfg, t=0.04, token=2)]
        result = list(resample(samples, 0.02))
        self.assertEqual([s.joint_token for s in result], [1, 1, 2])
        self.assertAlmostEqual(result[1].joint_age_s, 0.02)
        self.assertAlmostEqual(result[1].telemetry["motion_age_s"], 0.03)

    def test_offline_replay_with_injected_policy(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as tmp:
            path, output = Path(tmp) / "state.jsonl", Path(tmp) / "policy.jsonl"
            with JsonlWriter(path, {"mode": "test"}) as writer:
                for i in range(5):
                    writer.write({"type": "sample", "sample": sample(cfg, t=i * 0.02, token=i + 1).to_dict()})
            policy = Policy()
            with patch("bpx_dreamwaq.sim2real.sdk_robot.SDKSource.__init__", side_effect=AssertionError("hardware")):
                count = replay(cfg, path, output, (0, 0, 0), policies={"walk": policy, "stand": policy})
            self.assertEqual(count, 5)
            self.assertTrue(output.is_file())

    def test_replay_skips_incomplete_initial_udp_prefix(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as tmp:
            path, output = Path(tmp) / "state.jsonl", Path(tmp) / "policy.jsonl"
            with JsonlWriter(path, {}) as writer:
                writer.write({"type": "sample", "sample": Sample(0.0, connected=True).to_dict()})
                for i in range(1, 4):
                    writer.write({"type": "sample", "sample": sample(cfg, t=i * 0.02, token=i).to_dict()})
            policy = Policy()
            count = replay(cfg, path, output, (0, 0, 0), policies={"walk": policy, "stand": policy})
            self.assertEqual(count, 3)
            self.assertEqual(json.loads(output.read_text().splitlines()[0])["skipped_initial_samples"], 1)

    def test_recording_never_overwrites_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.jsonl"
            path.write_text("keep")
            with self.assertRaises(FileExistsError):
                with JsonlWriter(path, {}):
                    pass
            self.assertEqual(path.read_text(), "keep")

    def test_live_schedule_separates_policy_and_send_rates(self):
        cfg, clock = config(True), Clock()
        source = Source(cfg, clock)
        policy = Policy(0.1)
        with tempfile.TemporaryDirectory() as tmp:
            count = run_live(
                cfg,
                Path(tmp) / "log.jsonl",
                (0, 0, 0),
                0.1,
                armed=True,
                policies={"walk": policy, "stand": policy},
                source_factory=lambda *a, **k: source,
                clock=clock.now,
                sleep=clock.sleep,
            )
        self.assertEqual(count, 5)
        self.assertEqual(len(source.commands), 10)
        self.assertTrue(source.exited)
        np.testing.assert_allclose(source.commands[0][2], sample(cfg).q)

    def test_live_disconnect_stops_without_recovery(self):
        cfg, clock = config(True), Clock()
        source = Source(cfg, clock, disconnect_at=0.025)
        policy = Policy()
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(RuntimeError, "断开"):
            run_live(
                cfg,
                Path(tmp) / "log.jsonl",
                (0, 0, 0),
                0.2,
                armed=True,
                policies={"walk": policy, "stand": policy},
                source_factory=lambda *a, **k: source,
                clock=clock.now,
                sleep=clock.sleep,
            )
        self.assertLessEqual(len(source.commands), 3)
        self.assertTrue(source.exited)

    def test_live_inference_overrun_does_not_add_a_motion_stop(self):
        cfg, clock = config(), Clock()
        source = Source(cfg, clock)
        def slow(obs):
            clock.sleep(.03)
            return np.zeros(12)
        with tempfile.TemporaryDirectory() as tmp:
            run_live(cfg, Path(tmp)/"log.jsonl", (0,0,0), .12,
                     policies={"stand": slow, "walk": slow}, source_factory=lambda *a, **k: source,
                     clock=clock.now, sleep=clock.sleep)
        self.assertTrue(source.commands)
        self.assertTrue(source.exited)

    def test_config_still_requires_a_usable_joint_mapping(self):
        original = (ROOT / "configs/sim2real/runtime.toml").read_text()
        calibration = (ROOT / "configs/sim2real/joint_calibration.template.toml").read_text()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"runtime.toml"
            path.write_text(original)
            (Path(tmp)/"joint_calibration.template.toml").write_text(
                calibration+'\n[[joints]]\nname = "fr_hip_roll_joint"\nsdk_index = 0\n')
            with self.assertRaisesRegex(ValueError, "一一映射"):
                load_config(path)


class SDKTest(unittest.TestCase):
    def robot(self):
        r = Mock()
        r.connect.return_value = True
        r.isConnected.return_value = True
        r.getRobotModel.return_value = 1
        r.getRobotVersion.return_value = (1, 0, 15, 0, 0, 0)
        r.getRobotSerialNumber.return_value = "test-SN"
        r.getJointStateTimestamp.return_value = 10
        r.getImuTimestamp.return_value = 20
        r.getOdometryTimestamp.return_value = 30
        r.getMotionStateTimestamp.return_value = 40
        r.getJointPosition.return_value = [0.0] * 12
        r.getJointVelocity.return_value = [0.0] * 12
        r.getJointTorque.return_value = [0.0] * 12
        r.getImuQuat.return_value = [0, 0, 0, 1]
        r.getImuOmega.return_value = [0.0] * 3
        r.getImuAcc.return_value = [0.0, 0.0, 9.81]
        r.getLegOdom.return_value = {"velocity_body": [0.0] * 3}
        r.getMotorTemperature.return_value = [20.0] * 12
        r.getDriverTemperature.return_value = [20.0] * 12
        r.getCurrentMotionState.return_value = 0
        r.getCurrentGait.return_value = 0
        r.getControlMode.return_value = 1
        r.getBatteryLevel.return_value = 80
        r.setJointCommand.return_value = True
        r.setZeroJointCommand.return_value = True
        return r

    def test_readonly_object_has_no_motor_command_calls(self):
        r, clock = self.robot(), Clock()
        sdk = SimpleNamespace(RequestRobotState=Mock(return_value=r), JointLevelControl=Mock(), __version__="1.0.9")
        with SDKSource(config(), sdk=sdk, clock=clock.now) as source:
            source.read()
            clock.sleep(0.12)
            stale = source.read()
            self.assertAlmostEqual(stale.joint_age_s, 0.12)
            with self.assertRaisesRegex(RuntimeError, "只读"):
                source.send(*[[0.0] * 12] * 5)
        sdk.JointLevelControl.assert_not_called()
        r.setJointCommand.assert_not_called()
        r.setZeroJointCommand.assert_not_called()
        r.disconnect.assert_called_once()

    def test_high_rate_snapshot_retries_on_mid_read_update(self):
        r = self.robot()
        r.getJointStateSeqHighRate.side_effect = [1, 2, 2, 2]
        for name, size in [
            ("JointPosition", 12),
            ("JointVelocity", 12),
            ("JointTorque", 12),
            ("ImuQuat", 4),
            ("ImuOmega", 3),
            ("ImuAcc", 3),
        ]:
            getattr(r, "get" + name + "HighRate").return_value = [0.0] * size
        r.getImuQuatHighRate.return_value = [0, 0, 0, 1]
        r.getJointStateTimestampHighRate.return_value = 1.25
        sdk = SimpleNamespace(JointLevelControl=Mock(return_value=r), __version__="1.0.9")
        with SDKSource(config(True), joint_control=True, sdk=sdk) as source:
            state = source.read()
            self.assertEqual(state.joint_token, 2)
            self.assertEqual(r.getJointPositionHighRate.call_count, 2)

    def test_damping_is_attempted_after_send_failure_without_zero_command(self):
        r = self.robot()
        r.setJointCommand.side_effect = [False, True, True, True]
        sdk=SimpleNamespace(JointLevelControl=Mock(return_value=r), __version__="1.0.9")
        with self.assertRaisesRegex(RuntimeError, "发送失败"):
            with SDKSource(config(), joint_control=True, sdk=sdk) as source:
                source.send(*[[0.0]*12]*5)
        self.assertEqual(r.setJointCommand.call_count,4)
        for call in r.setJointCommand.call_args_list[1:]:
            self.assertEqual(call.args, ([0.]*12,[0.]*12,[1.]*12,[0.]*12,[0.]*12))
        r.setZeroJointCommand.assert_not_called()
        r.disconnect.assert_called_once()

    def test_normal_exit_uses_last_measured_position_and_unit_damping(self):
        r = self.robot()
        r.getJointPosition.return_value = [1.2]*12
        sdk=SimpleNamespace(JointLevelControl=Mock(return_value=r), __version__="1.0.9")
        source=SDKSource(config(), joint_control=True, sdk=sdk)
        source.last_position=[1.2]*12
        with source:
            source.send(*[[0.0]*12]*5)
        self.assertEqual(r.setJointCommand.call_args.args,
                         ([0.]*12,[1.2]*12,[1.]*12,[0.]*12,[0.]*12))
        r.setZeroJointCommand.assert_not_called()

    def test_exit_damping_error_does_not_mask_original_error(self):
        r=self.robot()
        sdk=SimpleNamespace(JointLevelControl=Mock(return_value=r), __version__="1.0.9")
        with self.assertRaisesRegex(ValueError, "original failure"):
            with SDKSource(config(), joint_control=True, sdk=sdk) as source:
                source.send(*[[0.0]*12]*5)
                r.setJointCommand.side_effect=RuntimeError("damping failed")
                raise ValueError("original failure")
        r.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
