"""坐标对照、预览历史和只读 SDK 生命周期的回归测试。"""

from dataclasses import replace
from pathlib import Path
import copy
import math
import queue
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bpx_dreamwaq.sim2real.config import load_config
from bpx_dreamwaq.sim2real.diagnostics import attitude, build_snapshot, demo_sample, PolicyPreview
from bpx_dreamwaq.sim2real.monitor import MonitorWindow, MonitorWorker, leg_points
from bpx_dreamwaq.sim2real.sdk_robot import SDKSource
from bpx_dreamwaq.sim2real.monitor_input import MonitorGamepad
from bpx_dreamwaq.sim2real.controller import ActorController


class MockGamepad:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self):
        return {"remote_command": [0, 0, 0], "remote_command_source": "模拟电脑手柄"}


class TestDiagnostics(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "configs/sim2real/runtime.toml")
        self.sample = demo_sample(self.cfg, 0.0, 0.0, 1)

    def policy(self):
        policy = Mock(return_value=np.full(12, 0.2))
        policy.session = None
        return policy

    def test_official_defaults_do_not_require_alignment_or_motion_confirmation(self):
        self.assertEqual([j.sdk_index for j in self.cfg.joints],[0,3,6,9,1,4,7,10,2,5,8,11])
        self.assertTrue(all(j.sign==1 and j.offset==0 for j in self.cfg.joints))
        self.assertTrue(all(j.minimum<j.maximum for j in self.cfg.joints))
        # 新工程不迁移权重；这里单独验证身份/标定记录不增加启动条件。
        with patch("bpx_dreamwaq.sim2real.config.Path.is_file", return_value=True):
            self.assertEqual(self.cfg.blockers(),[])

    def test_override_is_visible_against_unmodified_official_feedback(self):
        joints = tuple(replace(j, sign=-1, offset=0.2) if i == 4 else j for i, j in enumerate(self.cfg.joints))
        cfg = replace(self.cfg, joints=joints)
        snap = build_snapshot(cfg, self.sample)
        row = snap["rows"][4]
        self.assertAlmostEqual(row["q_model"], -row["q_official"] + 0.2)
        # 模拟反馈仍代表官方坐标，不会反向应用覆盖项来掩盖错误。
        np.testing.assert_allclose(demo_sample(cfg, 0.0, 0.0, 1).q, self.sample.q)

    def test_known_quaternion_and_sensor_extrinsic(self):
        sample = replace(self.sample, quat_xyzw=[math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
        att = attitude(self.cfg, sample)
        np.testing.assert_allclose(att["body_rpy_deg"], [90, 0, 0], atol=1e-6)
        np.testing.assert_allclose(att["gravity_body"], [0, -1, 0], atol=1e-6)
        cal = copy.deepcopy(self.cfg.calibration)
        cal["imu"].update(quaternion_frame="sensor_to_world", sensor_to_body=[1, 0, 0, 0, 0, -1, 0, 1, 0])
        corrected = attitude(replace(self.cfg, calibration=cal), sample)
        np.testing.assert_allclose(corrected["body_rotation"], np.eye(3), atol=1e-6)

    def test_missing_invalid_feedback_is_not_fabricated_as_zero(self):
        bad = replace(self.sample, q=None, torque=None, quat_xyzw=[0] * 4)
        snap = build_snapshot(self.cfg, bad, PolicyPreview(self.cfg, policy=self.policy()))
        self.assertIsNone(snap["rows"][0]["q_model"])
        self.assertIsNone(snap["rows"][0]["torque_sdk"])
        self.assertIsNone(snap["attitude"])
        self.assertFalse(snap["policies"])
        self.assertTrue(snap["warnings"])

    def test_single_policy_preserves_previous_action_across_modes(self):
        policy = self.policy()
        probe = PolicyPreview(self.cfg, policy=policy)
        snap = build_snapshot(self.cfg, self.sample, probe)
        self.assertEqual(set(snap["policies"]), {"policy"})
        self.assertAlmostEqual(snap["policies"]["policy"]["target_model"][4], 0.8)
        build_snapshot(self.cfg, replace(self.sample, host_time=0.02, joint_token=2, imu_token=2), probe)
        np.testing.assert_allclose(policy.call_args.args[0][-12:], 0.2)
        self.assertEqual(policy.call_count, 2)
        np.testing.assert_allclose(policy.call_args.args[0][60:90], 0.0)
        for tick in range(10):
            sample = replace(
                self.sample,
                host_time=0.04 + tick * self.cfg.period,
                telemetry={**self.sample.telemetry, "remote_command": [0.3, 0.1, 0.2]},
            )
            snap = build_snapshot(self.cfg, sample, probe)
            np.testing.assert_allclose(policy.call_args.args[0][-12:], 0.2)
        self.assertEqual(snap["control"]["mode"], "walk")
        self.assertEqual(set(snap["policies"]), {"policy"})
        self.assertAlmostEqual(snap["policies"]["policy"]["target_model"][4], 0.8)
        np.testing.assert_allclose(policy.call_args.args[0][-12:], 0.2)
        # 模型使用状态机限变化率后的输入，不是原始手柄阶跃。
        np.testing.assert_allclose(policy.call_args.args[0][87:90], snap["control"]["model_command"])

    def test_monitor_and_deployment_share_stand_walk_stopping_transitions(self):
        policy = self.policy()
        controller = ActorController(self.cfg, policy=policy)
        preview_policy = self.policy()
        probe = PolicyPreview(self.cfg, policy=preview_policy)
        modes = []
        commands = [[0, 0, 0]] * 5 + [[0.5, 0, 0]] * 40 + [[0, 0, 0]] * 110
        for tick, command in enumerate(commands):
            sample = replace(
                self.sample,
                host_time=tick * self.cfg.period,
                telemetry={**self.sample.telemetry, "remote_command": command},
            )
            deployed = controller.step(sample, command)
            controller.commit_target(deployed["target_model"])
            snap = build_snapshot(self.cfg, sample, probe)
            self.assertEqual(snap["control"]["mode"], deployed["mode"])
            np.testing.assert_allclose(snap["control"]["model_command"], deployed["command"])
            if tick:
                np.testing.assert_allclose(preview_policy.call_args.args[0][-12:], 0.2)
            modes.append(deployed["mode"])
        self.assertIn("stopping", modes)
        self.assertEqual(modes[-1], "stand")

    def test_deployment_and_preview_each_load_only_one_model(self):
        for factory, module in ((ActorController, "controller"), (PolicyPreview, "diagnostics")):
            with self.subTest(controller=factory.__name__):
                policy = self.policy()
                policy.session = Mock()
                policy.session.get_inputs.return_value = [Mock(shape=[1, 450], type="tensor(float)")]
                policy.session.get_outputs.return_value = [Mock(shape=[1, 12], type="tensor(float)")]
                with patch(f"bpx_dreamwaq.sim2real.{module}.OnnxPolicy", return_value=policy) as loader:
                    instance = factory(self.cfg)
                    for tick in range(20):
                        sample = replace(self.sample, host_time=tick * self.cfg.period,
                                         telemetry={**self.sample.telemetry, "remote_command": [.5, 0, 0]})
                        if factory is ActorController:
                            result = instance.step(sample, [.5, 0, 0])
                            instance.commit_target(result["target_model"])
                        else:
                            instance.step(sample)
                    loader.assert_called_once_with(self.cfg.model_path(), 450, 12)
                    self.assertEqual(policy.call_count, 21)  # 独立预热一次，再处理 20 帧反馈

    def test_no_input_unsupported_motion_and_stale_motion_never_fabricate_policy(self):
        for fields in (
            {"remote_command": None},
            {"remote_command": [float("nan"), 0, 0]},
            {"motion_state": None},
            {"motion_state": 0},
            {"motion_state": 1},
            {"motion_state": 2},
            {"motion_state": 3},
            {"gait": 12},
        ):
            sample = replace(self.sample, telemetry={**self.sample.telemetry, **fields})
            policy = self.policy()
            snap = build_snapshot(self.cfg, sample, PolicyPreview(self.cfg, policy=policy))
            self.assertFalse(snap["policies"], fields)
            self.assertIsNone(snap["control"]["policy"], fields)
            policy.assert_not_called()
            self.assertTrue(snap["warnings"], fields)

    def test_stale_feedback_and_missing_tokens_do_not_add_a_preview_gate(self):
        probe=PolicyPreview(self.cfg,policy=self.policy())
        for bad in (replace(self.sample,joint_age_s=100.),replace(self.sample,imu_token=None)):
            snap=build_snapshot(self.cfg,bad,probe)
            self.assertTrue(snap["policies"])
        disconnected=build_snapshot(self.cfg,replace(self.sample,connected=False),probe)
        self.assertFalse(disconnected["policies"])

    def test_tilt_drawing_and_preview_do_not_use_a_tilt_stop(self):
        sample=replace(self.sample,quat_xyzw=[math.sqrt(.5),0,0,math.sqrt(.5)])
        snap=build_snapshot(self.cfg,sample,PolicyPreview(self.cfg,policy=self.policy()))
        self.assertTrue(snap["policies"])
        self.assertIsNotNone(snap["attitude"])

    def test_actual_onnx_contract_and_finite_preview(self):
        if not self.cfg.model_path().exists():
            self.skipTest("仓库未附带本地模型")
        try:
            import onnxruntime
        except ImportError:
            self.skipTest("当前解释器未安装 ONNX Runtime")
        probe = PolicyPreview(self.cfg)
        self.assertFalse(probe.errors)
        snap = build_snapshot(self.cfg, self.sample, probe)
        self.assertEqual(set(snap["policies"]), {"policy"})
        for tick in range(10):
            moving = replace(
                self.sample,
                host_time=(tick + 1) * self.cfg.period,
                telemetry={**self.sample.telemetry, "remote_command": [0.3, 0, 0]},
            )
            snap = build_snapshot(self.cfg, moving, probe)
        self.assertEqual(set(snap["policies"]), {"policy"})
        for result in snap["policies"].values():
            self.assertEqual(len(result["action"]), 12)
            self.assertTrue(np.isfinite(result["target_sdk"]).all())

    def test_leg_geometry_matches_straight_model_reference(self):
        leg = leg_points(np.zeros(12))[0]
        np.testing.assert_allclose(leg[-1], [0.2386, 0.17122, -0.4615648])

    def test_gui_pump_draws_once_and_yields_even_when_producer_keeps_queue_full(self):
        snap = {"sample": self.sample, "received_at": time.monotonic(), "source": "DEMO"}

        class BusyQueue:
            maxsize = 8
            calls = 0

            def get_nowait(self):
                self.calls += 1
                if self.calls > self.maxsize:
                    raise AssertionError("GUI 不应无限排空持续更新的队列")
                return "snapshot", snap

        window = object.__new__(MonitorWindow)
        window.messages = BusyQueue()
        window.config, window.snapshot = self.cfg, None
        window.active, window.closing = True, False
        for name in ("status", "freshness", "root", "policy_status"):
            setattr(window, name, Mock())
        window.update_snapshot = Mock()
        window.refresh_checks = Mock()
        window.refresh_table = Mock()
        window.draw_robot = Mock()
        window.pump()
        self.assertEqual(window.messages.calls, 8)
        window.update_snapshot.assert_called_once()
        window.root.after.assert_called_once()

    def test_live_worker_only_constructs_request_robot_state(self):
        robot = Mock()
        robot.connect.return_value = True
        robot.isConnected.return_value = True
        robot.getRobotModel.return_value = 1
        robot.getRobotVersion.return_value = (1, 0, 15, 0, 0, 0)
        robot.getRobotSerialNumber.return_value = "test-SN"
        robot.getJointStateTimestamp.return_value = 1
        robot.getImuTimestamp.return_value = 1
        robot.getOdometryTimestamp.return_value = 1
        robot.getMotionStateTimestamp.return_value = 1
        for getter, value in [
            ("getJointPosition", self.sample.q),
            ("getJointVelocity", self.sample.dq),
            ("getJointTorque", self.sample.torque),
            ("getImuQuat", self.sample.quat_xyzw),
            ("getImuOmega", self.sample.omega),
            ("getImuAcc", self.sample.acceleration),
            ("getLegOdom", {"velocity_body": [0, 0, 0]}),
            ("getMotorTemperature", [25] * 12),
            ("getDriverTemperature", [25] * 12),
            ("getCurrentMotionState", 0),
            ("getCurrentGait", 0),
            ("getControlMode", 1),
            ("getBatteryLevel", 80),
        ]:
            getattr(robot, getter).return_value = value
        sdk = SimpleNamespace(
            RequestRobotState=Mock(return_value=robot),
            JointLevelControl=Mock(),
            MotionLevelControl=Mock(),
            __version__="1.0.9",
        )
        messages = queue.Queue(maxsize=8)
        worker = MonitorWorker(
            self.cfg,
            messages,
            no_policy=True,
            source_factory=lambda cfg, **kwargs: SDKSource(cfg, sdk=sdk, **kwargs),
            gamepad_factory=lambda *a, **k: MockGamepad(),
        )
        worker.start()
        try:
            deadline = time.monotonic() + 2
            seen = False
            while time.monotonic() < deadline:
                kind, value = messages.get(timeout=1)
                if kind == "error":
                    self.fail(value)
                if kind == "snapshot":
                    seen = True
                    break
            self.assertTrue(seen)
        finally:
            worker.stop_event.set()
            worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        sdk.JointLevelControl.assert_not_called()
        sdk.MotionLevelControl.assert_not_called()
        robot.setZeroPositionsFlag.assert_not_called()
        robot.setJointCommand.assert_not_called()
        robot.setZeroJointCommand.assert_not_called()
        robot.disconnect.assert_called_once()

    def test_gamepad_disconnect_is_missing_then_reconnects_and_closes_readonly_source(self):
        source = MagicMock()
        source.device.name = "test-gamepad"
        source.read.side_effect = [np.array([0.2, -0.1, 0.3]), OSError("手柄已拔出"), np.zeros(3)]
        clock = [0.0]
        reader = MonitorGamepad(
            self.cfg,
            clock=lambda: clock[0],
            selector=Mock(return_value=source.device),
            factory=Mock(return_value=source),
        )
        self.assertEqual(reader.read()["remote_command"], [0.2, -0.1, 0.3])
        lost = reader.read()
        self.assertIsNone(lost["remote_command"])
        self.assertIn("拔出", lost["remote_command_error"])
        source.close.assert_called_once()
        self.assertIsNone(reader.read()["remote_command"])
        clock[0] = 1.1
        self.assertEqual(reader.read()["remote_command"], [0.0, 0.0, 0.0])
        reader.close()
        self.assertEqual(source.close.call_count, 2)

    def test_demo_recording_contains_samples_and_never_constructs_sdk(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "monitor.jsonl"
            factory = Mock(side_effect=AssertionError("SDK 不应创建"))
            gamepad_factory = Mock(side_effect=AssertionError("DEMO 不应打开真实手柄"))
            worker = MonitorWorker(
                self.cfg,
                queue.Queue(maxsize=8),
                demo=True,
                policy=self.policy(),
                output=output,
                source_factory=factory,
                gamepad_factory=gamepad_factory,
            )
            worker.start()
            try:
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    kind, value = worker.messages.get(timeout=1)
                    if kind == "snapshot":
                        break
                    if kind == "error":
                        self.fail(value)
            finally:
                worker.stop_event.set()
                worker.join(timeout=2)
            factory.assert_not_called()
            gamepad_factory.assert_not_called()
            from bpx_dreamwaq.sim2real.recording import load_samples

            metadata, samples = load_samples(output)
            self.assertEqual(metadata["source"], "demo")
            self.assertTrue(samples)
            self.assertEqual(samples[0].telemetry["remote_command"], [0, 0, 0])

    def test_recording_replay_never_reads_current_gamepad(self):
        from bpx_dreamwaq.sim2real.recording import JsonlWriter

        with tempfile.TemporaryDirectory() as tmp:
            recording = Path(tmp) / "recording.jsonl"
            with JsonlWriter(recording, {}) as log:
                for tick in range(10):
                    sample = replace(
                        self.sample,
                        host_time=tick * self.cfg.period,
                        telemetry={**self.sample.telemetry, "remote_command": [0.5, 0, 0]},
                    )
                    log.write({"type": "sample", "sample": sample.to_dict()})
            forbidden = Mock(side_effect=AssertionError("回放不应访问设备"))
            messages = queue.Queue(maxsize=32)
            worker = MonitorWorker(
                self.cfg,
                messages,
                recording=recording,
                policy=self.policy(),
                source_factory=forbidden,
                gamepad_factory=forbidden,
            )
            worker.start()
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            forbidden.assert_not_called()
            snapshots = []
            while not messages.empty():
                kind, value = messages.get_nowait()
                self.assertNotEqual(kind, "error", value)
                if kind == "snapshot":
                    snapshots.append(value)
            self.assertEqual(snapshots[-1]["control"]["mode"], "walk")
            self.assertEqual(snapshots[-1]["control"]["remote_command"], [0.5, 0, 0])


if __name__ == "__main__":
    unittest.main()
