"""只读启动检查：软件检查、反馈时效和来源区分，不创建运动对象。"""

from dataclasses import replace
import queue
import unittest
from unittest.mock import Mock, patch

from test_runtime import config
from bpx_dreamwaq.sim2real.diagnostics import build_snapshot, demo_sample
from bpx_dreamwaq.sim2real.monitor_checks import PreflightWorker, startup_rows
from bpx_dreamwaq.sim2real.monitor_window import MonitorWindow
from bpx_dreamwaq.sim2real.preflight import check


class MonitorCheckTest(unittest.TestCase):
    def setUp(self):
        self.config = config()
        self.sample = demo_sample(self.config, 0., 0., 1)
        self.snapshot = build_snapshot(self.config, self.sample)
        self.snapshot.update(source="DEMO · 模拟反馈", received_at=0.)

    def rows(self, **kwargs):
        return {key: (status, detail) for key, _, status, detail in
                startup_rows(self.config, self.snapshot, **kwargs)}

    def test_demo_is_explicit_and_never_claims_hardware_ready(self):
        rows = self.rows(active=True)
        self.assertIn("DEMO", rows["source"][0])
        self.assertEqual(rows["joint"][0], "有效")
        self.assertEqual(rows["imu"][0], "有效")
        self.assertEqual(rows["gamepad"][0], "有效")
        self.assertNotIn("实机就绪", str(rows))

    def test_ended_or_stale_session_cannot_show_current_feedback_or_preview(self):
        self.snapshot["policies"] = {"policy": {}}
        for options in ({"active": False}, {"active": True, "elapsed": .2}):
            with self.subTest(options=options):
                rows = self.rows(**options)
                for key in ("joint", "imu", "gamepad", "preview"):
                    self.assertEqual(rows[key][0], "待检查")

    def test_aged_feedback_groups_are_flagged_independently(self):
        self.snapshot["sample"] = replace(self.sample, joint_age_s=.5)
        rows = self.rows(active=True)
        self.assertEqual(rows["joint"][0], "待检查")
        self.assertEqual(rows["imu"][0], "有效")

    def test_missing_or_invalid_values_are_not_marked_valid(self):
        bad = replace(self.sample, q=None, quat_xyzw=[0, 0, 0, 0],
                      telemetry={**self.sample.telemetry, "remote_command": None})
        self.snapshot = build_snapshot(self.config, bad)
        self.snapshot["source"] = "SDK · 只读反馈"
        rows = self.rows(active=True)
        for key in ("joint", "imu", "gamepad"):
            self.assertEqual(rows[key][0], "待检查")

    def test_zero_gamepad_is_valid_input(self):
        self.assertEqual(self.snapshot["control"]["remote_command"], [0., 0., 0.])
        self.assertEqual(self.rows(active=True)["gamepad"][0], "有效")

    def test_software_model_error_and_disabled_policy_are_visible(self):
        self.assertEqual(self.rows(report={"error": "broken model"})["model"], ("异常", "broken model"))
        rows = self.rows(no_policy=True)
        self.assertEqual(rows["model"][0], "已关闭")
        self.assertEqual(rows["preview"][0], "已关闭")

    def test_offline_worker_never_constructs_sdk_or_loads_disabled_policy(self):
        messages = queue.Queue(maxsize=1)
        with patch("bpx_dreamwaq.sim2real.sdk_robot.SDKSource.__init__", side_effect=AssertionError("SDK")), \
             patch("bpx_dreamwaq.sim2real.preflight.OnnxPolicy", side_effect=AssertionError("不应加载模型")):
            worker = PreflightWorker(self.config, messages, no_policy=True)
            worker.start()
            worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        report = messages.get_nowait()
        self.assertNotIn("error", report)
        self.assertFalse(report["model_checked"])
        self.assertFalse(report["ready"])

    def test_offline_worker_returns_errors_to_ui_instead_of_crashing(self):
        messages = queue.Queue(maxsize=1)
        with patch("bpx_dreamwaq.sim2real.monitor_checks.check", side_effect=OSError("missing config")):
            worker = PreflightWorker(self.config, messages)
            worker.start()
            worker.join(timeout=3)
        self.assertIn("missing config", messages.get_nowait()["error"])

    def test_close_waits_for_offline_check_without_blocking_tk(self):
        window = object.__new__(MonitorWindow)
        window.worker, window.preflight_worker, window.root = None, Mock(), Mock()
        window.preflight_worker.is_alive.return_value = True
        window._finish_close()
        window.root.after.assert_called_once_with(50, window._finish_close)
        window.root.destroy.assert_not_called()
        window.preflight_worker.is_alive.return_value = False
        window._finish_close()
        window.root.destroy.assert_called_once()

    def test_actual_model_preflight_can_be_rendered_without_feedback(self):
        import importlib.util

        if not self.config.model_path().is_file() or importlib.util.find_spec("onnxruntime") is None:
            self.skipTest("需要本地 ONNX 模型和推理依赖")
        with patch("bpx_dreamwaq.sim2real.sdk_robot.SDKSource.__init__", side_effect=AssertionError("SDK")):
            report = check(self.config)
        self.assertTrue(report["model_checked"])
        rows = self.rows(report=report)
        self.assertEqual(rows["model"][0], "通过")
        self.assertIn("450", rows["model"][1])
        self.assertEqual(rows["joint"][0], "待检查")


if __name__ == "__main__":
    unittest.main()
