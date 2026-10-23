"""DreamWaQ 实机部署配置、模型预检、调度和实机日志回放；不连接硬件。"""

from contextlib import redirect_stdout
from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from test_runtime import config, sample, Clock, Source, Policy, ROOT
from bpx_dreamwaq.sim2real.cli import check, main
from bpx_dreamwaq.sim2real.config import default_config, load_config
from bpx_dreamwaq.sim2real.controller import ActorController
from bpx_dreamwaq.sim2real.inference import warmup_policy
from bpx_dreamwaq.sim2real.recording import load_samples, summarize
from bpx_dreamwaq.sim2real.runner import replay, run_live


class DeploymentTest(unittest.TestCase):
    def test_controller_and_monitor_prefer_local_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            template = Path(tmp) / "runtime.toml"
            template.touch()
            with patch("bpx_dreamwaq.sim2real.config.DEFAULT_CONFIG", template):
                self.assertEqual(default_config(), template)
                local = template.with_name("local.runtime.toml")
                local.touch()
                self.assertEqual(default_config(), local)

    def test_policy_override_is_resolved_from_calling_directory(self):
        cfg = load_config(ROOT / "configs/sim2real/runtime.toml", policy_path="policies/alternative.onnx")
        self.assertEqual(cfg.model_path(), Path("policies/alternative.onnx").resolve())
        self.assertEqual(cfg.runtime["policy"]["model"], str(cfg.model_path()))

    def test_cli_policy_override_reaches_preflight(self):
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["--policy", "missing-model.onnx", "check"]), 0)
        report = json.loads(output.getvalue())
        self.assertFalse(report["ready"])
        self.assertEqual(report["models"]["policy"]["path"], str(Path("missing-model.onnx").resolve()))
        self.assertTrue(report["replay_blockers"])

    def test_preflight_reports_corrupt_model_without_constructing_sdk(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "broken.onnx"
            model.write_bytes(b"invalid onnx")
            cfg = load_config(ROOT / "configs/sim2real/runtime.toml", policy_path=model)
            with patch("bpx_dreamwaq.sim2real.preflight.importlib.util.find_spec", return_value=object()), \
                 patch("bpx_dreamwaq.sim2real.preflight.OnnxPolicy", side_effect=RuntimeError("broken graph")), \
                 patch("bpx_dreamwaq.sim2real.sdk_robot.SDKSource.__init__", side_effect=AssertionError("SDK")):
                report = check(cfg)
            self.assertFalse(report["ready"])
            self.assertIn("broken graph", report["models"]["policy"]["error"])
            self.assertTrue(report["deploy_blockers"])

    def test_missing_sdk_blocks_deployment_but_allows_offline_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.onnx"
            model.touch()
            cfg = load_config(ROOT / "configs/sim2real/runtime.toml", policy_path=model)
            policy = Policy(0.)
            with patch("bpx_dreamwaq.sim2real.preflight.importlib.util.find_spec",
                       side_effect=lambda name: None if name == "bpx_sdk" else object()), \
                 patch("bpx_dreamwaq.sim2real.preflight.OnnxPolicy", return_value=policy):
                report = check(cfg)
            self.assertFalse(report["ready"])
            self.assertEqual(report["replay_blockers"], [])
            self.assertEqual(report["deploy_blockers"], ["当前 Python 未安装 bpx_sdk"])
            observation = policy.observations[0]
            np.testing.assert_allclose(observation[30:60].reshape(10, 3), [[0, 0, -1]] * 10)
            np.testing.assert_allclose(observation[90:], 0.)

    def test_warmup_rejects_incompatible_shapes_and_dtypes(self):
        for input_shape, output_shape, dtype in (
                ([1, 48], [1, 12], "tensor(float)"),
                ([1, 450], [1, 24], "tensor(float)"),
                ([1, 450], [1, 12], "tensor(double)")):
            with self.subTest(input=input_shape, output=output_shape, dtype=dtype):
                policy = Policy(0.)
                policy.session = SimpleNamespace(
                    get_inputs=lambda: [SimpleNamespace(shape=input_shape, type=dtype)],
                    get_outputs=lambda: [SimpleNamespace(shape=output_shape, type=dtype)],
                )
                with self.assertRaises(ValueError):
                    warmup_policy(policy, config())
                self.assertEqual(policy.observations, [])

    def test_warmup_does_not_change_control_history_or_first_target(self):
        cfg, policy = config(), Policy(.2)
        with patch("bpx_dreamwaq.sim2real.controller.OnnxPolicy", return_value=policy):
            controller = ActorController(cfg)
        self.assertEqual(len(policy.observations), 1)
        self.assertIsNone(controller.history.frames)
        first = controller.step(sample(cfg), [0., 0., 0.])
        np.testing.assert_allclose(first["target_model"], cfg.policy["default_joint_position"])
        np.testing.assert_allclose(policy.observations[-1][330:], 0.)

    def test_invalid_model_warmup_fails_before_sdk_or_log_creation(self):
        source = Mock(side_effect=AssertionError("不能创建 SDK"))
        with tempfile.TemporaryDirectory() as tmp, \
             patch("bpx_dreamwaq.sim2real.controller.OnnxPolicy", return_value=Policy(np.nan)):
            path = Path(tmp) / "live.jsonl"
            with self.assertRaisesRegex(RuntimeError, "部署模型加载或预热失败"):
                run_live(config(), path, [0, 0, 0], .1, source_factory=source)
            self.assertFalse(path.exists())
        source.assert_not_called()

    def test_non_policy_send_cannot_postpone_an_overdue_policy(self):
        cfg, clock = config(), Clock()
        source, policy = Source(cfg, clock), Policy(0.)
        send = source.send

        def delayed_send(*args):
            send(*args)
            clock.sleep(.001 if len(source.commands) % 2 else .015)

        source.send = delayed_send
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "live.jsonl"
            run_live(cfg, path, [0, 0, 0], .15, policy=policy,
                     source_factory=lambda *a, **k: source, clock=clock.now, sleep=clock.sleep)
            rows = [row for line in path.read_text().splitlines()
                    if (row := json.loads(line)).get("type") == "command"]
        self.assertTrue(rows[0]["policy_due"])
        self.assertFalse(rows[1]["policy_due"])
        self.assertTrue(rows[2]["policy_due"])
        self.assertAlmostEqual(rows[2]["policy_lateness_s"], .015)
        self.assertGreaterEqual(len(policy.observations), 7)

    def test_live_log_can_be_analyzed_and_replayed_with_recorded_gamepad(self):
        cfg, clock = config(), Clock()
        source = Source(cfg, clock)
        with tempfile.TemporaryDirectory() as tmp:
            path, output = Path(tmp) / "live.jsonl", Path(tmp) / "replay.jsonl"
            run_live(cfg, path, [0, 0, 0], .1, policy=Policy(0.),
                     source_factory=lambda *a, **k: source, clock=clock.now, sleep=clock.sleep,
                     command_source=lambda: [.3, 0, 0])
            metadata, samples = load_samples(path)
            self.assertEqual(metadata["mode"], "live")
            self.assertEqual(metadata["model"]["path"], str(cfg.model_path()))
            self.assertEqual(summarize(samples)["samples"], 10)
            self.assertEqual(samples[0].telemetry["remote_command"], [.3, 0, 0])
            with patch("bpx_dreamwaq.sim2real.sdk_robot.SDKSource.__init__", side_effect=AssertionError("SDK")):
                self.assertEqual(replay(cfg, path, output, [0, 0, 0], policy=Policy(0.)), 5)

    def test_send_rate_must_support_the_policy_clock(self):
        text = (ROOT / "configs/sim2real/runtime.toml").read_text()
        calibration = (ROOT / "configs/sim2real/joint_calibration.template.toml").read_text()
        for rate in (25, 75, 120):
            with self.subTest(rate=rate), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "runtime.toml"
                path.write_text(text.replace("send_rate_hz = 100", f"send_rate_hz = {rate}"))
                (Path(tmp) / "joint_calibration.template.toml").write_text(calibration)
                with self.assertRaisesRegex(ValueError, "正整数倍"):
                    load_config(path)

    def test_actual_dreamwaq_model_through_wait_init_stand_walk_and_stop(self):
        cfg, clock = config(), Clock()
        if not cfg.model_path().is_file() or importlib.util.find_spec("onnxruntime") is None:
            self.skipTest("需要本地部署模型及 ONNX Runtime")
        source = Source(cfg, clock)
        source.q = ActorController(cfg, policy=Policy(0.)).calibration.position_to_sdk(
            [0.] * 4 + [.98] * 4 + [-2.62] * 4).tolist()
        read = source.read
        source.read = lambda: replace(read(), telemetry={"motion_state": 0, "gait": 0},
                                      velocity_body=None, odom_position=None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dreamwaq-live.jsonl"
            with patch("bpx_dreamwaq.sim2real.sdk_robot.SDKSource.__init__", side_effect=AssertionError("SDK")):
                run_live(cfg, path, [0, 0, 0], 9., source_factory=lambda *a, **k: source,
                         clock=clock.now, sleep=clock.sleep,
                         init_request_source=lambda: .1 <= clock.now() < .12,
                         command_source=lambda: [.3, 0, 0] if 6. <= clock.now() < 6.5 else [0, 0, 0])
            metadata, samples = load_samples(path)
            rows = [row for line in path.read_text().splitlines()
                    if (row := json.loads(line)).get("type") == "command"]
        modes = [row["mode"] for i, row in enumerate(rows) if i == 0 or row["mode"] != rows[i - 1]["mode"]]
        self.assertEqual(modes, ["waiting_init", "init", "stand", "walk", "stopping", "stand"])
        self.assertEqual(len(metadata["model"]["sha256"]), 64)
        self.assertEqual(metadata["model"]["warmup"]["input_shape"], [1, 450])
        self.assertEqual(len(samples), 900)
        for row in rows:
            self.assertTrue(np.isfinite(row["target_sdk"]).all())
            if "observation" in row["policy"]:
                self.assertEqual(len(row["policy"]["observation"]), 450)
        self.assertTrue(source.exited)


if __name__ == "__main__":
    unittest.main()
