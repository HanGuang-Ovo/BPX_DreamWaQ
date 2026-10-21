"""检查无权重工程的默认行为和策略入口错误信息。"""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from bpx_dreamwaq.runtime.policy import ZeroPolicy
from bpx_dreamwaq.sim2sim.cli import main, parse_args
from bpx_dreamwaq.sim2sim.config import load_config


class CliTest(unittest.TestCase):
    def test_default_selects_zero_backend_with_history_interface(self):
        args = parse_args([])
        config = load_config(args.config)
        self.assertEqual(args.backend, "zero")
        self.assertEqual(config.observation.dimension, 450)

    def test_explicit_missing_policy_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.onnx"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = main(["--backend", "onnx", "--policy", str(missing)])
        self.assertEqual(result, 2)
        self.assertIn("ONNX 文件不存在", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_policy_override_requires_inference_backend(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = main(["--policy", "unused.onnx"])
        self.assertEqual(result, 2)
        self.assertIn("--backend onnx 或 torchscript", stderr.getvalue())

    def test_single_policy_override_runs_with_either_inference_backend(self):
        for backend, policy_type, suffix in (("onnx", "OnnxPolicy", ".onnx"),
                                            ("torchscript", "TorchScriptPolicy", ".pt")):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / f"policy{suffix}"
                path.touch()
                with patch(f"bpx_dreamwaq.runtime.policy.{policy_type}", return_value=ZeroPolicy(12)) as loader, \
                     redirect_stdout(io.StringIO()):
                    result = main(["--backend", backend, "--policy", str(path),
                                   "--duration", "0.04", "--no-realtime"])
                self.assertEqual(result, 0)
                loader.assert_called_once()
                self.assertEqual(loader.call_args.args[:3], (path, 450, 12))

    def test_default_runs_pd_init_before_policy(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            log = Path(tmp) / "run.csv"
            result = main(["--duration", "0.04", "--no-realtime", "--log", str(log)])
            self.assertIn(",init,init_pd,", log.read_text())
        self.assertEqual(result, 0)

    def test_gamepad_wires_rb_and_custom_terrain_start(self):
        gamepad = Mock()
        gamepad.__enter__ = Mock(return_value=gamepad)
        gamepad.__exit__ = Mock()
        with patch("bpx_dreamwaq.sim2sim.cli.select_gamepad", return_value=Mock()), \
             patch("bpx_dreamwaq.sim2sim.cli.GamepadCommandSource", return_value=gamepad) as factory, \
             patch("bpx_dreamwaq.sim2sim.cli.Sim2SimRunner") as runner, redirect_stdout(io.StringIO()):
            runner.return_value.run.return_value.termination_reason = "duration"
            runner.return_value.run.return_value.simulated_seconds = 1.
            result = main(["--gamepad", "--terrain", "stairs", "--stair-level", "2",
                           "--step-heights", "0.04", "0.1", "--terrain-seed", "42"])
        self.assertEqual(result, 0)
        cfg = runner.call_args.args[0]
        self.assertEqual(cfg.terrain.kind, "stairs")
        self.assertEqual(cfg.terrain.step_heights, (.04, .1))
        self.assertEqual(cfg.terrain.seed, 42)
        self.assertEqual(cfg.initial_state.base_position, (0., -7.5, .13))
        self.assertEqual(factory.call_args.kwargs["init_button"], 311)
        self.assertEqual(runner.return_value.run.call_args.kwargs["command_source"], gamepad.read)
        self.assertEqual(runner.return_value.run.call_args.kwargs["init_request_source"], gamepad.consume_init_request)

    def test_invalid_terrain_and_workflow_arguments_have_clear_errors(self):
        for arguments in (["--terrain", "stairs", "--stair-level", "4"],
                          ["--stair-level", "1"], ["--terrain-seed", "-1"],
                          ["--step-heights", "0.1", "0.03"],
                          ["--direct-policy", "--gamepad"], ["--direct-policy", "--auto-init"]):
            stderr = io.StringIO()
            with self.subTest(arguments=arguments), redirect_stderr(stderr):
                self.assertEqual(main(arguments), 2)
                self.assertIn("ERROR:", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
