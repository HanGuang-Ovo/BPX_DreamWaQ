"""检查无权重工程的默认行为和策略入口错误信息。"""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
