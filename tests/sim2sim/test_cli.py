"""检查无权重工程的默认行为和策略入口错误信息。"""

from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
