"""验证迁移后工作区独立、默认配置一致、入口不受调用目录影响。"""

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from bpx_dreamwaq.paths import CONFIGS_DIR, PROJECT_ROOT
from bpx_dreamwaq.sim2real.config import default_config, load_config as load_real_config
from bpx_dreamwaq.sim2sim.config import load_config as load_sim_config
from bpx_dreamwaq.validation import check_assets


class WorkspaceTest(unittest.TestCase):
    def run_script(self, script, *args):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as cwd:
            return subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / script), *args],
                cwd=cwd, env=env, capture_output=True, text=True, timeout=30,
            )

    def test_model_references_and_joint_contract_are_complete(self):
        assets = check_assets()
        self.assertEqual(assets["missing_resources"], [])
        self.assertTrue(all(assets["joint_contract"].values()))

    def test_simulation_and_hardware_default_history_contract_matches(self):
        sim = load_sim_config(CONFIGS_DIR / "sim2sim/bpx_mixed.toml")
        real = load_real_config(CONFIGS_DIR / "sim2real/runtime.toml")
        self.assertEqual(sim.observation.dimension, real.policy["observation_dimension"])
        self.assertEqual(sim.observation.history_length, real.policy["history_length"])
        self.assertEqual(sim.observation.include_base_lin_vel, real.policy["include_base_lin_vel"])
        self.assertEqual(sim.joint_names, tuple(real.policy["joint_names"]))
        self.assertEqual(sim.initial_state.default_joint_position, tuple(real.policy["default_joint_position"]))
        self.assertEqual(sim.control.action_scale, real.policy["action_scale"])
        self.assertEqual(sim.simulation.policy_dt, real.period)
        self.assertEqual(sim.paths.policy, real.model_path())

    def test_source_has_no_imports_from_the_reference_project(self):
        legacy = {"BPX_test", "bpx_runtime", "bpx_sim2sim", "bpx_sim2real"}
        for path in (PROJECT_ROOT / "src").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                self.assertFalse(legacy.intersection(m.split(".")[0] for m in modules), str(path))

    def test_check_entry_runs_outside_project_without_installed_package(self):
        result = self.run_script("check_project.py")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        report = json.loads(result.stdout)
        self.assertEqual(report["project_root"], str(PROJECT_ROOT))
        self.assertEqual(report["errors"], [])

    def test_setup_entry_runs_outside_project(self):
        result = self.run_script("setup_env.py")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("[result] 检查通过", result.stdout)
        self.assertIn(str(PROJECT_ROOT), result.stdout)

    def test_hardware_preflight_runs_outside_project_without_connecting(self):
        result = self.run_script("sim2real.py", "check")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(Path(report["config"]), default_config())
        self.assertEqual(set(report["models"]), {"policy"})
        for model in report["models"].values():
            self.assertTrue(Path(model["path"]).is_relative_to(PROJECT_ROOT / "policies"))

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "需要 MuJoCo")
    def test_zero_policy_simulation_runs_outside_project(self):
        result = self.run_script("sim2sim.py", "--duration", "0.04", "--no-realtime")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("backend=zero", result.stdout)
        self.assertIn("policy_steps=2", result.stdout)

    def test_hardware_cli_import_does_not_load_sdk_or_simulators(self):
        code = """
import sys
from bpx_dreamwaq.sim2real.cli import main
assert not {'bpx_sdk', 'mujoco', 'isaaclab', 'onnxruntime', 'torch'} & sys.modules.keys()
"""
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_visualize_entry_runs_outside_project_without_connecting(self):
        result = self.run_script("visualize.py", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--demo", result.stdout)
        self.assertIn("--no-policy", result.stdout)
        self.assertIn("--recording", result.stdout)


if __name__ == "__main__":
    unittest.main()
