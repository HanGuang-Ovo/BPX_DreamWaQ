"""不启动 Isaac Sim，验证 BPX 环境只裁剪单步总奖励的下限。"""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


class _ManagerBasedRLEnv:
    def __init__(self, cfg, render_mode=None, **kwargs):
        self.cfg = cfg

    def step(self, action):
        return self._step_output


def _load_environment():
    modules = {name: ModuleType(name) for name in ("isaaclab", "isaaclab.envs")}
    modules["isaaclab"].envs = modules["isaaclab.envs"]
    modules["isaaclab.envs"].ManagerBasedRLEnv = _ManagerBasedRLEnv
    path = Path(__file__).resolve().parents[2] / "src/bpx_dreamwaq/tasks/rough_env.py"
    spec = importlib.util.spec_from_file_location("_bpx_rough_env_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


rough_env = _load_environment()


class TestTotalRewardClipping(unittest.TestCase):
    def test_only_values_below_minus_ten_are_clipped(self):
        env = rough_env.BPXManagerBasedRLEnv(SimpleNamespace(minimum_step_reward=-10.0))
        reward = torch.tensor([-100.0, -10.0, -9.0, 0.0, 25.0])
        env._step_output = ("obs", reward, "terminated", "truncated", {})

        result = env.step(None)

        torch.testing.assert_close(result[1], torch.tensor([-10.0, -10.0, -9.0, 0.0, 25.0]))
        self.assertIs(result[1], reward)

    def test_nonfinite_lower_bound_is_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                rough_env.BPXManagerBasedRLEnv(SimpleNamespace(minimum_step_reward=value))

    def test_nonfinite_rewards_are_preserved_for_the_training_guard(self):
        env = rough_env.BPXManagerBasedRLEnv(SimpleNamespace(minimum_step_reward=-10.0))
        reward = torch.tensor([float("-inf"), float("nan"), float("inf")])
        env._step_output = ("obs", reward, "terminated", "truncated", {})

        result = env.step(None)[1]

        self.assertTrue(torch.isneginf(result[0]))
        self.assertTrue(torch.isnan(result[1]))
        self.assertTrue(torch.isposinf(result[2]))


if __name__ == "__main__":
    unittest.main()
