"""Acceptance cannot hide failed environments or reuse AdaBoot rollout features."""

import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from bpx_dreamwaq.tasks.agents.dreamwaq_actor_critic import DeployableDreamWaQPolicy
from bpx_dreamwaq.tasks.agents.dreamwaq_ppo import DreamWaQRunner
from bpx_dreamwaq.tasks.evaluation import AcceptanceMetrics


def make_runner():
    history = torch.randn(2, 450)
    extras = {"observations": {"prop_history": history, "critic": torch.zeros(2, 8)}}
    env = SimpleNamespace(num_envs=2, num_actions=12, get_observations=lambda: (torch.zeros(2, 45), extras))
    cfg = dict(
        policy={"class_name": "DreamWaQActorCritic", "actor_hidden_dims": [16], "critic_hidden_dims": [16],
                "latent_actor_mode": "sample"},
        algorithm={"class_name": "DreamWaQPPO", "adaboot_enabled": True, "learning_rate": 1e-4},
        num_steps_per_env=2, save_interval=10, empirical_normalization=True,
    )
    return DreamWaQRunner(env, cfg, log_dir=None), history


class TestDeploymentInference(unittest.TestCase):
    def test_runner_inference_ignores_poisoned_features_and_sample_mode(self):
        runner, history = make_runner()
        runner.policy.update_normalization(history, torch.randn(2, 8))
        runner.policy.set_cenet_features(torch.full((2, 19), float("nan")))
        before = {k: v.clone() for k, v in runner.policy.obs_normalizer.state_dict().items()}
        inference = runner.get_inference_policy("cpu")
        with torch.inference_mode():
            a = inference(history)
            torch.testing.assert_close(a, inference(history), rtol=0, atol=0)
            torch.testing.assert_close(a, DeployableDreamWaQPolicy(runner.policy)(history))
        self.assertTrue(torch.isfinite(a).all())
        for key, value in before.items():
            torch.testing.assert_close(value, runner.policy.obs_normalizer.state_dict()[key])

    def test_finetuning_keeps_fresh_optimizer_and_loads_normalization(self):
        source, history = make_runner()
        source.policy.update_normalization(history, torch.randn(2, 8))
        source.alg.optimizer.zero_grad()
        sum(p.square().sum() for p in source.policy.parameters()).backward()
        source.alg.optimizer.step()
        source.current_learning_iteration = 1200
        source.alg.optimizer.param_groups[0]["lr"] = .01
        target, _ = make_runner()
        target.alg.adaboot_enabled = False
        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
            source.save(checkpoint.name)
            target.load_for_finetuning(checkpoint.name)
        self.assertEqual(target.current_learning_iteration, 0)
        self.assertFalse(target.alg.optimizer.state)
        self.assertEqual(target.alg.optimizer.param_groups[0]["lr"], 1e-4)
        self.assertFalse(target.alg.adaboot_enabled)
        for key, value in source.policy.state_dict().items():
            torch.testing.assert_close(value, target.policy.state_dict()[key])


class TestAcceptanceMetrics(unittest.TestCase):
    def test_cli_help_does_not_require_checkpoint_or_start_simulator(self):
        result = subprocess.run([sys.executable, "scripts/evaluate_policy.py", "--help"],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--checkpoint", result.stdout)
        self.assertNotIn("Starting kit application", result.stdout)

    def sample(self, metrics, velocity=None, terminated=(False, False), truncated=(False, False), score=True):
        velocity = np.zeros((2, 3)) if velocity is None else velocity
        metrics.update(velocity, np.zeros(2), np.zeros((2, 3)), np.zeros(2),
                       terminated, truncated, score=score)

    def test_zero_drift_passes_but_warmup_termination_fails(self):
        metrics = AcceptanceMetrics(2, (0, 0, 0), .05, .1)
        self.sample(metrics, terminated=(True, False), score=False)
        self.sample(metrics)
        result = metrics.result(completed=True)
        self.assertFalse(result["passed"])
        self.assertEqual(result["terminated_count"], 1)
        self.assertEqual(result["failed_env_count"], 1)

    def test_one_bad_env_is_not_hidden_by_average_or_signed_velocity(self):
        metrics = AcceptanceMetrics(2, (0, 0, 0), .05, .1)
        for speed in [.08, -.08]:
            self.sample(metrics, np.array([[0, 0, 0], [speed, 0, 0]]))
        result = metrics.result(completed=True)
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["max_linear_rmse"], .08)

    def test_tracking_uses_command_and_early_exit_cannot_pass(self):
        metrics = AcceptanceMetrics(2, (.3, 0, 0), .15, .2)
        self.sample(metrics, np.array([[.3, 0, 0], [.3, 0, 0]]))
        self.assertTrue(metrics.result(completed=True)["passed"])
        self.assertFalse(metrics.result(completed=False)["passed"])

    def test_truncation_and_nonfinite_values_fail(self):
        metrics = AcceptanceMetrics(2, (0, 0, 0), .05, .1)
        self.sample(metrics, truncated=(False, True))
        self.assertFalse(metrics.result(completed=True)["passed"])
        with self.assertRaises(FloatingPointError):
            self.sample(metrics, np.full((2, 3), float("nan")))


if __name__ == "__main__":
    unittest.main()
