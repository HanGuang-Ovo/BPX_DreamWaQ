"""DreamWaQPPO 单元测试：存储、更新、优化器与 AdaBoot（测试 2/3 离线部分）。

不启动 Isaac Sim：直接向扩展 RolloutStorage 手动填充一个伪 rollout，
执行 PPO + CENet 联合更新并检查参数确实被更新、指标有限、无 NaN。
"""

import math
import statistics
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

sys.path.insert(0, "src")

from bpx_dreamwaq.tasks.agents.dreamwaq_actor_critic import DreamWaQActorCritic
from bpx_dreamwaq.tasks.agents.dreamwaq_ppo import (
    DreamWaQPPO, DreamWaQRolloutStorage, DreamWaQRunner, _compute_reward_cv,
)
from bpx_dreamwaq.tasks.obs_contract import (
    ACTION_DIM,
    DEFAULT_HISTORY_LENGTH,
    history_obs_dim,
    single_obs_dim,
)

NUM_ENVS = 4
NUM_STEPS = 6
NUM_CRITIC_OBS = 8


def make_policy():
    torch.manual_seed(0)
    return DreamWaQActorCritic(
        num_actor_obs=single_obs_dim(),
        num_critic_obs=NUM_CRITIC_OBS,
        num_actions=ACTION_DIM,
        history_length=DEFAULT_HISTORY_LENGTH,
        cenet_encoder_hidden_dims=[32, 16],
        cenet_vel_head_hidden_dims=[16],
        cenet_vae_head_hidden_dims=[16],
        cenet_decoder_hidden_dims=[16, 16],
        cenet_latent_dim=16,
        actor_hidden_dims=[64, 32],
        critic_hidden_dims=[64, 32],
        init_noise_std=1.0,
    )


def make_alg(policy, **overrides):
    params = dict(
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=1.0e-3,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.01,
        desired_kl=None,
        max_grad_norm=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        device="cpu",
        lambda_vel=1.0,
        lambda_pred=1.0,
        beta_kl=1.0e-3,
    )
    params.update(overrides)
    return DreamWaQPPO(policy=policy, **params)


def rollout(alg: DreamWaQPPO, terminate_last_step: bool = False):
    """执行一个伪 rollout：数据随机，但维度与真实流程一致。"""
    obs = torch.randn(NUM_ENVS, single_obs_dim())
    critic_obs = torch.randn(NUM_ENVS, NUM_CRITIC_OBS)
    history = torch.randn(NUM_ENVS, history_obs_dim(DEFAULT_HISTORY_LENGTH))
    # 与 DreamWaQRunner.learn 一致：采集与 return 计算都在 inference_mode 内。
    with torch.inference_mode():
        for step in range(NUM_STEPS):
            true_vel = critic_obs[:, 0:3]
            actions = alg.act(obs, history, critic_obs, true_vel)
            dones = torch.zeros(NUM_ENVS, dtype=torch.long)
            if terminate_last_step and step == NUM_STEPS - 1:
                dones[:] = 1
            next_obs = torch.randn(NUM_ENVS, single_obs_dim())
            rewards = torch.randn(NUM_ENVS)
            alg.process_env_step(rewards, dones, {"time_outs": torch.zeros(NUM_ENVS, dtype=torch.bool)}, next_obs)
            obs, critic_obs, history = next_obs, torch.randn(NUM_ENVS, NUM_CRITIC_OBS), torch.randn(
                NUM_ENVS, history_obs_dim(DEFAULT_HISTORY_LENGTH)
            )
        alg.compute_returns(critic_obs)


class TestDreamWaQPPO(unittest.TestCase):
    def setUp(self):
        self.policy = make_policy()
        self.alg = make_alg(self.policy)
        self.alg.init_storage("rl", NUM_ENVS, NUM_STEPS, [single_obs_dim()], [NUM_CRITIC_OBS], [ACTION_DIM])

    def test_storage_buffers(self):
        storage = self.alg.storage
        self.assertIsInstance(storage, DreamWaQRolloutStorage)
        self.assertEqual(storage.history_observations.shape, (NUM_STEPS, NUM_ENVS, history_obs_dim(10)))
        self.assertEqual(
            storage.cenet_features.shape, (NUM_STEPS, NUM_ENVS, 3 + self.policy.cenet_latent_dim)
        )
        self.assertEqual(storage.true_velocities.shape, (NUM_STEPS, NUM_ENVS, 3))
        self.assertEqual(storage.next_observations.shape, (NUM_STEPS, NUM_ENVS, single_obs_dim()))

    def test_cenet_params_in_optimizer(self):
        optimizer_cenet_params = {
            id(p) for group in self.alg.optimizer.param_groups for p in group["params"]
        }
        for param in self.policy.cenet.parameters():
            self.assertIn(id(param), optimizer_cenet_params, "CENet 参数未注册到优化器")

    def test_update_runs_and_changes_params(self):
        rollout(self.alg)
        before = {
            "actor": self.policy.actor[0].weight.detach().clone(),
            "cenet": self.policy.cenet.encoder[0].weight.detach().clone(),
            "critic": self.policy.critic[0].weight.detach().clone(),
        }
        loss_dict = self.alg.update()
        self.assertIn("value_function", loss_dict)
        self.assertIn("surrogate", loss_dict)
        for key, value in loss_dict.items():
            self.assertTrue(torch.isfinite(torch.tensor(value)).item(), f"{key} 非有限值")
        # 优化器确实更新了 Actor、CENet 与 Critic。
        self.assertFalse(torch.allclose(before["actor"], self.policy.actor[0].weight))
        self.assertFalse(torch.allclose(before["cenet"], self.policy.cenet.encoder[0].weight))
        self.assertFalse(torch.allclose(before["critic"], self.policy.critic[0].weight))

    def test_cenet_metrics_reported(self):
        rollout(self.alg)
        self.alg.update()
        metrics = self.alg.cenet_metrics
        for key in (
            "cenet/velocity_mse",
            "cenet/velocity_error_x",
            "cenet/velocity_error_y",
            "cenet/velocity_error_z",
            "cenet/prediction_loss",
            "cenet/kl_loss",
            "cenet/total_auxiliary_loss",
            "cenet/adaboot_p_boot",
            "cenet/adaboot_est_fraction",
        ):
            self.assertIn(key, metrics)
            self.assertTrue(torch.isfinite(torch.tensor(metrics[key])).item(), f"{key} 非有限值")

    def test_done_masked_rollout_updates_without_nan(self):
        # 末步全部终止：next_obs 为 reset 后新观测，必须被屏蔽且不产生 NaN。
        rollout(self.alg, terminate_last_step=True)
        loss_dict = self.alg.update()
        for value in loss_dict.values():
            self.assertTrue(torch.isfinite(torch.tensor(value)).item())

    def test_adaboot_error_mode_uses_true_vel_early(self):
        alg = make_alg(self.policy, adaboot_enabled=True, adaboot_mode="error")
        alg.init_storage("rl", NUM_ENVS, NUM_STEPS, [single_obs_dim()], [NUM_CRITIC_OBS], [ACTION_DIM])
        # 初期速度误差 EMA 为 None → p 取 p_min（完全使用真实速度引导）。
        alg._update_adaboot()
        self.assertEqual(alg.adaboot_p, 0.0)
        rollout(alg)
        alg.update()
        # 误差 EMA 建立后 p 仍被限制在 [0, 1]。
        self.assertGreaterEqual(alg.adaboot_p, 0.0)
        self.assertLessEqual(alg.adaboot_p, 1.0)
        self.assertLessEqual(alg.cenet_metrics["cenet/adaboot_est_fraction"], 1.0)

    def test_adaboot_off_always_estimated(self):
        self.alg._update_adaboot()
        self.assertEqual(self.alg.adaboot_p, 1.0)
        rollout(self.alg)
        self.alg.update()
        self.assertEqual(self.alg.cenet_metrics["cenet/adaboot_est_fraction"], 1.0)

    def test_invalid_reward_cv_clears_stale_adaboot_state(self):
        alg = make_alg(self.policy, adaboot_enabled=True, adaboot_mode="reward_cv")
        alg.set_reward_cv(.5)
        alg._update_adaboot()
        self.assertAlmostEqual(alg.adaboot_p, 1.0 - math.tanh(.5), places=6)
        for invalid in (None, float("inf"), float("-inf"), float("nan"), -1.):
            with self.subTest(invalid=invalid):
                alg.set_reward_cv(.5)
                alg.set_reward_cv(invalid)
                alg._update_adaboot()
                self.assertIsNone(alg._reward_cv)
                self.assertEqual(alg.adaboot_p, alg.adaboot_p_min)

    def test_rollover_features_consistency(self):
        # rollout 保存的 cenet_features 与 AdaBoot 混合结果一致（无 AdaBoot 时
        # 应等于 CENet 估计特征）。
        rollout(self.alg)
        storage = self.alg.storage  # update 前访问
        with torch.inference_mode():
            features = self.policy.encode_cenet(storage.history_observations[0])
        self.assertTrue(torch.allclose(storage.cenet_features[0], features, atol=1e-6))


class TestRewardStatistics(unittest.TestCase):
    def test_finite_rewards_match_population_cv(self):
        for values in ([1., 2., 4.], [-1., -2., -4.], [2., 2.], [-3., 1., 1.]):
            with self.subTest(values=values):
                expected = statistics.pstdev(values) / abs(statistics.mean(values))
                self.assertAlmostEqual(_compute_reward_cv(values), expected)

    def test_nonfinite_rewards_and_insufficient_samples_are_unavailable(self):
        for values in ([], [1.], [0., 0.], [-1., 1.], [1e-8, 2e-8],
                       [1., float("inf")], [float("-inf"), 2.], [float("nan"), 2.]):
            with self.subTest(values=values):
                self.assertIsNone(_compute_reward_cv(values))

    def test_extreme_finite_rewards_do_not_overflow(self):
        self.assertAlmostEqual(_compute_reward_cv([1e308, 1e308, -1e308]), math.sqrt(8.))


class TestRunnerFiniteGuard(unittest.TestCase):
    def make_runner(self):
        runner = DreamWaQRunner.__new__(DreamWaQRunner)
        manager = SimpleNamespace(active_terms=["track", "dof_acc_l2"],
                                  _step_reward=torch.zeros(2, 2))
        runner.env = SimpleNamespace(num_envs=2, unwrapped=SimpleNamespace(reward_manager=manager))
        return runner

    def test_finite_data_passes(self):
        runner = self.make_runner()
        runner._check_rollout_finite(700, rewards=torch.tensor([1., -1.]), obs=torch.zeros(2, 45))

    def test_bad_reward_reports_environment_and_term(self):
        runner = self.make_runner()
        runner.env.unwrapped.reward_manager._step_reward[1, 1] = float("-inf")
        with self.assertRaisesRegex(FloatingPointError, r"iteration 718.*rewards: env_ids=\[1\].*dof_acc_l2"):
            runner._check_rollout_finite(718, rewards=torch.tensor([1., float("-inf")]))

    def test_bad_observations_are_also_rejected(self):
        runner = self.make_runner()
        obs = torch.zeros(2, 45)
        obs[0, 3] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, r"obs: env_ids=\[0\]"):
            runner._check_rollout_finite(718, rewards=torch.ones(2), obs=obs)

    def test_runner_stops_before_invalid_rewards_enter_ppo(self):
        runner = self.make_runner()
        runner.log_dir = None
        runner.device = "cpu"
        runner.env.device = "cpu"
        runner.num_steps_per_env = 1
        runner.current_learning_iteration = 718
        runner.privileged_obs_type = "critic"
        runner.true_vel_slice = (0, 3)
        runner.train_mode = Mock()
        obs = torch.zeros(2, 45)
        extras = {"observations": {"prop_history": torch.zeros(2, 450), "critic": torch.zeros(2, 8)}}
        runner.env.get_observations = Mock(return_value=(obs, extras))
        runner.env.step = Mock(return_value=(obs, torch.tensor([1., float("-inf")]),
                                             torch.tensor([0, 1]), extras))
        runner.alg = Mock()
        runner.alg.act.return_value = torch.zeros(2, 12)
        with self.assertRaisesRegex(FloatingPointError, "iteration 718"):
            runner.learn(1)
        runner.alg.process_env_step.assert_not_called()
        runner.alg.compute_returns.assert_not_called()
        runner.alg.update.assert_not_called()


if __name__ == "__main__":
    unittest.main()
