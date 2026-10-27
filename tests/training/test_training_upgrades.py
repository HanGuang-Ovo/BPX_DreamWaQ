"""Regression tests for gradient routing, AdaBoot, normalization and power costs."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

from test_dreamwaq_actor_critic import make_policy
from test_dreamwaq_ppo import make_alg, rollout, NUM_ENVS, NUM_STEPS, NUM_CRITIC_OBS
from test_rewards import rewards, _Scene
from types import SimpleNamespace
from bpx_dreamwaq.tasks.agents.dreamwaq_actor_critic import DeployableDreamWaQPolicy
from bpx_dreamwaq.tasks.agents.dreamwaq_ppo import DreamWaQRunner
from bpx_dreamwaq.tasks.agents.normalization import ObservationNormalizer


class TestGradientRouting(unittest.TestCase):
    def test_ppo_updates_encoder_only_when_enabled_with_aux_losses_disabled(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                p = make_policy(num_critic_obs=NUM_CRITIC_OBS)
                alg = make_alg(p, ppo_grad_to_cenet=enabled, lambda_vel=0., lambda_pred=0., beta_kl=0.)
                alg.init_storage('rl', NUM_ENVS, NUM_STEPS, [45], [NUM_CRITIC_OBS], [12])
                rollout(alg)
                before = p.cenet.encoder[0].weight.detach().clone()
                alg.update()
                self.assertEqual(not torch.equal(before, p.cenet.encoder[0].weight), enabled)

    def test_adaboot_branch_survives_gradient_recomputation(self):
        p = make_policy(num_critic_obs=NUM_CRITIC_OBS)
        alg = make_alg(p, ppo_grad_to_cenet=True, adaboot_enabled=True)
        alg.init_storage('rl', NUM_ENVS, NUM_STEPS, [45], [NUM_CRITIC_OBS], [12])
        for probability in (0., 1.):
            alg.adaboot_p = probability
            rollout(alg)
            batch = next(alg.storage.mini_batch_generator(1, 1))
            obs, _, actions = batch[:3]
            history, saved, true_vel = batch[12:15]
            use_est = batch[-1]
            self.assertEqual(bool(use_est.all()), bool(probability))
            velocity, mu = p.cenet.deterministic_latent(p.normalize_history(history))
            rebuilt = torch.cat([torch.where(use_est, velocity, true_vel), mu], -1)
            torch.testing.assert_close(rebuilt, saved)
            p.set_cenet_features(rebuilt, detach=False)
            p.update_distribution(obs)
            torch.testing.assert_close(p.get_actions_log_prob(actions), batch[6].squeeze(-1))
            alg.storage.clear()

    def test_sample_latent_with_ppo_gradient_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, 'latent_actor_mode'):
            make_alg(make_policy(latent_actor_mode='sample'), ppo_grad_to_cenet=True)


class TestNormalization(unittest.TestCase):
    def test_running_moments_match_combined_batch(self):
        normalizer = ObservationNormalizer(3)
        a, b = torch.randn(19, 3) * 4 + 10, torch.randn(7, 3) - 5
        normalizer.update(a)
        normalizer.update(b)
        data = torch.cat([a, b])
        torch.testing.assert_close(normalizer.mean, data.mean(0))
        torch.testing.assert_close(normalizer.variance, data.var(0, unbiased=False))
        torch.testing.assert_close(normalizer(data).mean(0), torch.zeros(3), atol=1e-6, rtol=0)

    def test_history_and_current_share_stats_and_inference_is_frozen(self):
        p = make_policy(observation_normalization=True)
        history, critic = torch.randn(8, 450) * 5 + 20, torch.randn(8, 71)
        p.update_normalization(history, critic)
        torch.testing.assert_close(p.current_frame(p.normalize_history(history)), p.obs_normalizer(p.current_frame(history)))
        state = {k: v.clone() for k, v in p.obs_normalizer.state_dict().items()}
        deployed = DeployableDreamWaQPolicy(p).eval()
        torch.testing.assert_close(deployed(history), p.act_inference(p.current_frame(history), history))
        traced = torch.jit.trace(deployed, history[:1])
        torch.testing.assert_close(traced(history), deployed(history))
        for k, value in p.obs_normalizer.state_dict().items():
            torch.testing.assert_close(value, state[k])

    def test_checkpoint_export_preserves_normalization_and_legacy_identity(self):
        spec = importlib.util.spec_from_file_location('bpx_export', Path('scripts/export_policy.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for enabled in (False, True):
            cfg = dict(actor_hidden_dims=[32], critic_hidden_dims=[32], observation_normalization=enabled)
            from bpx_dreamwaq.tasks.agents.dreamwaq_actor_critic import DreamWaQActorCritic
            p = DreamWaQActorCritic(45, 71, 12, **cfg)
            history = torch.randn(4, 450) * 3 + 10
            p.update_normalization(history, torch.randn(4, 71))
            train_cfg = {'policy': cfg, 'empirical_normalization': enabled}
            with tempfile.TemporaryDirectory() as directory:
                path = str(Path(directory) / 'model.pt')
                torch.save({'model_state_dict': p.state_dict(), 'infos': {'train_cfg': train_cfg}}, path)
                exported, _ = module.build_policy_from_checkpoint(path)
                torch.testing.assert_close(exported(history), DeployableDreamWaQPolicy(p)(history))


class TestNoLogAdaBoot(unittest.TestCase):
    def test_reward_cv_updates_without_log_directory(self):
        history = torch.randn(2, 450)
        critic = torch.randn(2, 8)
        extras = {'observations': {'prop_history': history, 'critic': critic}}
        env = SimpleNamespace(num_envs=2, num_actions=12, device='cpu',
                              get_observations=lambda: (torch.zeros(2, 45), extras),
                              step=lambda actions: (torch.zeros(2, 45), torch.tensor([1., 3.]), torch.ones(2), extras))
        cfg = dict(policy={'class_name': 'DreamWaQActorCritic', 'actor_hidden_dims': [16], 'critic_hidden_dims': [16]},
                   algorithm={'class_name': 'DreamWaQPPO', 'adaboot_enabled': True, 'adaboot_mode': 'reward_cv',
                              'num_learning_epochs': 1, 'num_mini_batches': 1},
                   num_steps_per_env=2, save_interval=10, empirical_normalization=True)
        runner = DreamWaQRunner(env, cfg, log_dir=None)
        runner.learn(2)
        self.assertAlmostEqual(runner.alg._reward_cv, 0.5)
        self.assertAlmostEqual(runner.alg.adaboot_p, 1 - torch.tanh(torch.tensor(0.5)).item())
        self.assertGreater(runner.policy.obs_normalizer.count.item(), 0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / 'runner.pt')
            runner.save(checkpoint)
            restored = DreamWaQRunner(env, cfg, log_dir=None)
            restored.load(checkpoint)
            torch.testing.assert_close(
                DeployableDreamWaQPolicy(restored.policy)(history),
                DeployableDreamWaQPolicy(runner.policy)(history),
            )


class TestPowerCosts(unittest.TestCase):
    def test_signed_power_variance_and_absolute_power_do_not_cancel(self):
        data = SimpleNamespace(applied_torque=torch.tensor([[2., -2.], [3., 3.], [0., 8.]]),
                               joint_vel=torch.tensor([[2., 2.], [2., 2.], [1., 0.]]))
        env = SimpleNamespace(scene=_Scene(robot=SimpleNamespace(data=data)))
        cfg = SimpleNamespace(name='robot', joint_ids=slice(None))
        torch.testing.assert_close(rewards.joint_power(env, cfg), torch.tensor([8., 12., 0.]))
        torch.testing.assert_close(rewards.power_distribution(env, cfg), torch.tensor([256., 0., 0.]))
        cfg.joint_ids = [0]
        torch.testing.assert_close(rewards.power_distribution(env, cfg), torch.zeros(3))

    def test_power_distribution_caps_large_watt_fourth_cost(self):
        data = SimpleNamespace(applied_torque=torch.tensor([[100., -100.]]),
                               joint_vel=torch.tensor([[100., 100.]]))
        env = SimpleNamespace(scene=_Scene(robot=SimpleNamespace(data=data)))
        cfg = SimpleNamespace(name='robot', joint_ids=slice(None))
        torch.testing.assert_close(
            rewards.power_distribution(env, cfg, penalty_cap=1.0e6), torch.tensor([1.0e6]),
        )


if __name__ == '__main__':
    unittest.main()
