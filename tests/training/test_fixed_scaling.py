"""Fixed-scale contracts across training, history, AdaBoot and deployment."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from test_dreamwaq_actor_critic import make_policy
from test_dreamwaq_ppo import make_alg
from bpx_dreamwaq.tasks.agents.checkpoint import build_policy_from_checkpoint
from bpx_dreamwaq.tasks.agents.dreamwaq_actor_critic import DeployableDreamWaQPolicy
from bpx_dreamwaq.tasks.agents.dreamwaq_ppo import DreamWaQRunner


class TestFixedScaling(unittest.TestCase):
    def make_fixed(self):
        return make_policy(num_critic_obs=235, fixed_observation_scaling=True)

    def test_frame_history_and_critic_units(self):
        p = self.make_fixed()
        expected = torch.tensor([.25]*3 + [1.]*3 + [2., 2., .25]
                                + [1.]*12 + [.05]*12 + [1.]*12)
        torch.testing.assert_close(p.obs_normalizer(torch.ones(45)), expected)
        frames = torch.arange(10*45, dtype=torch.float32).reshape(1, 10, 45)
        history = torch.cat([x.flatten(1) for x in torch.split(frames, [3,3,3,12,12,12], -1)], -1)
        torch.testing.assert_close(p.history_frames(p.normalize_history(history)), frames * expected)
        torch.testing.assert_close(p.critic_normalizer(torch.ones(235)),
                                   torch.cat([torch.full((3,), 2.), expected, torch.full((187,), 5.)]))

    def test_deployment_and_traced_graph_match_training(self):
        p = self.make_fixed().eval()
        history = torch.randn(3, 450)
        deploy = DeployableDreamWaQPolicy(p).eval()
        expected = p.act_inference(p.current_frame(history), history_obs=history)
        torch.testing.assert_close(deploy(history), expected)
        traced = torch.jit.trace(deploy, history)
        torch.testing.assert_close(traced(history * 2), deploy(history * 2))

    def test_checkpoint_export_infers_scales_and_preserves_legacy(self):
        for fixed in (False, True):
            cfg = dict(actor_hidden_dims=[32], critic_hidden_dims=[32])
            from bpx_dreamwaq.tasks.agents.dreamwaq_actor_critic import DreamWaQActorCritic
            p = DreamWaQActorCritic(45, 235, 12, fixed_observation_scaling=fixed, **cfg)
            with tempfile.TemporaryDirectory() as directory:
                path = str(Path(directory) / 'model.pt')
                # Deliberately omit the flag: persistent buffers identify new checkpoints.
                torch.save({'model_state_dict': p.state_dict(), 'infos': {'train_cfg': {'policy': cfg}}}, path)
                deployed, _ = build_policy_from_checkpoint(path)
                history = torch.randn(2, 450)
                self.assertEqual(deployed.policy.fixed_observation_scaling, fixed)
                torch.testing.assert_close(deployed(history), DeployableDreamWaQPolicy(p)(history))

    def test_adaboot_and_auxiliary_targets_are_scaled(self):
        for gradients in (False, True):
            p = self.make_fixed()
            alg = make_alg(p, adaboot_enabled=True, ppo_grad_to_cenet=gradients)
            alg.init_storage('rl', 4, 2, [45], [235], [12])
            alg.adaboot_p = 0.
            for _ in range(2):
                obs, history, critic = torch.ones(4,45), torch.ones(4,450), torch.ones(4,235)
                with torch.no_grad():
                    alg.act(obs, history, critic, torch.ones(4,3))
                    torch.testing.assert_close(alg.transition.cenet_features[:, :3], torch.full((4,3), 2.))
                    alg.process_env_step(torch.ones(4), torch.zeros(4), {}, next_obs=obs)
            with torch.no_grad():
                alg.compute_returns(critic)
            original = p.cenet.compute_aux_losses
            def check_targets(history, next_obs, true_vel, mask):
                torch.testing.assert_close(true_vel, torch.full_like(true_vel, 2.))
                torch.testing.assert_close(next_obs, p.obs_normalizer(torch.ones_like(next_obs)))
                return original(history, next_obs, true_vel, mask)
            with patch.object(p.cenet, 'compute_aux_losses', side_effect=check_targets):
                alg.update()
            self.assertTrue(torch.isfinite(torch.tensor(alg.cenet_metrics['cenet/velocity_mae'])))

    def test_rejects_double_normalization_and_incompatible_resume(self):
        with self.assertRaisesRegex(ValueError, '不能同时'):
            make_policy(fixed_observation_scaling=True, observation_normalization=True)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'old.pt')
            torch.save({'model_state_dict': make_policy().state_dict()}, path)
            from types import SimpleNamespace
            with self.assertRaisesRegex(ValueError, '固定缩放模式'):
                DreamWaQRunner.load(SimpleNamespace(policy=self.make_fixed()), path)


if __name__ == '__main__':
    unittest.main()
