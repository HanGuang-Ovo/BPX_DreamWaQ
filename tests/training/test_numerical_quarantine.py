"""Corrupt samples must not change healthy GAE, gradients, or optimizer updates."""
import copy
from types import SimpleNamespace
import unittest
import torch

from test_dreamwaq_ppo import make_policy, make_alg, rollout
from bpx_dreamwaq.tasks.mdp.numerical_guard import QuarantinedRewardTerm, QuarantinedRewardManager


class TestQuarantine(unittest.TestCase):
    def alg(self):
        alg = make_alg(make_policy())
        alg.init_storage('rl', 4, 6, [45], [8], [12])
        return alg

    def test_bad_transition_discards_prefix_but_not_previous_episode(self):
        alg = self.alg()
        with torch.inference_mode():
            for t in range(6):
                alg.act(torch.randn(4, 45), torch.randn(4, 450), torch.randn(4, 8), torch.randn(4, 3))
                done = torch.tensor([t in (1, 4), False, False, False])
                bad = torch.tensor([t == 4, False, False, False])
                reward = torch.ones(4)
                if t == 4:
                    reward[0] = float('nan')
                alg.process_env_step(reward, done, {'numerical_invalid': bad, 'time_outs': bad}, torch.zeros(4, 45))
        self.assertEqual(alg.storage.valid_transitions[:, 0, 0].tolist(), [True, True, False, False, False, True])
        self.assertTrue(alg.storage.valid_transitions[:, 1:].all())
        self.assertEqual(alg.storage.rewards[4, 0, 0], 0)
        with torch.inference_mode():
            alg.compute_returns(torch.zeros(4, 8))
        self.assertTrue(torch.isfinite(alg.storage.advantages).all())

    def test_poisoned_excluded_samples_have_identical_updates(self):
        original = self.alg()
        rollout(original)
        original.storage.valid_transitions[:3, 0] = False
        original.storage.dones[2, 0] = True
        clean = copy.deepcopy(original)
        poisoned = copy.deepcopy(original)
        for name in ('observations', 'privileged_observations', 'actions', 'values', 'rewards',
                     'mu', 'sigma', 'history_observations', 'cenet_features', 'true_velocities',
                     'next_observations', 'actions_log_prob'):
            getattr(poisoned.storage, name)[:3, 0] = float('nan')
        last = torch.zeros(4, 1)
        for alg in (clean, poisoned):
            alg.storage.compute_returns(last, alg.gamma, alg.lam)
        torch.testing.assert_close(clean.storage.advantages, poisoned.storage.advantages)
        for alg in (clean, poisoned):
            torch.manual_seed(71)
            alg.update()
        for a, b in zip(clean.policy.parameters(), poisoned.policy.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(clean.optimizer.state)
        self.assertTrue(clean.cenet_optimizer.state)

    def test_all_invalid_rollout_does_not_step_optimizers(self):
        alg = self.alg()
        rollout(alg)
        alg.storage.valid_transitions[:] = False
        before = copy.deepcopy(alg.policy.state_dict())
        alg.update()
        self.assertFalse(alg.optimizer.state)
        self.assertFalse(alg.cenet_optimizer.state)
        for name, value in alg.policy.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_terrain_does_not_learn_from_quarantined_failure(self):
        from test_terrain_curriculum import make_fake_env, sample
        from bpx_dreamwaq.tasks.mdp.curriculums import terrain_levels_vel
        env, term, terrain = make_fake_env(2)
        for i in range(3):
            sample(env, term, x=[i, i], vx=[1., 1.])
        env._numerical_invalid = torch.tensor([True, False])
        before = terrain.terrain_levels[0].clone()
        terrain_levels_vel(env, torch.tensor([0, 1]))
        self.assertEqual(terrain.terrain_levels[0], before)
        self.assertEqual(env.terrain_curriculum_stats['count'].sum(), 1.)

    def test_command_grid_does_not_record_faulty_interval(self):
        from test_terrain_command_integration import make_env, commands
        from unittest.mock import Mock
        env = make_env()
        cfg = commands.GridAdaptiveVelocityCommandCfg(asset_name='robot',
                resampling_time_range=(10., 10.), rel_standing_envs=0.)
        term = commands.GridAdaptiveVelocityCommand(cfg, env)
        term.reset()
        term.interval_steps[:] = 10
        term.interval_err_vx[:] = torch.tensor([float('nan'), 1.])
        term.interval_err_yaw[:] = 0
        term.grid.record_interval = Mock()
        env._numerical_invalid = torch.tensor([True, False])
        term.reset(torch.tensor([0, 1]))
        values = term.grid.record_interval.call_args.kwargs['mean_err_vx']
        torch.testing.assert_close(values, torch.tensor([.1]))

    def test_reward_masks_nan_before_accumulation(self):
        env = SimpleNamespace(_numerical_invalid=torch.tensor([False, True]))
        term = QuarantinedRewardTerm(lambda env: torch.tensor([3., float('inf')]))
        torch.testing.assert_close(term(env), torch.tensor([3., 0.]))

    def test_reward_logs_exclude_invalid_episodes_and_clear_all(self):
        env = SimpleNamespace(_numerical_invalid=torch.tensor([False, True]), max_episode_length_s=20.)
        sums = {'track': torch.tensor([40., 999.])}
        def reset(ids):
            sums['track'][ids] = 0
        manager = QuarantinedRewardManager(SimpleNamespace(_env=env, _episode_sums=sums, reset=reset))
        info = manager.reset(torch.tensor([0, 1]))
        self.assertEqual(info['Episode_Reward/track'], 2.)
        self.assertEqual(sums['track'].tolist(), [0., 0.])
