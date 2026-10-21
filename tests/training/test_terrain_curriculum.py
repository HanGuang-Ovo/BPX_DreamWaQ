"""地形课程：完整轨迹、指令切换、终止步、空间覆盖及局部重置。"""

import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, "src")
from bpx_dreamwaq.tasks.mdp.curriculums import (
    TerrainEpisodeMetrics, compute_terrain_level_moves, terrain_levels_vel,
)


class TestComputeTerrainLevelMoves(unittest.TestCase):
    def decide(self, **overrides):
        values = dict(
            command_distance=torch.tensor([10.0]), moving_time=torch.tensor([20.0]),
            episode_time=torch.tensor([20.0]), good_tracking_fraction=torch.tensor([0.95]),
            max_excursion=torch.tensor([3.0]), max_terrain_distance=torch.tensor([3.0]),
            completed=torch.tensor([True]), failed=torch.tensor([False]), terrain_length=8.0,
        )
        values.update(overrides)
        return tuple(x.tolist() for x in compute_terrain_level_moves(**values))

    def test_completed_and_covered_task_promotes(self):
        self.assertEqual(self.decide(), ([True], [False]))

    def test_coverage_without_completion_does_not_promote(self):
        self.assertEqual(self.decide(completed=torch.tensor([False])), ([False], [False]))

    def test_fall_overrides_timeout_and_coverage(self):
        self.assertEqual(self.decide(failed=torch.tensor([True])), ([False], [True]))

    def test_early_fall_during_clear_translation_demotes(self):
        self.assertEqual(self.decide(
            command_distance=torch.tensor([0.01]), moving_time=torch.tensor([0.02]),
            episode_time=torch.tensor([0.02]), failed=torch.tensor([True]),
            completed=torch.tensor([False]),
        ), ([False], [True]))

    def test_completed_covered_task_promotes_for_every_tracking_fraction(self):
        # 包括原来会降级的 <= 50% 达标比例，不能间接拦截升级。
        for fraction in (0.0, 0.3, 0.5, 0.7, 0.79, 0.8, 1.0):
            with self.subTest(fraction=fraction):
                self.assertEqual(self.decide(
                    good_tracking_fraction=torch.tensor([fraction]),
                ), ([True], [False]))

    def test_sustained_tracking_failure_still_demotes_without_coverage(self):
        self.assertEqual(self.decide(
            good_tracking_fraction=torch.tensor([0.3]), max_excursion=torch.tensor([0.5]),
        ), ([False], [True]))

    def test_isolated_tracking_errors_hold_without_coverage(self):
        self.assertEqual(self.decide(
            good_tracking_fraction=torch.tensor([0.7]), max_excursion=torch.tensor([0.5]),
        ), ([False], [False]))

    def test_good_motion_without_coverage_holds(self):
        self.assertEqual(self.decide(max_excursion=torch.tensor([0.5])), ([False], [False]))

    def test_motion_inside_terrain_center_platform_holds(self):
        self.assertEqual(self.decide(max_terrain_distance=torch.tensor([1.5])), ([False], [False]))

    def test_short_insufficient_task_holds(self):
        self.assertEqual(self.decide(
            command_distance=torch.tensor([0.5]), moving_time=torch.tensor([1.0]),
        ), ([False], [False]))

    def test_zero_translation_and_rotation_hold_even_after_fall(self):
        self.assertEqual(self.decide(
            command_distance=torch.tensor([0.0]), moving_time=torch.tensor([0.0]),
            failed=torch.tensor([True]),
        ), ([False], [False]))

    def test_brief_translation_in_mostly_rotation_episode_does_not_demote(self):
        self.assertEqual(self.decide(
            command_distance=torch.tensor([0.5]), moving_time=torch.tensor([1.0]),
            failed=torch.tensor([True]),
        ), ([False], [False]))

    def test_initial_reset_never_changes_level(self):
        self.assertEqual(self.decide(episode_time=torch.tensor([0.0])), ([False], [False]))

    def test_upgrade_only_disables_demotion_but_keeps_promotion_conditions(self):
        self.assertEqual(self.decide(failed=torch.tensor([True]), allow_demotion=False), ([False], [False]))
        self.assertEqual(self.decide(allow_demotion=False), ([True], [False]))

    def test_batch_rules_are_mutually_exclusive(self):
        up, down = self.decide(
            command_distance=torch.tensor([10., 10., 0., 0.5]),
            moving_time=torch.tensor([20., 20., 0., 1.]), episode_time=torch.tensor([20.] * 4),
            good_tracking_fraction=torch.tensor([1., 0.2, 1., 1.]),
            max_excursion=torch.tensor([3., 0.5, 3., 3.]), max_terrain_distance=torch.tensor([3.] * 4),
            completed=torch.tensor([True] * 4), failed=torch.tensor([False] * 4),
        )
        self.assertEqual(up, [True, False, False, False])
        self.assertEqual(down, [False, True, False, False])

    def test_invalid_parameters_rejected(self):
        for kwargs in (
            {"coverage_ratio": 0}, {"min_command_distance": 0},
            {"min_tracking_duration_s": 0}, {"min_moving_fraction": 0},
            {"failure_ratio": 0.0}, {"failure_ratio": 1.1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.decide(**kwargs)


class _FakeTerrain:
    def __init__(self, num_envs):
        self.device = "cpu"
        self.terrain_levels = torch.full((num_envs,), 2, dtype=torch.long)
        self.max_terrain_level = 10
        self.cfg = SimpleNamespace(terrain_generator=SimpleNamespace(size=(8.0, 8.0)))
        self.calls = []

    def update_env_origins(self, env_ids, move_up, move_down):
        self.calls.append((env_ids.clone(), move_up.clone(), move_down.clone()))
        self.terrain_levels[env_ids] += move_up.long() - move_down.long()
        self.terrain_levels.clamp_(min=0, max=9)


class _FakeScene(dict):
    def __init__(self, num_envs, terrain):
        super().__init__(robot=SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.zeros(num_envs, 3), root_lin_vel_b=torch.zeros(num_envs, 3),
            root_ang_vel_b=torch.zeros(num_envs, 3),
        )))
        self.terrain = terrain
        self.env_origins = torch.zeros(num_envs, 3)


class _FakeCommand:
    def __init__(self, env):
        self.env = env
        self.command = torch.zeros(env.num_envs, 3)
        self.terrain_metrics = TerrainEpisodeMetrics(env.num_envs, "cpu")
        self.reset()

    def reset(self, env_ids=None):
        self.terrain_metrics.reset(self.env.scene["robot"].data.root_pos_w[:, :2], env_ids)

    def update_terrain_metrics(self, env_ids=None):
        env = self.env
        data = env.scene["robot"].data
        self.terrain_metrics.update(
            data.root_pos_w[:, :2], env.scene.env_origins[:, :2], data.root_lin_vel_b[:, :2],
            data.root_ang_vel_b[:, 2], self.command, env.episode_length_buf, env.step_dt, env_ids,
        )


def make_fake_env(num_envs=1):
    terrain = _FakeTerrain(num_envs)
    env = SimpleNamespace(
        scene=_FakeScene(num_envs, terrain),
        reward_manager=SimpleNamespace(_episode_sums={"track": torch.full((num_envs,), 10.0)}),
        termination_manager=SimpleNamespace(
            time_outs=torch.ones(num_envs, dtype=torch.bool),
            terminated=torch.zeros(num_envs, dtype=torch.bool),
        ),
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long), num_envs=num_envs, step_dt=1.0,
    )
    term = _FakeCommand(env)
    env.command_manager = SimpleNamespace(get_term=lambda name: term)
    return env, term, terrain


def sample(env, term, x, vx, yaw=0.0, y=0.0, measured_vx=None, measured_yaw=None):
    """一个策略步：旧指令执行后的状态，随后才会发生指令重采样。"""
    data = env.scene["robot"].data
    env.episode_length_buf += 1
    data.root_pos_w[:, 0] = torch.as_tensor(x)
    data.root_pos_w[:, 1] = torch.as_tensor(y)
    term.command[:, 0] = torch.as_tensor(vx)
    term.command[:, 2] = torch.as_tensor(yaw)
    data.root_lin_vel_b[:, 0] = torch.as_tensor(vx if measured_vx is None else measured_vx)
    data.root_ang_vel_b[:, 2] = torch.as_tensor(yaw if measured_yaw is None else measured_yaw)
    term.update_terrain_metrics()


class TestEpisodeTrajectories(unittest.TestCase):
    def test_out_and_back_with_command_switch_promotes_despite_zero_net_displacement(self):
        env, term, terrain = make_fake_env()
        for i in range(1, 11):
            sample(env, term, i * 0.5, 0.5)
        for i in range(1, 11):
            sample(env, term, 5 - i * 0.5, -0.5)
        result = terrain_levels_vel(env, [0])
        self.assertEqual(terrain.terrain_levels.item(), 3)
        self.assertEqual(result["num_move_up"], 1)
        self.assertAlmostEqual(term.terrain_metrics.command_distance.item(), 10.0)

    def test_slow_well_tracked_walk_with_coverage_promotes(self):
        env, term, terrain = make_fake_env()
        for i in range(1, 21):
            sample(env, term, i * 0.15, 0.15)
        terrain_levels_vel(env, [0])
        self.assertEqual(terrain.terrain_levels.item(), 3)

    def test_tracking_tolerances_do_not_affect_promotion_on_same_trajectory(self):
        for lin_tolerance, yaw_tolerance in ((0.01, 0.01), (0.1, 0.5), (0.5, 1.0)):
            with self.subTest(lin_tolerance=lin_tolerance, yaw_tolerance=yaw_tolerance):
                env, term, terrain = make_fake_env()
                term.terrain_metrics.lin_vel_tolerance = lin_tolerance
                term.terrain_metrics.yaw_tolerance = yaw_tolerance
                for i in range(1, 21):
                    sample(env, term, i * 0.25, 0.5, yaw=0.5, measured_vx=0.25, measured_yaw=0.0)
                terrain_levels_vel(env, [0])
                self.assertEqual(terrain.terrain_levels.item(), 3)

    def test_below_previous_eighty_percent_gate_at_real_policy_dt_promotes(self):
        env, term, terrain = make_fake_env()
        env.step_dt = 0.02
        for i in range(1, 1001):
            sample(env, term, min(i, 500) * 0.01, 0.5, measured_vx=0.5 if i <= 500 else 0.0)
        terrain_levels_vel(env, [0])
        self.assertEqual(terrain.terrain_levels.item(), 3)

    def test_exact_half_episode_translation_at_real_policy_dt_can_demote(self):
        env, term, terrain = make_fake_env()
        env.step_dt = 0.02
        for i in range(1, 1001):
            sample(env, term, 0.0, 0.5 if i <= 500 else 0.0, measured_vx=0.0)
        terrain_levels_vel(env, [0])
        self.assertEqual(terrain.terrain_levels.item(), 1)

    def test_tight_circle_with_long_path_but_little_coverage_holds(self):
        env, term, terrain = make_fake_env()
        for i in range(1, 21):
            sample(env, term, 0.5 * torch.sin(torch.tensor(float(i))), 0.5,
                   yaw=1.0, y=0.5 * (1.0 - torch.cos(torch.tensor(float(i)))))
        terrain_levels_vel(env, [0])
        self.assertGreater(term.terrain_metrics.command_distance.item(), 4)
        self.assertEqual(terrain.terrain_levels.item(), 2)

    def test_platform_diagonal_does_not_count_as_coverage(self):
        env, term, terrain = make_fake_env()
        for _ in range(20):
            sample(env, term, 1.49, 0.5, y=1.49)
        terrain_levels_vel(env, [0])
        self.assertEqual(terrain.terrain_levels.item(), 2)

    def test_sustained_wrong_direction_without_coverage_demotes(self):
        env, term, terrain = make_fake_env()
        for i in range(1, 21):
            sample(env, term, -i * 0.02, 0.5, measured_vx=-0.02)
        terrain_levels_vel(env, [0])
        self.assertEqual(terrain.terrain_levels.item(), 1)

    def test_pure_rotation_and_standing_drift_hold(self):
        env, term, terrain = make_fake_env(2)
        for i in range(1, 21):
            sample(env, term, [float(i), float(i)], [0., 0.], yaw=[1., 0.], measured_vx=[1., 1.])
        result = terrain_levels_vel(env, slice(None))
        self.assertEqual(terrain.terrain_levels.tolist(), [2, 2])
        self.assertEqual(result["num_hold"], 2)

    def test_last_command_does_not_erase_prior_tracking_failure(self):
        env, term, terrain = make_fake_env()
        for _ in range(10):
            sample(env, term, 0.0, 1.0, measured_vx=0.0)
        for _ in range(10):
            sample(env, term, 0.0, 0.0)
        terrain_levels_vel(env, [0])
        self.assertEqual(terrain.terrain_levels.item(), 1)
        self.assertAlmostEqual(term.terrain_metrics.command_distance.item(), 10.0)

    def test_initial_reset_skips_moves_and_completed_episode_statistics(self):
        env, term, terrain = make_fake_env(2)
        env.scene["robot"].data.root_pos_w[:, :2] = 100
        result = terrain_levels_vel(env, slice(None))
        self.assertEqual(terrain.terrain_levels.tolist(), [2, 2])
        self.assertEqual(result["num_move_up"], 0)
        self.assertEqual(result["num_move_down"], 0)
        self.assertEqual(env.terrain_curriculum_stats["count"].sum().item(), 0)

    def test_terminal_step_is_sampled_before_reset_and_not_double_counted(self):
        env, term, terrain = make_fake_env()
        for i in range(1, 4):
            sample(env, term, i * 0.5, 0.5)
        env.episode_length_buf += 1
        env.scene["robot"].data.root_pos_w[:, 0] = 2.0
        result = terrain_levels_vel(env, [0])
        self.assertEqual(result["num_move_up"], 1)
        self.assertEqual(term.terrain_metrics.moving_time.item(), 4)
        term.update_terrain_metrics()
        self.assertEqual(term.terrain_metrics.moving_time.item(), 4)

    def test_partial_reset_preserves_other_env_and_does_not_count_teleport(self):
        env, term, terrain = make_fake_env(2)
        sample(env, term, [0.5, 0.5], [0.5, 0.5])
        sample(env, term, [1., 1.], [0.5, 0.5])
        env.scene["robot"].data.root_pos_w[0, 0] = 100
        env.scene.env_origins[0, 0] = 100
        term.reset(torch.tensor([0]))
        env.episode_length_buf[0] = 0
        term.update_terrain_metrics()
        torch.testing.assert_close(term.terrain_metrics.command_distance, torch.tensor([0., 1.]))
        sample(env, term, [100.5, 1.5], [0.5, 0.5])
        torch.testing.assert_close(term.terrain_metrics.max_excursion, torch.tensor([0.5, 1.5]))

    def test_world_translation_and_reset_offset_do_not_change_decision(self):
        env, term, terrain = make_fake_env(2)
        env.scene.env_origins[1, 0] = 100
        env.scene["robot"].data.root_pos_w[:, 0] = torch.tensor([0.5, 100.5])
        term.reset()
        for i in range(1, 21):
            sample(env, term, [0.5 + i * 0.15, 100.5 + i * 0.15], [0.15, 0.15])
        terrain_levels_vel(env, [0, 1])
        self.assertEqual(terrain.terrain_levels.tolist(), [3, 3])
        torch.testing.assert_close(term.terrain_metrics.max_excursion, torch.tensor([3., 3.]))

    def test_stats_record_previous_level_and_only_actual_demotions(self):
        env, term, terrain = make_fake_env(3)
        terrain.terrain_levels[:] = torch.tensor([2, 2, 0])
        for i in range(1, 21):
            sample(env, term, [i * 0.5, 0., 0.], [0.5, 0.5, 0.5], measured_vx=[0.5, 0., 0.])
        result = terrain_levels_vel(env, [0, 1, 2])
        self.assertEqual(terrain.terrain_levels.tolist(), [3, 1, 0])
        self.assertEqual(result["num_move_up"], 1)
        self.assertEqual(result["num_move_down"], 1)
        stats = env.terrain_curriculum_stats
        self.assertEqual(stats["reward_sum"][2].item(), 20)
        self.assertEqual(stats["count"][2].item(), 2)

    def test_nonfinite_position_cannot_create_coverage(self):
        env, term, terrain = make_fake_env()
        for _ in range(20):
            sample(env, term, float("inf"), 0.5)
        terrain_levels_vel(env, [0])
        self.assertEqual(terrain.terrain_levels.item(), 2)
        self.assertEqual(term.terrain_metrics.max_excursion.item(), 0)


if __name__ == "__main__":
    unittest.main()
