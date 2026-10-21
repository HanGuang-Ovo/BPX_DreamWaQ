"""离线验证新增奖励的地形参考、无效射线和异步回合重置行为。"""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


def _load_rewards():
    # 仅替换 Isaac Sim 的导入接口，执行实际奖励代码；不启动仿真器。
    modules = {name: ModuleType(name) for name in (
        "isaaclab", "isaaclab.envs", "isaaclab.envs.mdp", "isaaclab.managers", "isaaclab.sensors"
    )}
    modules["isaaclab"].envs = modules["isaaclab.envs"]
    modules["isaaclab.envs"].mdp = modules["isaaclab.envs.mdp"]
    for name in (
        "track_lin_vel_xy_exp", "track_ang_vel_z_exp", "lin_vel_z_l2", "ang_vel_xy_l2",
        "joint_torques_l2", "joint_acc_l2", "action_rate_l2", "undesired_contacts",
        "flat_orientation_l2", "joint_pos_limits",
    ):
        setattr(modules["isaaclab.envs.mdp"], name, None)

    class ManagerTermBase:
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env

    modules["isaaclab.managers"].ManagerTermBase = ManagerTermBase
    modules["isaaclab.managers"].RewardTermCfg = SimpleNamespace
    modules["isaaclab.managers"].SceneEntityCfg = SimpleNamespace
    modules["isaaclab.sensors"].ContactSensor = SimpleNamespace
    modules["isaaclab.sensors"].RayCaster = SimpleNamespace
    path = Path(__file__).resolve().parents[2] / "src/bpx_dreamwaq/tasks/mdp/rewards.py"
    spec = importlib.util.spec_from_file_location("_bpx_rewards_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


rewards = _load_rewards()


class _Scene(dict):
    @property
    def sensors(self):
        return self


def make_clearance_env(positions, velocities, hits):
    data = SimpleNamespace(
        body_link_pos_w=torch.tensor(positions, dtype=torch.float32),
        body_link_lin_vel_w=torch.tensor(velocities, dtype=torch.float32),
    )
    scanner = SimpleNamespace(data=SimpleNamespace(ray_hits_w=torch.tensor(hits, dtype=torch.float32)))
    return SimpleNamespace(scene=_Scene(robot=SimpleNamespace(data=data), height_scanner=scanner))


def clearance(env, **kwargs):
    return rewards.feet_clearance(
        env, target_height=0.1,
        asset_cfg=SimpleNamespace(name="robot", body_ids=slice(None)),
        sensor_cfg=SimpleNamespace(name="height_scanner"), **kwargs,
    )


class TestBaseHeight(unittest.TestCase):
    def penalty(self, root_positions, ray_hits):
        env = SimpleNamespace(scene=_Scene(
            robot=SimpleNamespace(data=SimpleNamespace(
                root_pos_w=torch.tensor(root_positions, dtype=torch.float32),
            )),
            height_scanner=SimpleNamespace(data=SimpleNamespace(
                ray_hits_w=torch.tensor(ray_hits, dtype=torch.float32),
            )),
        ))
        return rewards.base_height_l2(
            env, target_height=0.42,
            asset_cfg=SimpleNamespace(name="robot"),
            sensor_cfg=SimpleNamespace(name="height_scanner"),
        )

    def test_target_height_on_flat_and_raised_ground(self):
        result = self.penalty(
            [[0, 0, 0.42], [10, -5, 2.42]],
            [[[0, 0, 0]], [[10, -5, 2]]],
        )
        torch.testing.assert_close(result, torch.zeros(2), atol=1e-12, rtol=0)

    def test_crouching_and_excessive_height_are_both_penalized(self):
        result = self.penalty(
            [[0, 0, 0.32], [0, 0, 0.52]],
            [[[0, 0, 0]], [[0, 0, 0]]],
        )
        torch.testing.assert_close(result, torch.tensor([0.01, 0.01]))

    def test_sloped_ground_uses_mean_height(self):
        result = self.penalty(
            [[0, 0, 0.62]], [[[-0.1, 0, 0.1], [0.1, 0, 0.3]]],
        )
        torch.testing.assert_close(result, torch.zeros(1), atol=1e-12, rtol=0)

    def test_missing_rays_are_excluded_from_the_mean(self):
        result = self.penalty(
            [[0, 0, 0.72]],
            [[[0, 0, 0.2], [0.1, 0, float("inf")],
              [float("nan"), 0, 1.0], [0.2, 0, 0.4]]],
        )
        torch.testing.assert_close(result, torch.zeros(1), atol=1e-12, rtol=0)

    def test_all_missing_rays_only_skip_the_affected_environment(self):
        result = self.penalty(
            [[0, 0, 0.1], [0, 0, 0.32]],
            [[[0, 0, float("inf")], [0, 0, float("nan")]],
             [[0, 0, 0], [0.1, 0, 0]]],
        )
        torch.testing.assert_close(result, torch.tensor([0.0, 0.01]))


class TestFeetClearance(unittest.TestCase):
    def test_target_height_and_stationary_support_have_zero_cost(self):
        env = make_clearance_env(
            [[[0, 0, 0.1], [0.1, 0, 0.04]]],
            [[[2, 0, 0], [0, 0, 1]]], [[[0, 0, 0], [0.1, 0, 0]]],
        )
        torch.testing.assert_close(clearance(env), torch.zeros(1))

    def test_dragging_and_excessive_lift_are_both_penalized(self):
        env = make_clearance_env(
            [[[0, 0, 0]], [[0, 0, 0.2]]],
            [[[3, 4, 9]], [[3, 4, -9]]], [[[0, 0, 0]], [[0, 0, 0]]],
        )
        # 两者偏离目标 0.1 m，水平速度均为 5 m/s；竖直速度不参与。
        torch.testing.assert_close(clearance(env), torch.tensor([0.05, 0.05]))

    def test_local_terrain_and_world_translation(self):
        env = make_clearance_env(
            [[[0, 0, 0.1], [0.3, 0, 0.3]]],
            [[[1, 0, 0], [0, 1, 0]]], [[[0, 0, 0], [0.3, 0, 0.2]]],
        )
        torch.testing.assert_close(clearance(env), torch.zeros(1), atol=1e-7, rtol=0)
        # 整体平移到另一块地形后，奖励不变。
        offset = torch.tensor([10.0, -5.0, 2.0])
        env.scene["robot"].data.body_link_pos_w += offset
        env.scene["height_scanner"].data.ray_hits_w += offset
        torch.testing.assert_close(clearance(env), torch.zeros(1), atol=1e-7, rtol=0)

    def test_missing_rays_do_not_create_nan_or_mask_other_feet(self):
        env = make_clearance_env(
            [[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [1, 0, 0]]],
            [[[1, 0, 0], [1, 0, 0]], [[1, 0, 0], [1, 0, 0]]],
            [[[float("nan"), 0, 0], [0, 0, 0]],
             [[float("inf"), float("inf"), float("inf")], [float("nan"), 0, 0]]],
        )
        torch.testing.assert_close(clearance(env), torch.tensor([0.01, 0.0]))

    def test_uncovered_foot_is_skipped(self):
        env = make_clearance_env([[[1, 0, 0]]], [[[1, 0, 0]]], [[[0, 0, 0]]])
        torch.testing.assert_close(clearance(env), torch.zeros(1))

    def test_selected_toes_only(self):
        env = make_clearance_env(
            [[[0, 0, 10], [0, 0, 0]]], [[[10, 0, 0], [1, 0, 0]]], [[[0, 0, 0]]],
        )
        result = rewards.feet_clearance(
            env, target_height=0.1,
            asset_cfg=SimpleNamespace(name="robot", body_ids=[1]),
            sensor_cfg=SimpleNamespace(name="height_scanner"),
        )
        torch.testing.assert_close(result, torch.tensor([0.01]))


class TestActionSmoothness(unittest.TestCase):
    def setUp(self):
        self.env = SimpleNamespace(action_manager=SimpleNamespace(
            action=torch.zeros(2, 3), prev_action=torch.zeros(2, 3),
        ))
        self.term = rewards.action_smoothness_l2(SimpleNamespace(), self.env)

    def step(self, actions):
        manager = self.env.action_manager
        manager.prev_action.copy_(manager.action)
        manager.action.copy_(torch.tensor(actions, dtype=torch.float32))
        return self.term(self.env)

    def test_linear_ramp_has_no_curvature_after_initial_step(self):
        torch.testing.assert_close(self.step([[1]*3, [2]*3]), torch.tensor([3.0, 12.0]))
        torch.testing.assert_close(self.step([[2]*3, [4]*3]), torch.zeros(2))
        torch.testing.assert_close(self.step([[3]*3, [6]*3]), torch.zeros(2))

    def test_alternating_actions_are_penalized(self):
        self.step([[1]*3, [-1]*3])
        self.step([[-1]*3, [1]*3])
        torch.testing.assert_close(self.step([[1]*3, [-1]*3]), torch.tensor([48.0, 48.0]))

    def test_partial_reset_does_not_leak_history_or_reset_other_envs(self):
        self.step([[1]*3, [1]*3])
        self.step([[2]*3, [2]*3])
        # 模拟 ActionManager 先 reset，RewardManager 随后 reset 的实际顺序。
        self.env.action_manager.action[0] = 0
        self.env.action_manager.prev_action[0] = 0
        self.term.reset(torch.tensor([0]))
        torch.testing.assert_close(self.step([[1]*3, [3]*3]), torch.tensor([3.0, 0.0]))

    def test_full_reset_uses_zero_padding(self):
        self.step([[4]*3, [5]*3])
        self.step([[6]*3, [7]*3])
        self.env.action_manager.action.zero_()
        self.env.action_manager.prev_action.zero_()
        self.term.reset()
        torch.testing.assert_close(self.step([[1]*3, [1]*3]), torch.tensor([3.0, 3.0]))


class TestPersistentAirTime(unittest.TestCase):
    def penalty(self, times, **kwargs):
        env = SimpleNamespace(scene=_Scene(contact_forces=SimpleNamespace(data=SimpleNamespace(
            current_air_time=torch.tensor(times, dtype=torch.float32),
        ))))
        return rewards.feet_air_time_excess(
            env, sensor_cfg=SimpleNamespace(name="contact_forces", body_ids=slice(None)), **kwargs,
        )

    def test_normal_swing_is_allowed_and_stationary_raised_leg_has_cost(self):
        # No command/foot-velocity gate: an immobile raised foot still incurs cost.
        result = self.penalty([[0., .2, .4, .6], [0., 0., 5., 0.]])
        torch.testing.assert_close(result, torch.tensor([0., 1.]))

    def test_growth_is_bounded_and_each_foot_counts(self):
        result = self.penalty([[.7, .8, 1., 1.2], [100., 100., 100., 100.]])
        torch.testing.assert_close(result, torch.tensor([1.3, 4.]))

    def test_contact_and_sensor_reset_clear_only_affected_feet(self):
        result = self.penalty([[0., 0., 0., 0.], [0., 0., .9, 0.]])
        torch.testing.assert_close(result, torch.tensor([0., .3]))

    def test_invalid_time_settings_are_rejected(self):
        for kwargs in ({"maximum_air_time": 0.}, {"maximum_air_time": float("nan")},
                       {"penalty_cap": -1.}, {"penalty_cap": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.penalty([[0., 0., 0., 0.]], **kwargs)


class TestContactBalance(unittest.TestCase):
    def make_term(self, num_envs=2, dt=.02):
        sensor = SimpleNamespace(
            body_names=[f"{leg}_toe_link" for leg in ("fl", "fr", "hl", "hr")],
            data=SimpleNamespace(current_contact_time=torch.zeros(num_envs, 4),
                                 current_air_time=torch.zeros(num_envs, 4)),
        )
        env = SimpleNamespace(scene=_Scene(contact_forces=sensor), step_dt=dt)
        sensor_cfg = SimpleNamespace(name="contact_forces", body_ids=slice(None))
        term = rewards.feet_contact_balance(SimpleNamespace(params={"sensor_cfg": sensor_cfg}), env)
        return term, env, sensor, sensor_cfg

    def test_standing_and_alternating_trot_are_better_than_permanent_three_feet(self):
        term, env, sensor, cfg = self.make_term(3)
        costs = []
        for step in range(500):
            sensor.data.current_contact_time[0] = 1.
            # Equal contact occupancy, different diagonal phase every 0.2 seconds.
            sensor.data.current_contact_time[1] = torch.tensor(
                [1., 0., 0., 1.] if (step // 10) % 2 else [0., 1., 1., 0.],
            )
            sensor.data.current_contact_time[2] = torch.tensor([1., 1., 0., 1.])
            penalty = term(env, cfg)
            if step > 300:
                costs.append(penalty)
        mean_cost = torch.stack(costs).mean(0)
        self.assertEqual(mean_cost[0], 0.)
        self.assertLess(mean_cost[1], .02)
        self.assertGreater(mean_cost[2], .7)
        self.assertLess(term.contact_fraction[2, 2], .001)
        self.assertEqual(term.foot_names, ["fl", "fr", "hl", "hr"])

    def test_asynchronous_reset_preserves_other_environment(self):
        term, env, sensor, cfg = self.make_term()
        sensor.data.current_contact_time[:] = torch.tensor([1., 1., 0., 1.])
        sensor.data.current_air_time[:, 2] = 2.
        for _ in range(100):
            term(env, cfg)
        previous = term.contact_fraction[1].clone()
        term.reset(torch.tensor([0]))
        torch.testing.assert_close(term.contact_fraction[0], torch.full((4,), .5))
        torch.testing.assert_close(term.contact_fraction[1], previous)
        torch.testing.assert_close(term.air_time[:, 2], torch.tensor([0., 2.]))
        term.reset()
        torch.testing.assert_close(term.contact_fraction, torch.full((2, 4), .5))
        torch.testing.assert_close(term.air_time, torch.zeros(2, 4))

    def test_time_constant_does_not_depend_on_policy_frequency(self):
        results = []
        for dt, steps in ((.02, 100), (.01, 200)):
            term, env, sensor, cfg = self.make_term(1, dt)
            sensor.data.current_contact_time[:] = torch.tensor([1., 1., 0., 1.])
            for _ in range(steps):
                term(env, cfg)
            results.append(term.contact_fraction)
        torch.testing.assert_close(*results)


if __name__ == "__main__":
    unittest.main()
