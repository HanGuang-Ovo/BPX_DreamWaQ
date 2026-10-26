"""离线验证奖励的地形参考、足距、无效射线和异步回合重置行为。"""

import importlib.util
import math
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

    class SceneEntityCfg(SimpleNamespace):
        # 奖励函数把 SceneEntityCfg("robot") 作为默认参数，运行时会实例化。
        def __init__(self, name=None, **kwargs):
            super().__init__(name=name, **kwargs)

    modules["isaaclab.managers"].SceneEntityCfg = SceneEntityCfg
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


class TestJointAccelerationPenalty(unittest.TestCase):
    def penalty(self, acceleration, joint_ids=slice(None), **kwargs):
        env = SimpleNamespace(scene=_Scene(robot=SimpleNamespace(
            data=SimpleNamespace(joint_acc=acceleration),
        )))
        return rewards.joint_acc_l2(
            env, asset_cfg=SimpleNamespace(name="robot", joint_ids=joint_ids), **kwargs,
        )

    def test_normal_accelerations_match_original_and_preserve_input(self):
        acceleration = torch.tensor([[0., 100., -200.], [350., -500., 20.]])
        original = acceleration.clone()
        torch.testing.assert_close(self.penalty(acceleration), acceleration.square().sum(dim=1))
        torch.testing.assert_close(acceleration, original)

    def test_extreme_finite_accelerations_cannot_overflow(self):
        acceleration = torch.full((2, 12), torch.finfo(torch.float32).max)
        acceleration[1] *= -1
        result = self.penalty(acceleration)
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(result, torch.full((2,), 1.0e8))
        # 实际配置乘权重和策略 dt 后的贡献也必须有限、有下限。
        torch.testing.assert_close(result * -2.5e-7 * .02, torch.full((2,), -.5))

    def test_nonfinite_acceleration_receives_maximum_penalty_per_environment(self):
        acceleration = torch.tensor([
            [float("nan"), 0.], [float("inf"), 0.],
            [float("-inf"), 0.], [3., 4.],
        ])
        result = self.penalty(acceleration, penalty_cap=100.)
        torch.testing.assert_close(result, torch.tensor([100., 100., 100., 25.]))
        self.assertTrue(torch.isnan(acceleration[0, 0]))
        self.assertTrue(torch.isinf(acceleration[1:3, 0]).all())

    def test_cap_applies_to_sum_and_joint_selection(self):
        acceleration = torch.tensor([[float("nan"), 8., -8.], [1., 3., 4.]])
        result = self.penalty(acceleration, joint_ids=[1, 2], penalty_cap=100.)
        torch.testing.assert_close(result, torch.tensor([100., 25.]))

    def test_invalid_caps_are_rejected(self):
        for cap in (0., -1., float("nan"), float("inf"), 1.0e38):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                self.penalty(torch.zeros(1, 12), penalty_cap=cap)


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


class TestFeetLateralDistance(unittest.TestCase):
    def penalty(self, positions, quaternions=None, body_ids=slice(None), **kwargs):
        if quaternions is None:
            quaternions = [[1., 0., 0., 0.]] * len(positions)
        env = SimpleNamespace(scene=_Scene(robot=SimpleNamespace(data=SimpleNamespace(
            body_link_pos_w=torch.tensor(positions, dtype=torch.float32),
            root_quat_w=torch.tensor(quaternions, dtype=torch.float32),
        ))))
        return rewards.feet_lateral_distance(
            env, asset_cfg=SimpleNamespace(name="robot", body_ids=body_ids), **kwargs,
        )

    def feet(self, front_width, hind_width):
        return [[.24, front_width / 2, 0], [.24, -front_width / 2, 0],
                [-.24, hind_width / 2, 0], [-.24, -hind_width / 2, 0]]

    def test_nominal_threshold_and_wider_stances_have_zero_cost(self):
        result = self.penalty([self.feet(.34244, .34244), self.feet(.26, .26), self.feet(.5, .5)])
        torch.testing.assert_close(result, torch.zeros(3))

    def test_each_pair_counts_and_crossed_feet_cannot_escape(self):
        result = self.penalty([self.feet(.13, .26), self.feet(.13, .13),
                               self.feet(0., 0.), self.feet(-.13, .26)])
        torch.testing.assert_close(result, torch.tensor([.25, .5, 2., 2.25]))

    def test_fore_aft_offsets_and_raised_feet_do_not_hide_narrow_stance(self):
        feet = self.feet(.13, .13)
        feet[0][0], feet[0][2] = 1., .4
        feet[3][0], feet[3][2] = -1., .8
        torch.testing.assert_close(self.penalty([feet]), torch.tensor([.5]))

    def test_world_translation_yaw_and_root_tilt(self):
        positions, quaternions = [], []
        for yaw in (0., math.pi / 2, -math.pi / 2, 2.3):
            c, s = math.cos(yaw), math.sin(yaw)
            positions.append([[c*x - s*y + 10, s*x + c*y - 5, z + 2]
                              for x, y, z in self.feet(.13, .13)])
            # q = yaw * pitch * roll；足端高度变化不应混入横向距离。
            cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
            cp, sp = math.cos(.2), math.sin(.2)
            cr, sr = math.cos(.3), math.sin(.3)
            quaternions.append([cy*cp*cr + sy*sp*sr, cy*cp*sr - sy*sp*cr,
                                cy*sp*cr + sy*cp*sr, sy*cp*cr - cy*sp*sr])
        torch.testing.assert_close(self.penalty(positions, quaternions), torch.full((4,), .5))

    def test_explicit_body_selection_restores_pair_order(self):
        fl, fr, hl, hr = self.feet(.13, .26)
        positions = [[[0, 0, 10], hr, fr, hl, fl]]
        torch.testing.assert_close(self.penalty(positions, body_ids=[4, 2, 3, 1]), torch.tensor([.25]))

    def test_invalid_distance_or_foot_count_is_rejected(self):
        for distance in (0., -1., float("nan"), float("inf")):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                self.penalty([self.feet(.26, .26)], minimum_distance=distance)
        with self.assertRaises(ValueError):
            self.penalty([[[0, 0, 0]] * 3])


class TestFeetLongitudinalDistance(unittest.TestCase):
    def penalty(self, positions, quaternions=None, body_ids=slice(None), **kwargs):
        if quaternions is None:
            quaternions = [[1., 0., 0., 0.]] * len(positions)
        env = SimpleNamespace(scene=_Scene(robot=SimpleNamespace(data=SimpleNamespace(
            body_link_pos_w=torch.tensor(positions, dtype=torch.float32),
            root_quat_w=torch.tensor(quaternions, dtype=torch.float32),
        ))))
        return rewards.feet_longitudinal_distance(
            env, asset_cfg=SimpleNamespace(name="robot", body_ids=body_ids), **kwargs,
        )

    @staticmethod
    def feet(left_length, right_length):
        return [[left_length / 2, .17, 0], [right_length / 2, -.17, 0],
                [-left_length / 2, .17, 0], [-right_length / 2, -.17, 0]]

    def test_nominal_threshold_and_longer_stances_have_zero_cost(self):
        result = self.penalty([self.feet(.4772, .4772), self.feet(.30, .30), self.feet(.6, .6)])
        torch.testing.assert_close(result, torch.zeros(3))

    def test_each_side_counts_and_crossed_feet_cannot_escape(self):
        result = self.penalty([self.feet(.15, .30), self.feet(.15, .15),
                               self.feet(0., 0.), self.feet(-.15, .30)])
        torch.testing.assert_close(result, torch.tensor([.25, .5, 2., 2.25]))

    def test_world_translation_yaw_and_height_do_not_change_cost(self):
        positions, quaternions = [], []
        for yaw in (0., math.pi / 2, -math.pi / 2, 2.3):
            c, s = math.cos(yaw), math.sin(yaw)
            feet = self.feet(.15, .15)
            feet[0][2], feet[3][2] = .4, .8
            positions.append([[c*x - s*y + 10, s*x + c*y - 5, z + 2] for x, y, z in feet])
            quaternions.append([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)])
        torch.testing.assert_close(self.penalty(positions, quaternions), torch.full((4,), .5))

    def test_explicit_body_selection_restores_pair_order(self):
        fl, fr, hl, hr = self.feet(.15, .30)
        positions = [[[0, 0, 10], hr, fr, hl, fl]]
        torch.testing.assert_close(self.penalty(positions, body_ids=[4, 2, 3, 1]), torch.tensor([.25]))

    def test_invalid_distance_or_foot_count_is_rejected(self):
        for distance in (0., -1., float("nan"), float("inf")):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                self.penalty([self.feet(.30, .30)], minimum_distance=distance)
        with self.assertRaises(ValueError):
            self.penalty([[[0, 0, 0]] * 3])


class TestStandTrackingRewards(unittest.TestCase):
    def evaluate(self, term, commands, lin_vels=None, ang_vel_z=None, **kwargs):
        count = len(commands)
        if lin_vels is None:
            lin_vels = [[0., 0., 0.]] * count
        if ang_vel_z is None:
            ang_vel_z = [0.] * count
        ang_vel = torch.zeros(count, 3)
        ang_vel[:, 2] = torch.tensor(ang_vel_z, dtype=torch.float32)
        command = torch.tensor(commands, dtype=torch.float32)
        env = SimpleNamespace(
            scene=_Scene(robot=SimpleNamespace(data=SimpleNamespace(
                root_lin_vel_b=torch.tensor(lin_vels, dtype=torch.float32),
                root_ang_vel_b=ang_vel,
            ))),
            command_manager=SimpleNamespace(get_command=lambda _name: command),
        )
        kwargs.setdefault("std", .1)
        return getattr(rewards, term)(
            env, command_name="base_velocity", asset_cfg=SimpleNamespace(name="robot"), **kwargs,
        )

    def test_only_zero_command_is_scored(self):
        result = self.evaluate(
            "track_lin_vel_xy_stand_exp",
            [[0., 0., 0.], [0., 0., .06], [.2, 0., 0.], [0., 0., 0.]],
        )
        torch.testing.assert_close(result, torch.tensor([1., 0., 0., 1.]))

    def test_narrow_kernel_penalizes_residual_drift_that_default_kernel_ignores(self):
        lin = self.evaluate(
            "track_lin_vel_xy_stand_exp", [[0., 0., 0.]] * 4,
            lin_vels=[[0., 0., 0.], [.01, 0., 0.], [0., .07, 0.], [0., 0., 9.]],
        )
        # 0.07 m/s 残速在默认 std=0.5 下只损失 1.9%，收紧到 0.1 后损失 39%。
        self.assertAlmostEqual(lin[1].item(), math.exp(-.01), places=6)
        self.assertAlmostEqual(lin[2].item(), math.exp(-.49), places=6)
        self.assertLess(lin[2].item(), .62)
        self.assertEqual(lin[3].item(), 1.)  # z 速度不参与水平项
        yaw = self.evaluate(
            "track_ang_vel_z_stand_exp", [[0., 0., 0.]] * 2, ang_vel_z=[0., .07],
        )
        self.assertAlmostEqual(yaw[1].item(), math.exp(-.49), places=6)

    def test_command_gate_is_off_for_nonzero_components(self):
        # 指令与真实角速度一致时奖励为 1，门控只取决于指令是否近零。
        result = self.evaluate(
            "track_ang_vel_z_stand_exp",
            [[0., 0., .04], [0., 0., -.04], [0., 0., .06]],
            ang_vel_z=[.04, -.04, .06],
            command_threshold=.05,
        )
        torch.testing.assert_close(result, torch.tensor([1., 1., 0.]))

    def test_invalid_parameters_are_rejected(self):
        for term in ("track_lin_vel_xy_stand_exp", "track_ang_vel_z_stand_exp"):
            for std in (0., -1., float("nan"), float("inf")):
                with self.subTest(term=term, std=std), self.assertRaises(ValueError):
                    self.evaluate(term, [[0., 0., 0.]], std=std)
            for threshold in (0., -1., float("nan"), float("inf")):
                with self.subTest(term=term, threshold=threshold), self.assertRaises(ValueError):
                    self.evaluate(term, [[0., 0., 0.]], command_threshold=threshold)


class TestFeetAirTime(unittest.TestCase):
    def reward(self, commands, air_times=None, first_contacts=None, **kwargs):
        count = len(commands)
        if air_times is None:
            air_times = [[.6, .6, .6, .6]] * count
        if first_contacts is None:
            first_contacts = [[1., 1., 1., 1.]] * count
        sensor = SimpleNamespace(
            data=SimpleNamespace(last_air_time=torch.tensor(air_times, dtype=torch.float32)),
            compute_first_contact=lambda _dt: torch.tensor(first_contacts, dtype=torch.float32),
        )
        command = torch.tensor(commands, dtype=torch.float32)
        env = SimpleNamespace(
            scene=_Scene(contact_forces=sensor), step_dt=.02,
            command_manager=SimpleNamespace(get_command=lambda _name: command),
        )
        return rewards.feet_air_time(
            env, command_name="base_velocity", threshold=.5,
            sensor_cfg=SimpleNamespace(name="contact_forces", body_ids=slice(None)), **kwargs,
        )

    def test_translation_and_pure_yaw_both_enable_stepping_reward(self):
        result = self.reward([[.2, 0., 0.], [0., 0., .2], [0., 0., -.2], [0., 0., 0.]])
        torch.testing.assert_close(result, torch.tensor([.4, .4, .4, 0.]))

    def test_command_deadbands_are_independent(self):
        result = self.reward(
            [[.1, 0., .2], [.2, 0., .1], [.1, 0., .1]],
            command_threshold=.1, yaw_command_threshold=.1,
        )
        torch.testing.assert_close(result, torch.tensor([.4, .4, 0.]))

    def test_invalid_command_threshold_is_rejected(self):
        for kwargs in ({"command_threshold": -1.}, {"command_threshold": float("nan")},
                       {"yaw_command_threshold": -1.}, {"yaw_command_threshold": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.reward([[0., 0., 0.]], **kwargs)

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
