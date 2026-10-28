"""The terminal state must be inspected before Isaac Lab resets it."""
from types import SimpleNamespace
import unittest

import torch
from bpx_dreamwaq.tasks.mdp.numerical_guard import check_robot_state


def make_env():
    data = SimpleNamespace(root_pos_w=torch.zeros(2, 3), root_quat_w=torch.ones(2, 4),
                           root_lin_vel_w=torch.zeros(2, 3), root_ang_vel_w=torch.zeros(2, 3),
                           joint_pos=torch.zeros(2, 12), joint_vel=torch.zeros(2, 12),
                           applied_torque=torch.zeros(2, 12))
    return SimpleNamespace(scene={"robot": SimpleNamespace(data=data)}, num_envs=2,
                           episode_length_buf=torch.tensor([10, 20]))


class TestNumericalGuard(unittest.TestCase):
    def test_normal_state_is_not_terminated(self):
        env = make_env()
        env.scene['robot'].data.joint_vel[0, 0] = 25
        self.assertFalse(check_robot_state(env).any())

    def test_extreme_finite_state_and_nonfinite_state_stop_with_details(self):
        for field, value in (("root_ang_vel_w", 1e12), ("root_lin_vel_w", float('nan')),
                             ("joint_vel", 101.), ("joint_pos", float('inf'))):
            with self.subTest(field=field):
                env = make_env()
                getattr(env.scene['robot'].data, field)[1, 0] = value
                with self.assertRaisesRegex(FloatingPointError, r"env_id=1.*episode_step.*20") as ctx:
                    check_robot_state(env, fail_fast=True)
                self.assertIn(field, str(ctx.exception))

    def test_default_quarantines_only_bad_env_and_clears_on_next_step(self):
        env = make_env()
        env.scene['robot'].data.root_ang_vel_w[1, 0] = 1e12
        self.assertEqual(check_robot_state(env).tolist(), [False, True])
        self.assertEqual(env._numerical_fault_count, 1)
        env.scene['robot'].data.root_ang_vel_w.zero_()
        self.assertFalse(check_robot_state(env).any())
        self.assertFalse(env._numerical_invalid.any())

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            check_robot_state(make_env(), max_joint_speed=float('inf'))
