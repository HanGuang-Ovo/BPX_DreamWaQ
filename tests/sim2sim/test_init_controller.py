"""Run: python -m unittest discover -s tests/sim2sim -p 'test_init_controller.py' -v"""

from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from bpx_dreamwaq.sim2sim.config import load_config
from bpx_dreamwaq.sim2sim.init_controller import FootTrajectoryInit, LegKinematics, quintic
from bpx_dreamwaq.runtime.policy import ZeroPolicy
from bpx_dreamwaq.sim2sim.runner import Sim2SimRunner
from bpx_dreamwaq.runtime.supervisor import BehaviorMode, BehaviorSupervisor, SupervisorState


CONFIG = Path(__file__).resolve().parents[2] / 'configs/sim2sim/bpx_terrain_history.toml'


def make_runner(config=None):
    cfg = config or load_config(CONFIG)
    return Sim2SimRunner(cfg, ZeroPolicy(12), BehaviorSupervisor(cfg.supervisor))


class InitControllerTest(unittest.TestCase):
    def test_fk_ik_against_mujoco_with_shuffled_joint_order(self):
        cfg = load_config(CONFIG)
        cfg = replace(cfg, joint_names=tuple(reversed(cfg.joint_names)))
        runner = make_runner(cfg)
        robot, model, data = runner.robot, runner.model, runner.data
        robot.reset()
        kin = LegKinematics(robot)
        rng = np.random.default_rng(174)
        for _ in range(30):
            q = np.empty(12)
            q[kin.indices] = rng.uniform([-.2, .7, -2.6], [.2, 1.1, -1.2], (4, 3))
            data.qpos[robot.joint_qpos_addresses] = q
            mujoco.mj_forward(model, data)
            rotation = data.xmat[robot.base_body_id].reshape(3, 3)
            actual = np.array([
                (data.xpos[model.body(f'{leg}_toe_link').id]
                 - data.xpos[model.body(f'{leg}_hip_link').id]) @ rotation
                for leg in ('fl', 'fr', 'hl', 'hr')
            ])
            np.testing.assert_allclose(kin.forward(q), actual, atol=1e-12)
            np.testing.assert_allclose(kin.inverse(actual), q, atol=1e-12)

    def test_invalid_targets_rejected(self):
        controller = make_runner().init_controller
        for depth in (1.0, .001, -.2):
            feet = controller.final_feet.copy()
            feet[:, 2] = -depth
            with self.assertRaises(ValueError):
                controller.kinematics.inverse(feet)
        feet = controller.final_feet.copy()
        feet[0, 0] = np.nan
        with self.assertRaises(ValueError):
            controller.kinematics.inverse(feet)

    def test_quintic_has_zero_endpoint_velocity_and_acceleration(self):
        h = 1e-4
        for endpoint in (0., 1.):
            direction = 1 if endpoint == 0 else -1
            samples = np.array([quintic(endpoint + direction * i * h) for i in range(4)])
            velocity = (-3 * samples[0] + 4 * samples[1] - samples[2]) / (2 * h)
            acceleration = (2 * samples[0] - 5 * samples[1] + 4 * samples[2] - samples[3]) / h**2
            self.assertLess(abs(velocity), 1e-6)
            self.assertLess(abs(acceleration), 1e-4)
        self.assertEqual(quintic(-1), 0)
        self.assertEqual(quintic(2), 1)

    def test_capture_restart_pause_and_speed_limit(self):
        runner = make_runner()
        robot, data = runner.robot, runner.data
        robot.reset()
        controller = runner.init_controller
        controller.start()
        np.testing.assert_allclose(robot.desired_joint_position(controller()), robot.joint_position(), atol=1e-7)
        data.time += .1
        first = controller().copy()
        previous = controller.target.copy()
        data.qpos[robot.joint_qpos_addresses[0]] += .5
        data.time += .005
        controller()
        self.assertAlmostEqual(controller.progress, .1)
        self.assertEqual(controller.phase, 'paused')
        self.assertLessEqual(np.max(abs(controller.target - previous)), .005 * controller.config.joint_speed_limit + 1e-12)
        robot.reset()
        data.time = 3.0
        controller.start()
        self.assertEqual(controller.progress, 0)
        self.assertEqual(controller.started_at, 3.0)
        self.assertFalse(controller.complete)
        self.assertFalse(np.array_equal(controller(), first))
        controller.progress = controller.config.support_duration
        with patch.object(robot, 'feet_in_contact', return_value=3):
            data.time += .005
            controller()
        self.assertEqual(controller.progress, controller.config.support_duration)
        self.assertEqual(controller.phase, 'paused')

    def test_supervisor_waits_for_trajectory_before_confirming(self):
        cfg = load_config(CONFIG)
        supervisor = BehaviorSupervisor(cfg.supervisor, init_available=True)
        supervisor.update((0, 0, 0), SupervisorState(.13, 0, 0, 0), .02, True)
        for _ in range(250):
            result = supervisor.update((0, 0, 0), SupervisorState(.4, 0, 0, 0, 4, False), .02)
        self.assertEqual(result.mode, BehaviorMode.INIT)
        for _ in range(25):
            result = supervisor.update((0, 0, 0), SupervisorState(.4, 0, 0, 0, 4, True), .02)
        self.assertEqual(result.mode, BehaviorMode.STAND)

    def test_physics_delayed_trigger_and_repeated_runs(self):
        runner = make_runner()
        actual_apply_pd = runner.robot.apply_pd
        samples = []

        def record(action, **kwargs):
            result = actual_apply_pd(action, **kwargs)
            if runner.supervisor.mode == BehaviorMode.INIT:
                samples.append((runner.data.time, result[0].copy(), result[1].copy(),
                                runner.robot.tilt_angle(), kwargs['stiffness'],
                                np.max(abs(runner.robot.joint_velocity())),
                                np.linalg.norm(runner.robot.state().base_angular_velocity)))
            return result

        # A settled prone pose exposes the Kd=3 / 5 ms explicit-PD instability.
        for delay in (0., .3, 3., 10.):
            samples.clear()
            with patch.object(runner.robot, 'apply_pd', side_effect=record), redirect_stdout(io.StringIO()):
                result = runner.run((0, 0, 0), init_request_source=lambda: runner.data.time >= delay,
                                    duration=delay + 4.65, realtime=False, print_interval=100)
            self.assertEqual(result.termination_reason, 'duration')
            self.assertEqual(runner.supervisor.mode, BehaviorMode.STAND)
            self.assertTrue(runner.init_controller.complete)
            self.assertGreater(runner.robot.base_height(), .30)
            self.assertEqual(runner.robot.feet_in_contact(), 4)
            self.assertGreaterEqual(samples[0][0], delay)
            self.assertGreaterEqual(samples[-1][0] - samples[0][0], 4.45)
            self.assertLess(max(s[3] for s in samples), .15)
            self.assertLess(max(np.max(abs(s[2])) for s in samples), 20.)
            self.assertLess(max(s[5] for s in samples), 3.)
            self.assertLess(max(s[6] for s in samples), 1.)
            self.assertAlmostEqual(samples[0][4], 40.)
            self.assertAlmostEqual(samples[-1][4], 80.)
            for previous, current in zip(samples, samples[1:]):
                dt = current[0] - previous[0]
                if dt > 1e-8:
                    self.assertLessEqual(np.max(abs(current[1] - previous[1])) / dt, 2.001)

    def test_timeout_does_not_exit_when_support_never_established(self):
        cfg = load_config(CONFIG)
        cfg = replace(cfg, init_controller=replace(cfg.init_controller, timeout=4.2))
        runner = make_runner(cfg)
        with patch.object(runner.robot, 'feet_in_contact', return_value=0), redirect_stdout(io.StringIO()):
            result = runner.run((0, 0, 0), init_request_source=lambda: True, duration=5,
                                realtime=False, print_interval=100)
        self.assertEqual(result.termination_reason, 'duration')
        self.assertGreaterEqual(result.simulated_seconds, 5.)
        self.assertFalse(runner.init_controller.complete)
        self.assertAlmostEqual(runner.init_controller.progress, 1.)

    def test_policy_continues_after_low_height_tilt_and_torso_contact(self):
        runner = make_runner()
        actual_height = runner.robot.base_height
        actual_tilt = runner.robot.tilt_angle
        calls = []
        runner.policy = lambda observation: calls.append(float(runner.data.time)) or np.zeros(12)
        with patch.object(runner.robot, 'base_height', side_effect=lambda: (
                actual_height() if runner.data.time < 5. else .10)), \
             patch.object(runner.robot, 'tilt_angle', side_effect=lambda: (
                actual_tilt() if runner.data.time < 5. else 1.2)), \
             patch.object(runner.robot, 'torso_touches_floor', return_value=True), \
             redirect_stdout(io.StringIO()):
            result = runner.run((.4, 0, 0), init_request_source=lambda: True, duration=6,
                                realtime=False, print_interval=100, terminate_on_fall=True)
        self.assertEqual(result.termination_reason, 'duration')
        self.assertGreaterEqual(result.simulated_seconds, 6.)
        self.assertGreater(max(calls), 5.9)
        self.assertEqual(runner.supervisor.mode, BehaviorMode.WALK)

    def test_pose_protection_remains_enabled_outside_sim2sim(self):
        cfg = load_config(CONFIG)
        supervisor = BehaviorSupervisor(cfg.supervisor, init_available=True)
        result = supervisor.update((.4, 0, 0), SupervisorState(.4, 1.2, 0, 0), .02)
        self.assertEqual(result.mode, BehaviorMode.DISABLED)
        self.assertEqual(result.reason, 'unsupported_init_pose')

    def test_init_tilt_does_not_exit_simulation(self):
        runner = make_runner()
        with patch.object(runner.robot, 'tilt_angle', return_value=1.2), \
             redirect_stdout(io.StringIO()):
            result = runner.run((0, 0, 0), init_request_source=lambda: True, duration=.1,
                                realtime=False, print_interval=100)
        self.assertEqual(result.termination_reason, 'duration')
        self.assertEqual(runner.supervisor.mode, BehaviorMode.INIT)

    def test_config_validation(self):
        source = CONFIG.read_text()
        for key, invalid in [('duration', '0.0'), ('support_duration', '4.0'),
                             ('standing_depth', '0.1'), ('stiffness', 'nan'), ('mode', '"policy"')]:
            section_start = source.index('[init_controller]')
            section_end = source.index('[initial_state]', section_start)
            lines = source[section_start:section_end].splitlines()
            lines = [f'{key} = {invalid}' if line.startswith(f'{key} =') else line for line in lines]
            changed = source[:section_start] + '\n'.join(lines) + '\n' + source[section_end:]
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'config.toml'
                path.write_text(changed)
                with self.assertRaises(ValueError):
                    load_config(path)


if __name__ == '__main__':
    unittest.main()
