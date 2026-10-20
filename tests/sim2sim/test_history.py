"""Run: python -m unittest discover -s tests/sim2sim -p 'test_history.py' -v"""

from pathlib import Path
from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from bpx_dreamwaq.sim2sim.config import load_config
from bpx_dreamwaq.runtime.history import ObservationHistory
from bpx_dreamwaq.sim2sim.runner import Sim2SimRunner, _policy_path_text, _update_viewer_overlay
from bpx_dreamwaq.runtime.supervisor import BehaviorMode, SupervisorDecision


class RecordingPolicy:
    def __init__(self, dimension, action):
        self.dimension = dimension
        self.action = action
        self.inputs = []

    def __call__(self, observation):
        assert observation.shape == (self.dimension,)
        self.inputs.append(observation.copy())
        return np.full(12, self.action, dtype=np.float32)


class HistoryTest(unittest.TestCase):
    def test_layout_fill_and_rollover(self):
        history = ObservationHistory(10)
        frames = [np.arange(48, dtype=np.float32) + 100 * i for i in range(13)]
        # Independent reference: explicit term/time/feature traversal.
        boundaries = (0, 3, 6, 9, 12, 24, 36, 48)
        for i, frame in enumerate(frames):
            result = history.append(frame)
            times = [max(0, t) for t in range(i - 9, i + 1)]
            expected = [frames[t][j] for lo, hi in zip(boundaries, boundaries[1:])
                        for t in times for j in range(lo, hi)]
            np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(ObservationHistory(1).append(frames[-1]), frames[-1])

    def test_actor_layout_without_velocity(self):
        history = ObservationHistory(10, include_base_lin_vel=False)
        reference = ObservationHistory(10)
        for i in range(13):
            full = np.arange(48, dtype=np.float32) + 100 * i
            np.testing.assert_array_equal(history.append(full[3:]), reference.append(full)[30:])

    def test_velocity_cannot_reach_actor(self):
        cfg = load_config(Path(__file__).resolve().parents[2] / 'configs/sim2sim/bpx_terrain_history.toml')
        policy = RecordingPolicy(450, 0)
        runner = Sim2SimRunner(cfg, policy)
        frames = [np.arange(48, dtype=np.float32) for _ in range(3)]
        for i, frame in enumerate(frames):
            frame[:3] = 1000 * i
        with patch.object(runner.robot, 'observation', side_effect=frames):
            runner.run((0, 0, 0), duration=.05, realtime=False, print_interval=100)
        self.assertEqual(len(policy.inputs), 3)
        for value in policy.inputs[1:]:
            np.testing.assert_array_equal(value, policy.inputs[0])

    def test_stand_walk_stopping_share_policy_and_continuous_history(self):
        for name in ('bpx_history.toml', 'bpx_flat.toml'):
            with self.subTest(config=name):
                cfg = load_config(Path(__file__).resolve().parents[2] / 'configs/sim2sim' / name)
                policy = RecordingPolicy(cfg.observation.dimension, 0)
                supervisor = Mock()
                modes = (BehaviorMode.STAND, BehaviorMode.WALK, BehaviorMode.STOPPING, BehaviorMode.STAND)
                supervisor.update.side_effect = [
                    SupervisorDecision(mode, np.array([.2, 0, 0]) if mode == BehaviorMode.WALK else np.zeros(3),
                                       True, 'test')
                    for mode in modes
                ]
                runner = Sim2SimRunner(cfg, policy, supervisor)
                frames = [np.arange(48, dtype=np.float32) + 100 * i for i in range(4)]

                def observation(command, action):
                    frame = frames[len(policy.inputs)].copy()
                    frame[9:12] = command
                    return frame

                with patch.object(runner.robot, 'observation', side_effect=observation):
                    runner.run((.2, 0, 0), duration=.07, realtime=False, print_interval=100)
                self.assertEqual(len(policy.inputs), 4)
                expected = ObservationHistory(cfg.observation.history_length,
                                              include_base_lin_vel=cfg.observation.include_base_lin_vel)
                for actual, frame, mode in zip(policy.inputs, frames, modes):
                    frame[9:12] = [.2, 0, 0] if mode == BehaviorMode.WALK else 0
                    if not cfg.observation.include_base_lin_vel:
                        frame = frame[3:]
                    np.testing.assert_array_equal(actual, expected.append(frame))

    def test_viewer_displays_single_loaded_policy_and_controller(self):
        cfg = load_config(Path(__file__).resolve().parents[2] / 'configs/sim2sim/bpx_history.toml')
        policy = RecordingPolicy(450, 0)
        policy.path = cfg.repository_root / 'overridden_policy.onnx'
        runner = Sim2SimRunner(cfg, policy)
        path = _policy_path_text(policy, cfg.repository_root)
        viewer = Mock()
        runner.robot.reset()
        zeros = np.zeros(12, dtype=np.float32)
        _update_viewer_overlay(viewer, runner.robot, np.zeros(3), zeros, zeros, 'stand', 'policy', path)
        overlays = viewer.set_texts.call_args.args[0]
        self.assertEqual(len(overlays), 2)
        self.assertIn('mode', overlays[0][2])
        self.assertEqual(overlays[1][2], 'Policy: overridden_policy.onnx\nController: policy')

    def test_runner_reset_and_previous_action(self):
        cfg = load_config(Path(__file__).resolve().parents[2] / 'configs/sim2sim/bpx_terrain_history.toml')
        policy = RecordingPolicy(450, .1)
        runner = Sim2SimRunner(cfg, policy)
        for _ in range(2):
            policy.inputs.clear()
            runner.run((.2, 0, 0), duration=.05, realtime=False, print_interval=100)
            self.assertGreaterEqual(len(policy.inputs), 2)
            np.testing.assert_array_equal(policy.inputs[0][330:], 0)
            np.testing.assert_allclose(policy.inputs[1][-12:], .1)
            np.testing.assert_array_equal(policy.inputs[1][330:-12], 0)

    def test_init_to_shared_policy_history_contains_actual_applied_actions(self):
        cfg = load_config(Path(__file__).resolve().parents[2] / 'configs/sim2sim/bpx_terrain_history.toml')
        cfg = replace(cfg, supervisor=replace(cfg.supervisor, action_blend_duration=.2))
        policy = RecordingPolicy(450, .2)
        supervisor = Mock()
        supervisor.update.side_effect = [
            SupervisorDecision(mode, np.zeros(3, dtype=np.float32), True, 'test')
            for mode in (BehaviorMode.INIT, BehaviorMode.STAND, BehaviorMode.WALK)
        ]
        runner = Sim2SimRunner(cfg, policy, supervisor)
        applied = []
        desired_position = runner.robot.desired_joint_position

        def record_action(action):
            applied.append((float(runner.data.time), action.copy()))
            return desired_position(action)

        with patch.object(runner.robot, 'desired_joint_position', side_effect=record_action):
            runner.run((0, 0, 0), duration=.05, realtime=False, print_interval=100)
        self.assertEqual(len(policy.inputs), 2)
        reset_action = (
            np.asarray(cfg.initial_state.joint_position, dtype=np.float32)
            - np.asarray(cfg.initial_state.default_joint_position, dtype=np.float32)
        ) / cfg.control.action_scale
        np.testing.assert_array_equal(policy.inputs[-1][330:-24], np.tile(reset_action, 8))
        init_action = [action for time, action in applied if time < .02 - 1e-6][-1]
        stand_action = [action for time, action in applied if time < .04 - 1e-6][-1]
        np.testing.assert_array_equal(policy.inputs[-1][-24:-12], init_action)
        np.testing.assert_array_equal(policy.inputs[-1][-12:], stand_action)
        self.assertFalse(np.allclose(policy.inputs[-1][-12:], policy.action))


if __name__ == '__main__':
    unittest.main()
