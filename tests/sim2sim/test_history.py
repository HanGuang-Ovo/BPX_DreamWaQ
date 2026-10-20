"""Run: python -m unittest discover -s tests/sim2sim -p 'test_history.py' -v"""

from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np

from bpx_dreamwaq.sim2sim.behaviors import BehaviorPolicies
from bpx_dreamwaq.sim2sim.config import load_config
from bpx_dreamwaq.runtime.history import ObservationHistory
from bpx_dreamwaq.runtime.policy import OnnxPolicy, StandOnnxPolicy
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

    def test_stand_onnx_adapter_selects_history_from_model_input(self):
        frame = np.arange(48, dtype=np.float32)
        history = ObservationHistory(10, include_base_lin_vel=False).append(frame[3:])
        for input_width, expected in ((450, history), (48, frame), (45, frame[3:])):
            def fake_init(policy, path, observation_dimension, action_dimension):
                policy.session = Mock()
                policy.session.get_inputs.return_value = [Mock(shape=[1, input_width])]

            with patch.object(OnnxPolicy, '__init__', fake_init), \
                 patch.object(OnnxPolicy, '__call__', lambda policy, value: value.copy()):
                policy = StandOnnxPolicy(Path('unused.onnx'), 48, 12)
                self.assertEqual(policy.uses_history, input_width == 450)
                actual = policy(history if policy.uses_history else frame)
                np.testing.assert_array_equal(actual, expected)

    def test_stand_history_omits_velocity_and_zeros_commands(self):
        cfg = load_config(Path(__file__).resolve().parents[2] / 'configs/sim2sim/bpx_terrain_history.toml')
        walk = RecordingPolicy(450, 0)
        stand = RecordingPolicy(450, 0)
        stand.uses_history = True
        supervisor = Mock()
        supervisor.update.return_value = SupervisorDecision(
            BehaviorMode.STAND, np.zeros(3, dtype=np.float32), True, 'test'
        )
        runner = Sim2SimRunner(cfg, walk, supervisor, BehaviorPolicies(walk, stand))
        frames = [np.arange(48, dtype=np.float32) + 100 * i for i in range(3)]
        for i, frame in enumerate(frames):
            frame[:3] = 1000 * (i + 1)
            frame[9:12] = [1, 2, 3]
        with patch.object(runner.robot, 'observation', side_effect=frames):
            runner.run((0, 0, 0), duration=.05, realtime=False, print_interval=100)
        self.assertEqual(len(stand.inputs), 3)
        expected = ObservationHistory(10, include_base_lin_vel=False)
        for actual, frame in zip(stand.inputs, frames):
            stand_frame = frame[3:].copy()
            stand_frame[6:9] = 0
            np.testing.assert_array_equal(actual, expected.append(stand_frame))
        np.testing.assert_array_equal(stand.inputs[0][:30], np.tile(frames[0][3:6], 10))
        np.testing.assert_array_equal(stand.inputs[-1][60:90], 0)

    def test_stand_history_restarts_after_walk_with_flat_locomotion(self):
        cfg = load_config(Path(__file__).resolve().parents[2] / 'configs/sim2sim/bpx_flat.toml')
        walk = RecordingPolicy(48, 0)
        stand = RecordingPolicy(450, 0)
        stand.uses_history = True
        supervisor = Mock()
        supervisor.update.side_effect = [
            SupervisorDecision(mode, np.zeros(3, dtype=np.float32), True, 'test')
            for mode in (BehaviorMode.STAND, BehaviorMode.WALK, BehaviorMode.STAND)
        ]
        runner = Sim2SimRunner(cfg, walk, supervisor, BehaviorPolicies(walk, stand))
        frames = [np.arange(48, dtype=np.float32) + 100 * i for i in range(3)]
        with patch.object(runner.robot, 'observation', side_effect=frames):
            runner.run((0, 0, 0), duration=.05, realtime=False, print_interval=100)
        self.assertEqual(len(walk.inputs), 1)
        self.assertEqual(len(stand.inputs), 2)
        for actual, frame in zip(stand.inputs, (frames[0], frames[2])):
            stand_frame = frame[3:].copy()
            stand_frame[6:9] = 0
            expected = ObservationHistory(10, include_base_lin_vel=False).append(stand_frame)
            np.testing.assert_array_equal(actual, expected)

    def test_viewer_displays_loaded_paths_and_active_policy(self):
        cfg = load_config(Path(__file__).resolve().parents[2] / 'configs/sim2sim/bpx_terrain_history.toml')
        walk = RecordingPolicy(450, 0)
        stand = RecordingPolicy(48, 0)
        init = RecordingPolicy(48, 0)
        walk.path = cfg.repository_root / 'walk.onnx'
        stand.path = cfg.repository_root / 'overridden_stand.onnx'
        init.path = cfg.repository_root / 'init.onnx'
        runner = Sim2SimRunner(cfg, walk, Mock(), BehaviorPolicies(walk, stand, init))
        paths = tuple(_policy_path_text(policy, cfg.repository_root) for policy in (walk, stand, init))
        viewer = Mock()
        runner.robot.reset()
        zeros = np.zeros(12, dtype=np.float32)
        _update_viewer_overlay(viewer, runner.robot, np.zeros(3), zeros, zeros, 'walk', 'stand', paths)
        overlays = viewer.set_texts.call_args.args[0]
        self.assertEqual(len(overlays), 2)
        self.assertIn('mode', overlays[0][2])
        labels = overlays[1][2]
        self.assertIn('  WALK: walk.onnx', labels)
        self.assertIn('* STAND: overridden_stand.onnx', labels)
        self.assertIn('  INIT: init.onnx', labels)

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

    def test_supervisor_mixed_dimensions_and_actual_action(self):
        cfg = load_config(Path(__file__).resolve().parents[2] / 'configs/sim2sim/bpx_terrain_history.toml')
        walk = RecordingPolicy(450, .2)
        stand = RecordingPolicy(48, -.1)
        init = RecordingPolicy(48, .1)
        supervisor = Mock()
        supervisor.update.side_effect = [
            SupervisorDecision(mode, np.zeros(3, dtype=np.float32), True, 'test')
            for mode in (BehaviorMode.INIT, BehaviorMode.STAND, BehaviorMode.WALK)
        ]
        runner = Sim2SimRunner(cfg, walk, supervisor, BehaviorPolicies(walk, stand, init))
        applied = []
        desired_position = runner.robot.desired_joint_position

        def record_action(action):
            applied.append((float(runner.data.time), action.copy()))
            return desired_position(action)

        with patch.object(runner.robot, 'desired_joint_position', side_effect=record_action):
            runner.run((0, 0, 0), duration=.05, realtime=False, print_interval=100)
        self.assertEqual([len(p.inputs) for p in (init, stand, walk)], [1, 1, 1])
        # Compare the history with the actual mixed actions sent to the PD target.
        reset_action = (
            np.asarray(cfg.initial_state.joint_position, dtype=np.float32)
            - np.asarray(cfg.initial_state.default_joint_position, dtype=np.float32)
        ) / cfg.control.action_scale
        np.testing.assert_array_equal(walk.inputs[0][330:-24], np.tile(reset_action, 8))
        init_action = [action for time, action in applied if time < .02 - 1e-6][-1]
        stand_action = [action for time, action in applied if time < .04 - 1e-6][-1]
        np.testing.assert_array_equal(walk.inputs[0][-24:-12], init_action)
        np.testing.assert_array_equal(walk.inputs[0][-12:], stand_action)
        self.assertFalse(np.allclose(walk.inputs[0][-12:], stand.action))


if __name__ == '__main__':
    unittest.main()
