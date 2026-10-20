"""Torque telemetry and plot data tests (no desktop required)."""

import csv
from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from bpx_dreamwaq.sim2sim.config import load_config
from bpx_dreamwaq.runtime.policy import ZeroPolicy
from bpx_dreamwaq.sim2sim.runner import Sim2SimRunner
from bpx_dreamwaq.sim2sim.torque_plot import TorqueHistory, TorquePlot, envelope_indices


class TorquePlotTest(unittest.TestCase):
    def test_history_window_reset_and_export(self):
        history = TorqueHistory(('fr_knee_joint', 'fl_hip_roll_joint'), .005)
        for t in range(41):
            history.append((t, (t, -t), 'init', 0))
        self.assertEqual(history.rows[0][0], 10)
        rows = history.visible(5)
        self.assertEqual([row[0] for row in rows], list(range(35, 41)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'torques.csv'
            history.export(path, rows)
            with path.open() as stream:
                exported = list(csv.DictReader(stream))
            self.assertEqual(float(exported[-1]['fr_knee_joint']), 40)
            self.assertEqual(float(exported[-1]['fl_hip_roll_joint']), -40)
        history.append((.005, (1, 2), 'waiting_init', 3))
        self.assertEqual(len(history.rows), 1)
        history.append((.01, (np.nan, 2), 'init', 3))
        self.assertEqual(len(history.rows), 1)

    def test_envelope_preserves_positive_negative_spikes(self):
        values = np.zeros(6000)
        values[500] = 30
        values[501] = -30
        indices = envelope_indices(values, 240)
        self.assertIn(500, indices)
        self.assertIn(501, indices)
        self.assertLessEqual(len(indices), 482)
        self.assertTrue(np.all(np.diff(indices) > 0))

    def test_window_validation(self):
        for seconds in (0, -1, 31, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                TorquePlot((), 30, .005, seconds)

    def test_runner_publishes_actual_force_each_physics_step(self):
        cfg = load_config(Path(__file__).resolve().parents[2] / 'configs/sim2sim/bpx_terrain_history.toml')
        for mode in ('explicit', 'implicit'):
            runner = Sim2SimRunner(replace(cfg, control=replace(cfg.control, mode=mode)), ZeroPolicy(12))
            samples = []

            class Recorder:
                def __init__(self, *args):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def publish(self, timestamp, values, behavior):
                    expected = runner.data.qfrc_actuator[runner.robot.joint_dof_addresses]
                    np.testing.assert_array_equal(values, expected)
                    samples.append((timestamp, values.copy()))

            original = runner.robot.apply_pd

            def misleading_estimate(*args, **kwargs):
                target, _ = original(*args, **kwargs)
                return target, np.full(12, 999.)

            with patch('bpx_dreamwaq.sim2sim.torque_plot.TorquePlot', Recorder), \
                    patch.object(runner.robot, 'apply_pd', side_effect=misleading_estimate), \
                    redirect_stdout(io.StringIO()):
                result = runner.run((0, 0, 0), duration=.05, realtime=False, torque_plot=True)
            self.assertEqual(result.termination_reason, 'duration')
            self.assertGreater(len(samples), result.policy_steps)
            np.testing.assert_allclose([s[0] for s in samples], np.arange(1, len(samples) + 1) * .005)
            self.assertLessEqual(np.max(abs(np.array([s[1] for s in samples]))), 30.00001)


if __name__ == '__main__':
    unittest.main()
