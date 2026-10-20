"""速度遥测集成检查，不依赖桌面。"""

from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from bpx_dreamwaq.sim2sim.config import load_config
from bpx_dreamwaq.runtime.policy import ZeroPolicy
from bpx_dreamwaq.sim2sim.runner import Sim2SimRunner
from bpx_dreamwaq.runtime.supervisor import BehaviorSupervisor
from bpx_dreamwaq.sim2sim.velocity_plot import VelocityPlot


class VelocityPlotTest(unittest.TestCase):
    def test_runner_samples_body_velocity_and_filtered_command(self):
        cfg = load_config(Path(__file__).resolve().parents[2] / 'configs/sim2sim/bpx_terrain_history.toml')
        for supervised in (False, True):
            supervisor = BehaviorSupervisor(cfg.supervisor, init_available=True) if supervised else None
            runner = Sim2SimRunner(cfg, ZeroPolicy(12), supervisor=supervisor)
            samples = []

            class Recorder:
                def __init__(self, *args):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def publish(self, timestamp, command, actual, mode):
                    linear, angular = runner.robot.base_velocity_body()
                    np.testing.assert_allclose(actual, (linear[0], linear[1], angular[2]))
                    np.testing.assert_allclose(command, (0, 0, 0) if supervised else (.3, -.2, .4))
                    self_mode = 'waiting_init' if supervised else 'direct'
                    if mode != self_mode:
                        raise AssertionError((mode, self_mode))
                    samples.append(timestamp)

            with patch('bpx_dreamwaq.sim2sim.velocity_plot.VelocityPlot', Recorder), redirect_stdout(io.StringIO()):
                result = runner.run((.3, -.2, .4), duration=.05, realtime=False, velocity_plot=True)
            self.assertEqual(result.termination_reason, 'duration')
            self.assertGreater(len(samples), result.policy_steps)
            np.testing.assert_allclose(samples, np.arange(1, len(samples) + 1) * cfg.simulation.timestep)

    def test_window_validation(self):
        for seconds in (0, -1, 31, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                VelocityPlot(.005, seconds)


if __name__ == '__main__':
    unittest.main()
