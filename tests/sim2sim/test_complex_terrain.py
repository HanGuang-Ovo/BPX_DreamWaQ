"""Exercise complex-course collisions, ground queries, reproducibility and Init."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import unittest

import mujoco
import numpy as np

from bpx_dreamwaq.runtime.policy import ZeroPolicy
from bpx_dreamwaq.runtime.supervisor import BehaviorSupervisor
from bpx_dreamwaq.sim2sim.complex_terrain import complex_lanes
from bpx_dreamwaq.sim2sim.config import COMPLEX_TERRAIN_KINDS, TerrainConfig, load_config
from bpx_dreamwaq.sim2sim.robot import BpxMujocoRobot
from bpx_dreamwaq.sim2sim.runner import Sim2SimRunner
from bpx_dreamwaq.sim2sim.terrain import load_model


CONFIG = Path(__file__).resolve().parents[2] / "configs/sim2sim/bpx_mixed.toml"


class ComplexTerrainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(CONFIG)
        cls.model = load_model(cls.cfg)

    def robot(self, model=None, config=None):
        model = self.model if model is None else model
        robot = BpxMujocoRobot(model, mujoco.MjData(model), config or self.cfg)
        robot.reset()
        return robot

    def move(self, robot, x, y, z=1.5):
        robot.data.qpos[robot.base_qpos_address:robot.base_qpos_address + 3] = (x, y, z)
        mujoco.mj_forward(robot.model, robot.data)

    def test_metric_surfaces_and_flat_gaps(self):
        robot = self.robot()
        expected = {
            "ramps": [(3.0, 0.0, np.tan(np.deg2rad(12))),
                      (6.0, 0.0, 2 * np.tan(np.deg2rad(12))),
                      (9.0, 0.0, np.tan(np.deg2rad(12)))],
            "cross_slope": [(6.0, -0.5, .5 * np.tan(np.deg2rad(10))),
                            (6.0, 0.5, 1.5 * np.tan(np.deg2rad(10)))],
            "waves": [(3.2, 0.0, .08), (2.4, 0.0, .07168), (2.8, 0.0, 0.0)],
            "hurdles": [(2.4, 0.0, .08), (2.7, 0.0, 0.0)],
            "stepping_stones": [(2.3, -.075, .10), (2.6, 0.0, 0.0)],
        }
        for lane in complex_lanes(self.cfg.terrain):
            for x, offset, height in expected.get(lane.kind, []):
                with self.subTest(lane=lane.kind, x=x, offset=offset):
                    self.move(robot, x, lane.y + offset)
                    self.assertAlmostEqual(robot.ground_height(), height, places=6)
                    self.assertIn(lane.label, robot.terrain_region())
            for x, y in ((0, lane.y), (11, lane.y), (6, lane.y + 1.25)):
                self.move(robot, x, y)
                self.assertAlmostEqual(robot.ground_height(), 0.0, places=6)
            self.move(robot, 0, lane.y)
            self.assertIn("APPROACH", robot.terrain_region())
        block = self.model.geom("terrain_ground_blocks_0_2")
        self.move(robot, 2.2, 13.5)
        self.assertAlmostEqual(robot.ground_height(), 2 * block.size[2], places=6)
        self.move(robot, 2.4, 13.5)
        self.assertAlmostEqual(robot.ground_height(), 0.0, places=6)

    def test_heightfield_boundaries_join_flat_ground_without_holes(self):
        robot = self.robot()
        for lane in complex_lanes(self.cfg.terrain)[:3]:
            for x, y in ((2, lane.y), (10, lane.y), (6, lane.y - 1), (6, lane.y + 1),
                         (1.999, lane.y), (10.001, lane.y), (6, lane.y - 1.001), (6, lane.y + 1.001)):
                self.move(robot, x, y)
                self.assertAlmostEqual(robot.ground_height(), 0.0, places=6)

    def test_obstacles_preserve_vertical_faces_and_generate_contacts(self):
        robot = self.robot()
        self.move(robot, 2.4 - .06 - .001, 16.5)
        self.assertAlmostEqual(robot.ground_height(), 0.0, places=6)
        self.move(robot, 2.4 - .06 + .001, 16.5)
        self.assertAlmostEqual(robot.ground_height(), .08, places=6)
        for lane in complex_lanes(self.cfg.terrain):
            x = {"blocks": 2.2, "hurdles": 2.4, "stepping_stones": 2.3}.get(lane.kind, 6.0)
            self.move(robot, x, lane.y)
            height = robot.ground_height()
            robot.data.qpos[robot.joint_qpos_addresses] = self.cfg.initial_state.default_joint_position
            self.move(robot, x, lane.y, height + .10)
            self.assertTrue(robot.torso_touches_floor(), lane.kind)
            # Put one toe onto the actual obstacle, even when a narrow hurdle lies
            # between the four nominal foot positions beneath the base.
            toe_id = min(robot.toe_body_ids)
            offset = robot.data.xpos[toe_id] - robot.data.xpos[robot.base_body_id]
            self.move(robot, x - offset[0], lane.y - offset[1], height - offset[2] + .02)
            self.assertGreater(robot.feet_in_contact(), 0, lane.kind)

    def test_seeded_blocks_and_height_scaling(self):
        repeated = load_model(self.cfg)
        changed = load_model(replace(self.cfg, terrain=replace(self.cfg.terrain, seed=42)))
        reordered = load_model(replace(self.cfg, terrain=replace(
            self.cfg.terrain, complex_sections=("blocks",))))
        names = [f"terrain_ground_blocks_{col}_{row}" for col in range(20) for row in range(5)]
        heights = lambda model: np.array([2 * model.geom(name).size[2] for name in names])
        np.testing.assert_array_equal(heights(self.model), heights(repeated))
        np.testing.assert_array_equal(heights(self.model), heights(reordered))
        self.assertFalse(np.array_equal(heights(self.model), heights(changed)))
        self.assertTrue(np.all((heights(self.model) >= .03) & (heights(self.model) <= .12)))
        cfg = replace(self.cfg, terrain=replace(self.cfg.terrain, complex_scale=2.0))
        robot = self.robot(load_model(cfg), cfg)
        original = self.robot()
        for x, y in ((6, 4.5), (6, 7.5), (2.4, 10.5), (2.2, 13.5), (2.4, 16.5), (2.3, 19.425)):
            self.move(original, x, y)
            self.move(robot, x, y)
            self.assertAlmostEqual(robot.ground_height(), 2 * original.ground_height(), places=6)

    def test_optional_sections_validation_and_other_modes(self):
        for changes in ({"complex_sections": ["unknown"]}, {"complex_sections": ["ramps", "ramps"]},
                        {"complex_sections": "ramps"}, {"complex_scale": float("nan")},
                        {"complex_scale": float("inf")}, {"complex_scale": 0}, {"complex_scale": 2.1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                TerrainConfig(**changes)
        self.assertEqual(len(complex_lanes(self.cfg.terrain)), len(COMPLEX_TERRAIN_KINDS))
        for kind in ("flat", "three_level", "stairs"):
            config = replace(self.cfg, terrain=replace(self.cfg.terrain, kind=kind))
            self.assertEqual(complex_lanes(config.terrain), ())
            self.assertEqual(load_model(config).nhfield, int(kind == "three_level"))
        config = replace(self.cfg, terrain=replace(self.cfg.terrain, complex_sections=()))
        self.assertEqual(load_model(config).nhfield, 1)

    def test_every_approach_supports_pd_init_and_history_policy(self):
        for lane in complex_lanes(self.cfg.terrain):
            config = replace(self.cfg, initial_state=replace(
                self.cfg.initial_state, base_position=(0, lane.y, .13)))
            runner = Sim2SimRunner(config, ZeroPolicy(12), BehaviorSupervisor(config.supervisor))
            with self.subTest(lane=lane.kind), redirect_stdout(io.StringIO()):
                result = runner.run((0, 0, 0), init_request_source=lambda: True,
                                    duration=6, realtime=False, print_interval=100)
                self.assertEqual(result.termination_reason, "duration")
                self.assertEqual(runner.supervisor.mode.value, "stand")
                self.assertTrue(np.isfinite(runner.data.qpos).all())
                self.assertGreater(runner.robot.base_height(), .25)


if __name__ == "__main__":
    unittest.main()
