"""Run: python -m unittest discover -s tests/sim2sim -p 'test_terrain.py' -v"""
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from bpx_dreamwaq.sim2sim.config import load_config
from bpx_dreamwaq.sim2sim.robot import BpxMujocoRobot
from bpx_dreamwaq.sim2sim import terrain
from bpx_dreamwaq.sim2sim.terrain import add_course_labels, course_heights, load_model


class TerrainIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(Path(__file__).resolve().parents[2] / 'configs/sim2sim/bpx_terrain_standing.toml')
        self.model = load_model(self.cfg)
        self.data = mujoco.MjData(self.model)
        self.robot = BpxMujocoRobot(self.model, self.data, self.cfg)
        self.robot.reset()

    def move(self, x, y, z):
        self.data.qpos[self.robot.base_qpos_address:self.robot.base_qpos_address + 3] = (x, y, z)
        mujoco.mj_forward(self.model, self.data)

    def test_reproducibility_limits_and_flat_edges(self):
        heights = course_heights(self.cfg.terrain.seed)
        np.testing.assert_array_equal(heights, course_heights(self.cfg.terrain.seed))
        self.assertFalse(np.array_equal(heights, course_heights(self.cfg.terrain.seed + 1)))
        self.assertLessEqual(np.abs(heights).max(), .04)
        self.assertLessEqual(np.abs(heights[:, 200:360]).max(), .02)
        np.testing.assert_array_equal(heights[:, :201], 0)
        for edge in (heights[0], heights[-1], heights[:, 0], heights[:, -1]):
            np.testing.assert_array_equal(edge, 0)
        mild = np.std(heights[60:180, 220:340])
        moderate = np.std(heights[60:180, 380:500])
        self.assertGreater(moderate, mild * 1.5)

    def test_normalization_preserves_metric_heights_when_difficulty_changes(self):
        for mild, moderate in ((.02, .04), (.03, .06)):
            with patch.object(terrain, 'MILD_AMPLITUDE', mild), patch.object(terrain, 'MODERATE_AMPLITUDE', moderate):
                model = load_model(self.cfg)
                heights = course_heights(self.cfg.terrain.seed)
            hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, 'three_level_course')
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self.cfg.floor_geom_name)
            start = model.hfield_adr[hid]
            encoded = model.hfield_data[start:start + heights.size]
            self.assertTrue(np.all((encoded >= 0) & (encoded <= 1)))
            decoded = encoded.reshape(heights.shape) * model.hfield_size[hid, 2] + model.geom_pos[gid, 2]
            np.testing.assert_allclose(decoded, heights, atol=1e-8)
            robot = BpxMujocoRobot(model, mujoco.MjData(model), self.cfg)
            robot.reset()
            for index in (np.argmin(heights), np.argmax(heights)):
                row, col = np.unravel_index(index, heights.shape)
                robot.data.qpos[robot.base_qpos_address:robot.base_qpos_address + 3] = (-3 + col*.025, -3 + row*.025, .4)
                mujoco.mj_forward(model, robot.data)
                self.assertAlmostEqual(robot.ground_height(), heights[row, col], places=6)

    def test_ground_ray_respects_valleys_and_clearance(self):
        heights = course_heights(self.cfg.terrain.seed)
        row, col = np.unravel_index(np.argmin(heights), heights.shape)
        self.move(-3 + col * .025, -3 + row * .025, .4)
        self.assertLess(self.robot.ground_height(), -.005)
        self.assertAlmostEqual(self.robot.ground_height(), heights[row, col], places=6)
        self.assertAlmostEqual(self.robot.base_height(), .4 - heights[row, col], places=6)
        for x, y in ((0, 0), (2, 0), (4, 2), (11, 0), (12, 0), (4, 4), (-4, 0)):
            self.move(x, y, .4)
            self.assertAlmostEqual(self.robot.ground_height(), 0, places=6)

    def test_contacts_on_field_and_surrounding_ground(self):
        for x, y in ((0, 0), (4, 0), (8, 0), (12, 0), (4, 4)):
            self.robot.reset()
            self.move(x, y, .32)
            self.assertGreater(self.robot.feet_in_contact(), 0, (x, y))
            self.move(x, y, .10)
            self.assertTrue(self.robot.torso_touches_floor(), (x, y))

    def test_robot_and_policy_interface_unchanged(self):
        flat_cfg = replace(self.cfg, terrain=replace(self.cfg.terrain, kind='flat'))
        flat_model = load_model(flat_cfg)
        flat_robot = BpxMujocoRobot(flat_model, mujoco.MjData(flat_model), flat_cfg)
        flat_robot.reset()
        for field in ('body_mass', 'body_inertia', 'jnt_range', 'actuator_ctrlrange'):
            np.testing.assert_array_equal(getattr(self.model, field), getattr(flat_model, field))
        np.testing.assert_array_equal(self.robot.observation(np.zeros(3), np.zeros(12)),
                                      flat_robot.observation(np.zeros(3), np.zeros(12)))

    def test_labels_and_region(self):
        class Viewer:
            user_scn = mujoco.MjvScene(self.model, maxgeom=10)
            opt = mujoco.MjvOption()
            cam = mujoco.MjvCamera()
            def lock(self):
                return nullcontext()
        viewer = Viewer()
        add_course_labels(viewer, self.cfg)
        self.assertEqual(viewer.user_scn.ngeom, 3)
        self.assertIn('MILD | +/-2 cm', viewer.user_scn.geoms[1].label)
        self.assertIn('MODERATE | +/-4 cm', viewer.user_scn.geoms[2].label)
        for x, expected in ((0, 'LEVEL 0'), (2.2, 'TRANSITION'), (4, 'LEVEL 1'), (8, 'LEVEL 2')):
            self.move(x, 0, .4)
            self.assertIn(expected, self.robot.terrain_region())


if __name__ == '__main__':
    unittest.main()
