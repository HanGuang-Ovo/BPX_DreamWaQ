"""Verify real MuJoCo stair collision geometry, joint semantics and PD handover."""

from contextlib import nullcontext, redirect_stdout
import csv
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from bpx_dreamwaq.runtime.joints import POLICY_JOINT_NAMES
from bpx_dreamwaq.runtime.policy import ZeroPolicy
from bpx_dreamwaq.runtime.supervisor import BehaviorSupervisor
from bpx_dreamwaq.sim2sim.config import TerrainConfig, load_config
from bpx_dreamwaq.sim2sim.robot import BpxMujocoRobot
from bpx_dreamwaq.sim2sim.runner import Sim2SimRunner
from bpx_dreamwaq.sim2sim.terrain import add_course_labels, load_model, stair_lanes

CONFIG = Path(__file__).resolve().parents[2] / "configs/sim2sim/bpx_mixed.toml"


class StairTerrainTest(unittest.TestCase):
    def test_collision_heights_risers_and_contacts_for_every_level(self):
        for kind in ("stairs", "mixed"):
            cfg = load_config(CONFIG)
            cfg = replace(cfg, terrain=replace(cfg.terrain, kind=kind))
            model = load_model(cfg)
            data = mujoco.MjData(model)
            robot = BpxMujocoRobot(model, data, cfg)
            robot.reset()

            def move(x, y, z):
                data.qpos[robot.base_qpos_address:robot.base_qpos_address + 3] = (x, y, z)
                mujoco.mj_forward(model, data)

            self.assertAlmostEqual(robot.ground_height(), 0.)
            for lane in stair_lanes(cfg.terrain):
                for start, end, height in lane.segments:
                    move((start + end) / 2, lane.y, height + .4)
                    self.assertAlmostEqual(robot.ground_height(), height, places=6)
                    self.assertAlmostEqual(robot.base_height(), .4, places=6)
                    self.assertIn(f"STAIRS {lane.level}", robot.terrain_region())
                for x in (1., lane.segments[-1][1] + .5):
                    move(x, lane.y, .4)
                    self.assertAlmostEqual(robot.ground_height(), 0.)
                # A true vertical box riser: a millimetre on either side has a full step jump.
                move(2. - .001, lane.y, 1.)
                self.assertAlmostEqual(robot.ground_height(), 0.)
                move(2. + .001, lane.y, 1.)
                self.assertAlmostEqual(robot.ground_height(), lane.step_height)
                start, end, height = lane.segments[cfg.terrain.step_count - 1]
                data.qpos[robot.joint_qpos_addresses] = cfg.initial_state.default_joint_position
                move((start + end) / 2, lane.y, height + .32)
                self.assertEqual(robot.feet_in_contact(), 4)
                move((start + end) / 2, lane.y, height + .10)
                self.assertTrue(robot.torso_touches_floor())

    def test_course_labels_include_roughness_and_stair_difficulty(self):
        cfg = load_config(CONFIG)
        model = load_model(cfg)

        class Viewer:
            user_scn = mujoco.MjvScene(model, maxgeom=10)
            opt = mujoco.MjvOption()
            cam = mujoco.MjvCamera()

            def lock(self):
                return nullcontext()

        viewer = Viewer()
        add_course_labels(viewer, cfg)
        self.assertEqual(viewer.user_scn.ngeom, 7)
        for index, height in enumerate((3, 8, 10, 15), start=3):
            self.assertIn(f"{height} cm", viewer.user_scn.geoms[index].label)
        self.assertEqual(viewer.opt.geomgroup[5], 1)

    def test_custom_geometry_and_invalid_parameters(self):
        cfg = TerrainConfig(kind="stairs", step_heights=[.04, .10], step_count=3, step_width=.4)
        lanes = stair_lanes(cfg)
        self.assertEqual(len(lanes), 2)
        self.assertEqual(len(lanes[0].segments), 5)
        self.assertAlmostEqual(lanes[1].segments[2][2], .30)
        self.assertAlmostEqual(lanes[1].segments[-1][1], 4.8)
        for overrides in ({"kind": "bad"}, {"seed": -1}, {"step_heights": []},
                          {"step_heights": [.15, .03]}, {"step_heights": [float("nan")]},
                          {"step_width": 0}, {"stair_width": float("inf")},
                          {"step_count": 2.5}, {"step_count": 0}, {"platform_width": 100}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                TerrainConfig(**overrides)

    def test_one_hot_policy_actions_and_observations_use_named_joints(self):
        cfg = load_config(CONFIG)
        self.assertEqual(cfg.joint_names, POLICY_JOINT_NAMES)
        swapped = CONFIG.read_text().replace('"fl_hip_roll_joint"', '"SWAP"')
        swapped = swapped.replace('"fr_hip_roll_joint"', '"fl_hip_roll_joint"')
        swapped = swapped.replace('"SWAP"', '"fr_hip_roll_joint"')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "swapped.toml"
            path.write_text(swapped)
            with self.assertRaisesRegex(ValueError, "策略关节顺序"):
                load_config(path)
        model = load_model(cfg)
        robot = BpxMujocoRobot(model, mujoco.MjData(model), cfg)
        for index, name in enumerate(POLICY_JOINT_NAMES):
            robot.reset()
            robot.data.qpos[robot.joint_qpos_addresses] = robot.default_joint_position
            action = np.eye(12)[index] * .2
            target, _ = robot.apply_pd(action)
            expected = robot.default_joint_position.copy()
            expected[index] += .1
            np.testing.assert_allclose(target, expected)
            actuator = robot.actuator_ids[index]
            self.assertEqual(model.actuator_trnid[actuator, 0], model.joint(name).id)
            expected_ctrl = np.zeros(12)
            expected_ctrl[index] = 4.
            np.testing.assert_allclose(robot.data.ctrl[robot.actuator_ids], expected_ctrl, atol=1e-12)
            robot.data.qpos[model.joint(name).qposadr[0]] += .1
            robot.data.qvel[model.joint(name).dofadr[0]] = .2
            observation = robot.observation(np.zeros(3), action)
            np.testing.assert_allclose(observation[12:24], action * .5, atol=1e-7)
            np.testing.assert_allclose(observation[24:36], action, atol=1e-7)
            np.testing.assert_allclose(observation[36:48], action, atol=1e-7)


class ProneToPolicyTest(unittest.TestCase):
    def test_wait_for_button_init_then_policy_and_stick_release(self):
        for kind in ("mixed", "stairs"):
            cfg = load_config(CONFIG)
            cfg = replace(cfg, terrain=replace(cfg.terrain, kind=kind))
            calls = []
            runner = None

            def policy(observation):
                self.assertTrue(runner.init_controller.complete)
                self.assertEqual(observation.shape, (450,))
                calls.append(float(runner.data.time))
                return ZeroPolicy(12)(observation)

            runner = Sim2SimRunner(cfg, policy, BehaviorSupervisor(cfg.supervisor))
            # Emulate sticks already deflected while waiting and throughout Init.
            def stick():
                return (.3, .15, .2) if runner.data.time < 6.8 else (0., 0., 0.)

            with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                log = Path(tmp) / "run.csv"
                result = runner.run((0, 0, 0), command_source=stick,
                                    init_request_source=lambda: .5 <= runner.data.time < .52,
                                    duration=9, realtime=False, print_interval=100, log_path=log)
                with log.open() as stream:
                    rows = list(csv.DictReader(stream))
            self.assertEqual(result.termination_reason, "duration")
            self.assertGreaterEqual(calls[0], 4.98)
            modes = {row["behavior_mode"] for row in rows}
            self.assertEqual(modes, {"waiting_init", "init", "stand", "walk", "stopping"})
            self.assertEqual(rows[-1]["behavior_mode"], "stand")
            for row in rows:
                if row["behavior_mode"] in ("waiting_init", "init", "stand"):
                    self.assertEqual([float(row[f"command_{axis}"]) for axis in ("vx", "vy", "wz")], [0., 0., 0.])
            self.assertGreater(runner.robot.base_height(), .25)


if __name__ == "__main__":
    unittest.main()
