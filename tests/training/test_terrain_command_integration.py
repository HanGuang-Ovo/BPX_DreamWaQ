"""不启动 Kit，使用最小 manager 接口验证真实指令类的整回合统计和配置接线。"""

import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, "src")
from bpx_dreamwaq.tasks.mdp import curriculums

ROOT = Path(__file__).resolve().parents[2]


class _CommandTerm:
    """提供 Isaac Lab 的顺序：统计旧指令，再重采样；reset 先重采样。"""

    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env
        self.num_envs, self.device = env.num_envs, env.device
        self.metrics = {}
        self.time_left = torch.zeros(env.num_envs)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        extras = {k: float(v[ids].mean()) for k, v in self.metrics.items()}
        for value in self.metrics.values():
            value[ids] = 0
        self._resample_command(ids)
        self.time_left[ids] = self.cfg.resampling_time_range[0]
        return extras

    def compute(self, dt):
        self._update_metrics()
        self.time_left -= dt
        ids = (self.time_left <= 0).nonzero().flatten()
        if ids.numel():
            self._resample_command(ids)
            self.time_left[ids] = self.cfg.resampling_time_range[0]
        self._update_command()


class _UniformVelocityCommand(_CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.vel_command_b = torch.zeros(env.num_envs, 3)
        self.sample_count = 0

    @property
    def command(self):
        return self.vel_command_b

    def _resample_command(self, ids):
        self.sample_count += 1
        self.vel_command_b[ids, 0] = 0.5 if self.sample_count % 2 else -0.5

    def _update_metrics(self):
        pass

    def _update_command(self):
        pass


def _configclass(cls):
    cls.__init__ = lambda self, **kwargs: self.__dict__.update(kwargs)
    return cls


def _load_commands():
    modules = {name: ModuleType(name) for name in (
        "isaaclab", "isaaclab.managers", "isaaclab.utils", "isaaclab.envs",
        "isaaclab.envs.mdp", "isaaclab.envs.mdp.commands",
    )}
    modules["isaaclab.managers"].CommandTerm = _CommandTerm
    modules["isaaclab.managers"].CommandTermCfg = type("CommandTermCfg", (), {})
    modules["isaaclab.utils"].configclass = _configclass
    modules["isaaclab.envs.mdp.commands"].UniformVelocityCommand = _UniformVelocityCommand
    name = "bpx_dreamwaq.tasks.mdp._commands_integration_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "src/bpx_dreamwaq/tasks/mdp/commands.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


commands = _load_commands()


class _Scene(dict):
    pass


def make_env(enabled=True):
    scene = _Scene(robot=SimpleNamespace(data=SimpleNamespace(
        root_pos_w=torch.zeros(2, 3), root_lin_vel_b=torch.zeros(2, 3),
        root_ang_vel_b=torch.zeros(2, 3),
    )))
    scene.env_origins = torch.zeros(2, 3)
    return SimpleNamespace(
        cfg=SimpleNamespace(
            enable_terrain_curriculum=enabled, terrain_curriculum_min_command_speed=0.1,
            terrain_curriculum_lin_vel_tolerance=0.25, terrain_curriculum_yaw_tolerance=0.5,
        ), scene=scene, num_envs=2, device="cpu", step_dt=0.02,
        episode_length_buf=torch.zeros(2, dtype=torch.long),
    )


class TestTerrainCommandIntegration(unittest.TestCase):
    def generators(self, enabled=True):
        for cls in (commands.GridAdaptiveVelocityCommand, commands.TerrainAwareUniformVelocityCommand):
            env = make_env(enabled)
            cfg = commands.GridAdaptiveVelocityCommandCfg(
                asset_name="robot", resampling_time_range=(0.1, 0.1), rel_standing_envs=0.0,
            )
            term = cls(cfg, env)
            term.reset()
            yield env, term

    def test_both_generators_measure_executed_commands_across_resampling(self):
        for env, term in self.generators():
            with self.subTest(generator=type(term).__name__):
                expected = torch.zeros(2)
                data = env.scene["robot"].data
                for _ in range(100):
                    executed = term.command.clone()
                    data.root_lin_vel_b.copy_(executed)
                    data.root_ang_vel_b[:, 2] = executed[:, 2]
                    env.episode_length_buf += 1
                    term.compute(env.step_dt)
                    speed = torch.linalg.vector_norm(executed[:, :2], dim=1)
                    expected += torch.where(speed > 0.1, speed * env.step_dt, 0.)
                torch.testing.assert_close(term.terrain_metrics.command_distance, expected)
                torch.testing.assert_close(term.terrain_metrics.good_tracking_fraction, torch.ones(2))

    def test_both_generators_reset_only_requested_env_after_robot_teleport(self):
        for env, term in self.generators():
            with self.subTest(generator=type(term).__name__):
                term.vel_command_b[:] = torch.tensor([0.5, 0., 0.])
                env.scene["robot"].data.root_lin_vel_b[:, 0] = 0.5
                env.episode_length_buf[:] = 1
                term.compute(env.step_dt)
                other_distance = term.terrain_metrics.command_distance[1].clone()
                env.scene["robot"].data.root_pos_w[0, 0] = 100
                term.reset(torch.tensor([0]))
                env.episode_length_buf[0] = 0
                term.compute(env.step_dt)
                self.assertEqual(term.terrain_metrics.command_distance[0].item(), 0)
                self.assertEqual(term.terrain_metrics.max_excursion[0].item(), 0)
                torch.testing.assert_close(term.terrain_metrics.command_distance[1], other_distance)
                self.assertEqual(term.terrain_metrics.start_xy[0, 0].item(), 100)

    def test_disabled_terrain_curriculum_needs_no_tracker(self):
        for env, term in self.generators(enabled=False):
            self.assertIsNone(term.terrain_metrics)
            term.compute(env.step_dt)
            term.reset(torch.tensor([0]))

    def test_low_speed_branch_limits_all_axes_and_preserves_interval_ownership(self):
        env = make_env(enabled=False)
        cfg = commands.GridAdaptiveVelocityCommandCfg(
            asset_name="robot", resampling_time_range=(.1, .1), rel_standing_envs=0.,
            low_speed_sampling_prob=1.,
        )
        term = commands.GridAdaptiveVelocityCommand(cfg, env)
        samples = []
        for _ in range(100):
            term.reset()
            samples.append(term.command.clone())
            self.assertTrue((term.cell_idx == 12).all())
            self.assertTrue((term.command[:, 0].abs() <= .25).all())
            self.assertTrue((term.command[:, 1].abs() <= .1).all())
            self.assertTrue((term.command[:, 2].abs() <= .25).all())
        samples = torch.cat(samples)
        self.assertTrue((samples[:, 0] > .15).any())
        self.assertTrue((samples[:, 0] < -.15).any())
        # 指令区间结束后，连续低速的表现归中心格；不错误记入相邻边缘格。
        term.interval_steps[:] = 5
        term.interval_err_vx[:] = .1
        term.interval_err_yaw[:] = .2
        term.reset(torch.tensor([0]))
        self.assertEqual(term.grid.samples[12].item(), 1)
        self.assertEqual(term.grid.samples.sum().item(), 1)
        self.assertEqual(term.interval_steps[1].item(), 5)

    def test_regular_branch_keeps_lateral_range_and_standing_stays_exactly_zero(self):
        env = make_env(enabled=False)
        cfg = commands.GridAdaptiveVelocityCommandCfg(
            asset_name="robot", resampling_time_range=(.1, .1), rel_standing_envs=0.,
            low_speed_sampling_prob=0., low_speed_vy_range=(.6, .7),
        )
        term = commands.GridAdaptiveVelocityCommand(cfg, env)
        samples = []
        for _ in range(100):
            term.reset()
            samples.append(term.command.clone())
        vy = torch.cat(samples)[:, 1]
        self.assertTrue((vy.abs() > .25).any())
        self.assertTrue((vy.abs() <= .5).all())
        cfg.rel_standing_envs = 1.
        term.reset()
        term._update_command()
        torch.testing.assert_close(term.command, torch.zeros(2, 3))
        term._update_metrics()
        self.assertTrue((term.interval_steps == 0).all())

    def test_low_speed_lateral_range_is_intersected_with_task_limits(self):
        env = make_env(enabled=False)
        cfg = commands.GridAdaptiveVelocityCommandCfg(
            asset_name="robot", resampling_time_range=(.1, .1), rel_standing_envs=0.,
            low_speed_sampling_prob=1., lin_vel_y_range=(-.03, .04),
        )
        term = commands.GridAdaptiveVelocityCommand(cfg, env)
        for _ in range(30):
            term.reset()
            self.assertTrue(((term.command[:, 1] >= -.03) & (term.command[:, 1] <= .04)).all())
        cfg.lin_vel_y_range = (.2, .5)
        with self.assertRaises(ValueError):
            commands.GridAdaptiveVelocityCommand(cfg, env)


class TestEnvironmentConfiguration(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((ROOT / "src/bpx_dreamwaq/tasks/rough_env_cfg.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BPXRoughEnvCfg")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__")
        defaults = {}
        for n in cls.body:
            if isinstance(n, ast.AnnAssign):
                try:
                    defaults[n.target.id] = ast.literal_eval(n.value)
                except (ValueError, TypeError):
                    pass
        uniform_cfg = type("UniformVelocityCommandCfg", (SimpleNamespace,), {"Ranges": SimpleNamespace})
        context = dict(
            _make_bpx_terrain_generator=lambda _: SimpleNamespace(), CurrTerm=SimpleNamespace,
            bpx_curriculums=curriculums, bpx_commands=commands,
            lab_mdp=SimpleNamespace(UniformVelocityCommandCfg=uniform_cfg),
        )
        exec(compile(ast.Module(body=[init], type_ignores=[]), "rough_env_cfg.__post_init__", "exec"), context)
        self.configure = context["__post_init__"]
        self.cfg = SimpleNamespace(
            **defaults, scene=SimpleNamespace(
                terrain=SimpleNamespace(physics_material=object()),
                height_scanner=SimpleNamespace(), contact_forces=SimpleNamespace(),
            ), sim=SimpleNamespace(physx=SimpleNamespace()),
            commands=SimpleNamespace(), curriculum=SimpleNamespace(),
        )

    def test_default_grid_and_uniform_fallback_both_support_terrain_metrics(self):
        self.configure(self.cfg)
        self.assertIs(self.cfg.commands.base_velocity.class_type, commands.GridAdaptiveVelocityCommand)
        self.assertEqual(self.cfg.commands.base_velocity.rel_standing_envs, .1)
        self.cfg.enable_grid_adaptive_command_curriculum = False
        self.configure(self.cfg)
        self.assertIs(self.cfg.commands.base_velocity.class_type, commands.TerrainAwareUniformVelocityCommand)
        self.assertEqual(self.cfg.commands.base_velocity.rel_standing_envs, .1)
        self.assertIs(self.cfg.curriculum.terrain_levels.func, curriculums.terrain_levels_vel)
        self.assertNotIn("success_ratio", self.cfg.curriculum.terrain_levels.params)

    def test_upgrade_only_and_disabled_terrain_curriculum_are_wired(self):
        self.cfg.terrain_curriculum_mode = "upgrade_only"
        self.configure(self.cfg)
        self.assertFalse(self.cfg.curriculum.terrain_levels.params["allow_demotion"])
        self.cfg.enable_terrain_curriculum = False
        self.configure(self.cfg)
        self.assertIsNone(self.cfg.curriculum.terrain_levels)

    def test_continuous_low_speed_defaults_and_overrides_reach_command_term(self):
        self.configure(self.cfg)
        command = self.cfg.commands.base_velocity
        self.assertTrue(command.continuous_sampling)
        self.assertEqual(command.rel_standing_envs, .1)
        self.assertEqual(command.low_speed_sampling_prob, .3)
        self.assertEqual(command.low_speed_vx_range, (-.25, .25))
        self.assertEqual(command.low_speed_vy_range, (-.1, .1))
        self.assertEqual(command.low_speed_yaw_range, (-.25, .25))
        self.cfg.command_grid_continuous_sampling = False
        self.cfg.command_rel_standing_envs = .2
        self.cfg.command_low_speed_sampling_prob = .15
        self.cfg.command_low_speed_vx_range = (-.2, .2)
        self.cfg.command_low_speed_vy_range = (-.05, .05)
        self.cfg.command_low_speed_yaw_range = (-.1, .1)
        self.configure(self.cfg)
        command = self.cfg.commands.base_velocity
        self.assertFalse(command.continuous_sampling)
        self.assertEqual(command.rel_standing_envs, .2)
        self.assertEqual(command.low_speed_sampling_prob, .15)
        self.assertEqual(command.low_speed_vx_range, (-.2, .2))
        self.assertEqual(command.low_speed_vy_range, (-.05, .05))
        self.assertEqual(command.low_speed_yaw_range, (-.1, .1))
        self.cfg.enable_grid_adaptive_command_curriculum = False
        self.configure(self.cfg)
        self.assertEqual(self.cfg.commands.base_velocity.rel_standing_envs, .2)


if __name__ == "__main__":
    unittest.main()
