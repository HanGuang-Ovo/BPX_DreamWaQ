"""地形课程单元测试：升级/降级判定、互斥、保护逻辑与环境接线。"""

import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, "src")

from bpx_dreamwaq.tasks.mdp.curriculums import compute_terrain_level_moves, terrain_levels_vel


class TestComputeTerrainLevelMoves(unittest.TestCase):
    def setUp(self):
        self.terrain_length = 8.0
        self.episode_length_s = 20.0

    def test_move_up_on_far_travel(self):
        # 走了 5 m > 8 m / 2：升级。
        distance = torch.tensor([5.0, 1.0])
        command = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        up, down = compute_terrain_level_moves(distance, command, self.terrain_length, self.episode_length_s)
        self.assertTrue(up[0].item())
        self.assertFalse(down[0].item())

    def test_move_down_on_short_travel(self):
        # 指令 1.0 m/s × 20 s 预期 20 m，一半 10 m；只走 1 m：降级。
        distance = torch.tensor([5.0, 1.0])
        command = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        up, down = compute_terrain_level_moves(distance, command, self.terrain_length, self.episode_length_s)
        self.assertTrue(down[1].item())
        self.assertFalse(up[1].item())

    def test_zero_command_never_downgrades(self):
        # 静止指令：走 0.1 m 也不降级（低速保护）。
        distance = torch.tensor([0.1])
        command = torch.tensor([[0.0, 0.0]])
        up, down = compute_terrain_level_moves(distance, command, self.terrain_length, self.episode_length_s)
        self.assertFalse(up.item())
        self.assertFalse(down.item())

    def test_mutually_exclusive(self):
        # 构造同时满足两个原始条件的输入：升级必须压制降级。
        distance = torch.tensor([5.0])
        command = torch.tensor([[1.0, 0.0]])
        up, down = compute_terrain_level_moves(distance, command, self.terrain_length, self.episode_length_s)
        self.assertFalse(up.item() and down.item())
        # 大指令时 5 m 同时 < 10 m（降级原始条件成立）且 > 4 m（升级成立）。
        self.assertTrue(up.item())
        self.assertFalse(down.item())

    def test_upgrade_only_mode(self):
        distance = torch.tensor([0.1])
        command = torch.tensor([[1.0, 0.0]])
        up, down = compute_terrain_level_moves(
            distance, command, self.terrain_length, self.episode_length_s, down_distance_ratio=0.0
        )
        self.assertFalse(up.item())
        self.assertFalse(down.item())

    def test_batch_consistency(self):
        distance = torch.tensor([9.0, 5.0, 0.1, 0.1])
        command = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        up, down = compute_terrain_level_moves(distance, command, self.terrain_length, self.episode_length_s)
        self.assertEqual(up.tolist(), [True, True, False, False])
        self.assertEqual(down.tolist(), [False, False, True, False])


class _FakeTerrain:
    def __init__(self, num_envs=4, num_levels=10):
        self.device = "cpu"
        self.terrain_levels = torch.zeros(num_envs, dtype=torch.long)
        self.max_terrain_level = num_levels
        self.env_origins = torch.zeros(num_envs, 3)
        self.terrain_origins = torch.zeros(num_levels, 2, 3)
        self.cfg = SimpleNamespace(terrain_generator=SimpleNamespace(size=(8.0, 8.0)))
        self.calls = []

    def update_env_origins(self, env_ids, move_up, move_down):
        self.calls.append((env_ids.clone(), move_up.clone(), move_down.clone()))
        self.terrain_levels[env_ids] += 1 * move_up.long() - 1 * move_down.long()
        self.terrain_levels = torch.clamp(self.terrain_levels, 0)


class _FakeRewardManager:
    def __init__(self, num_envs):
        self._episode_sums = {"track": torch.ones(num_envs) * 10.0}


class _FakeScene:
    def __init__(self, num_envs, terrain):
        self._entities = {
            "robot": SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(num_envs, 3))),
            "terrain": terrain,
        }
        self.terrain = terrain
        self.env_origins = torch.zeros(num_envs, 3)

    def __getitem__(self, name):
        return self._entities[name]


def make_fake_env(num_envs=4):
    terrain = _FakeTerrain(num_envs)
    env = SimpleNamespace(
        scene=_FakeScene(num_envs, terrain),
        command_manager=SimpleNamespace(
            get_command=lambda name: torch.tensor([[1.0, 0.0, 0.0]] * num_envs)
        ),
        reward_manager=_FakeRewardManager(num_envs),
        episode_length_buf=torch.full((num_envs,), 200, dtype=torch.long),
        num_envs=num_envs,
        max_episode_length_s=20.0,
    )
    return env, terrain


class TestTerrainLevelsEnvIntegration(unittest.TestCase):
    def test_updates_and_stats(self):
        env, terrain = make_fake_env(4)
        # 机器人都走到 5 m：升级条件 5 > 4 成立。
        env.scene["robot"].data.root_pos_w[:, :2] = 5.0
        result = terrain_levels_vel(env, torch.tensor([0, 1]), asset_cfg=SimpleNamespace(name="robot"))
        self.assertEqual(terrain.terrain_levels[:2].tolist(), [1, 1])
        self.assertEqual(terrain.terrain_levels[2:].tolist(), [0, 0])
        self.assertIn("mean_level", result)
        self.assertEqual(result["num_move_up"], 2)
        stats = env.terrain_curriculum_stats
        self.assertEqual(stats["move_up_count"], 2)
        # count 按等级索引：两个回合都发生在等级 0。
        self.assertEqual(stats["count"][0].item(), 2.0)
        self.assertEqual(stats["count"][1].item(), 0.0)
        # 逐等级回合奖励：等级 0 的两个回合各 10.0。
        self.assertAlmostEqual(stats["reward_sum"][0].item(), 20.0)

    def test_stats_persist_across_calls(self):
        env, terrain = make_fake_env(4)
        env.scene["robot"].data.root_pos_w[:, :2] = 0.0
        # 走 0 m 且指令 1 m/s：降级（但等级 0 不再下降）。
        terrain_levels_vel(env, torch.tensor([2]), asset_cfg=SimpleNamespace(name="robot"))
        stats = env.terrain_curriculum_stats
        self.assertEqual(stats["move_down_count"], 1)
        terrain_levels_vel(env, torch.tensor([3]), asset_cfg=SimpleNamespace(name="robot"))
        self.assertEqual(env.terrain_curriculum_stats["move_down_count"], 2)

    def test_initial_reset_skipped(self):
        # 环境构建期间的首次内部 reset：episode_length_buf 为 0，课程不更新。
        env, terrain = make_fake_env(4)
        env.scene["robot"].data.root_pos_w[:, :2] = 50.0  # 默认初始位置远离原点
        env.episode_length_buf[:] = 0
        result = terrain_levels_vel(env, torch.tensor([0, 1]), asset_cfg=SimpleNamespace(name="robot"))
        self.assertEqual(result["num_move_up"], 0)
        self.assertEqual(terrain.terrain_levels.tolist(), [0, 0, 0, 0])

    def test_slice_env_ids(self):
        env, terrain = make_fake_env(4)
        result = terrain_levels_vel(env, slice(None), asset_cfg=SimpleNamespace(name="robot"))
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
