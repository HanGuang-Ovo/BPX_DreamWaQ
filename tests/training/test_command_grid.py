"""Grid-Adaptive 指令课程调度器单元测试：分布、扩展门控与保护逻辑。"""

import sys
import unittest

import torch

sys.path.insert(0, "src")

from bpx_dreamwaq.tasks.mdp.command_grid import GridAdaptiveCommandScheduler


def make_scheduler(**overrides):
    params = dict(
        vx_resolution=0.5,
        yaw_resolution=0.5,
        initial_vx_range=(-0.5, 0.5),
        initial_yaw_range=(-0.5, 0.5),
        max_vx_range=(-1.0, 1.0),
        max_yaw_range=(-1.0, 1.0),
        success_threshold_vx=0.25,
        success_threshold_yaw=0.5,
        # 测试默认放宽成功率 EMA 门控（一次成功即达标），便于隔离
        # min_samples / 双轴条件等其他门控逻辑；可按用例覆盖。
        success_ratio_threshold=0.5,
        min_samples=3,
        ema_alpha=0.5,
        exploration_prob=0.1,
        continuous_sampling=False,
        low_speed_sampling_prob=0.0,
    )
    params.update(overrides)
    return GridAdaptiveCommandScheduler(**params)


class TestGridAdaptiveCommandScheduler(unittest.TestCase):
    def test_initial_grid_and_distribution(self):
        grid = make_scheduler(continuous_sampling=True)
        # 网格：vx ∈ {-1,-0.5,0,0.5,1}，yaw 同 → 5×5=25 单元。
        self.assertEqual(grid.num_cells, 25)
        # 初始激活 3×3=9。
        self.assertEqual(grid.num_activated, 9)
        # 概率归一化。
        self.assertAlmostEqual(grid.probs.sum().item(), 1.0, places=5)
        self.assertTrue((grid.probs >= 0).all())
        # 初始范围。
        self.assertAlmostEqual(grid.active_vx_range, 0.75, places=5)
        self.assertAlmostEqual(grid.active_yaw_range, 0.75, places=5)

    def test_joint_sampling_never_outside_max(self):
        grid = make_scheduler()
        cell_idx, vx, wz = grid.sample(2000)
        self.assertTrue((vx.abs() <= 1.0 + 1e-6).all())
        self.assertTrue((wz.abs() <= 1.0 + 1e-6).all())
        # 联合分布：采样值成对来自同一格中心（无抖动时完全离散）。
        vx_idx = torch.div(cell_idx, grid.num_yaw, rounding_mode="floor")
        self.assertTrue(torch.allclose(vx, grid.vx_centers[vx_idx]))

    def test_exploration_prob_samples_hard_cells(self):
        grid = make_scheduler(exploration_prob=0.5)
        _, vx, wz = grid.sample(2000)
        # 高探索比例下应出现初始范围外的单元。
        outside = (vx.abs() > 0.6) | (wz.abs() > 0.6)
        self.assertTrue(outside.any().item())

    def test_expansion_gated_by_min_samples(self):
        grid = make_scheduler(min_samples=3)
        # 初始区边缘格 (vx=0.5, yaw=0.5)：其 (1.0, ·) 与 (·, 1.0) 邻域未激活。
        cell = torch.tensor([3 * 5 + 3])
        err = torch.tensor([0.0])
        # 只记录 2 次（< min_samples）：不扩展。
        grid.record_interval(cell, err, err)
        grid.record_interval(cell, err, err)
        self.assertEqual(grid.num_activated, 9)
        self.assertEqual(grid.samples[cell.item()].item(), 2)
        # 第 3 次达标后扩展未激活邻域（2 个）。
        grid.record_interval(cell, err, err)
        self.assertEqual(grid.num_activated, 11)

    def test_no_expansion_on_failure(self):
        grid = make_scheduler(min_samples=1)
        cell = torch.tensor([12])
        big_vx = torch.tensor([5.0])  # 远超阈值
        ok_yaw = torch.tensor([0.0])
        for _ in range(5):
            grid.record_interval(cell, big_vx, ok_yaw)
        self.assertEqual(grid.num_activated, 9)

    def test_success_requires_both_axes(self):
        grid = make_scheduler(min_samples=1)
        cell = torch.tensor([3 * 5 + 3])  # 边缘格
        grid.record_interval(cell, torch.tensor([0.0]), torch.tensor([2.0]))  # yaw 不达标
        self.assertEqual(grid.num_activated, 9)
        grid.record_interval(cell, torch.tensor([0.0]), torch.tensor([0.0]))  # 双轴达标
        self.assertEqual(grid.num_activated, 11)

    def test_expansion_bounded_by_max_range(self):
        # 初始即覆盖全部范围 → 扩展不越界（数量不再增长）。
        grid = make_scheduler(
            initial_vx_range=(-1.0, 1.0), initial_yaw_range=(-1.0, 1.0), min_samples=1
        )
        self.assertEqual(grid.num_activated, 25)
        center = torch.tensor([12])
        zero = torch.tensor([0.0])
        grid.record_interval(center, zero, zero)
        self.assertEqual(grid.num_activated, 25)
        self.assertAlmostEqual(grid.probs.sum().item(), 1.0, places=5)

    def test_activation_monotonic(self):
        grid = make_scheduler(min_samples=1)
        center = torch.tensor([12])
        zero = torch.tensor([0.0])
        grid.record_interval(center, zero, zero)
        activated_before = grid.activated.clone()
        # 之后即使表现变差，已激活单元不会被撤销。
        grid.record_interval(torch.tensor([12, 16]), torch.tensor([9.0, 9.0]), torch.tensor([9.0, 9.0]))
        self.assertTrue((grid.activated & ~activated_before).sum().item() >= 0)
        self.assertTrue(grid.activated.sum().item() >= activated_before.sum().item())

    def test_ema_updates(self):
        grid = make_scheduler(ema_alpha=0.5)
        cell = torch.tensor([12])
        grid.record_interval(cell, torch.tensor([0.2]), torch.tensor([0.1]))
        self.assertAlmostEqual(grid.err_vx_ema[12].item(), 0.5 * 0.0 + 0.5 * 0.2)
        self.assertAlmostEqual(grid.err_yaw_ema[12].item(), 0.5 * 0.1)

    def test_empty_record_is_noop(self):
        grid = make_scheduler()
        count = grid.expansion_count
        grid.record_interval(torch.tensor([], dtype=torch.long), torch.tensor([]), torch.tensor([]))
        self.assertEqual(grid.expansion_count, count)

    def test_graduated_cell_does_not_reexpand(self):
        grid = make_scheduler(min_samples=1)
        edge = torch.tensor([3 * 5 + 3])
        zero = torch.tensor([0.0])
        grid.record_interval(edge, zero, zero)
        self.assertEqual(grid.num_activated, 11)
        self.assertEqual(grid.expansion_count, 2)
        # 同一格再次成功不会重复扩张（graduated 标记）。
        grid.record_interval(edge, zero, zero)
        self.assertEqual(grid.num_activated, 11)
        self.assertEqual(grid.expansion_count, 2)

    def test_boundary_cell_expands_only_valid_neighbors(self):
        grid = make_scheduler(min_samples=1)
        zero = torch.tensor([0.0])
        edge = torch.tensor([3 * 5 + 3])  # (0.5, 0.5)
        grid.record_interval(edge, zero, zero)
        self.assertEqual(grid.num_activated, 11)
        # 新激活的 (1.0, 0.5) 位于最大范围边界，只向外扩展 (1.0, ±1.0)。
        boundary = torch.tensor([4 * 5 + 3])
        grid.record_interval(boundary, zero, zero)
        self.assertEqual(grid.num_activated, 13)

    def test_probability_floor_over_activated(self):
        grid = make_scheduler(exploration_prob=0.0)
        _, vx, wz = grid.sample(500)
        # 零探索时全部采样落在初始激活范围 ±0.5 内。
        self.assertTrue(((vx.abs() <= 0.5) & (wz.abs() <= 0.5)).all())

    def test_continuous_commands_cover_cells_and_keep_statistical_ownership(self):
        grid = make_scheduler(continuous_sampling=True, initial_vx_range=(-1., 1.),
                              initial_yaw_range=(-1., 1.))
        cell, vx, yaw = grid.sample(30000, generator=torch.Generator().manual_seed(7))
        vx_index = cell // grid.num_yaw
        yaw_index = cell % grid.num_yaw
        self.assertTrue((vx.abs() <= 1.).all())
        self.assertTrue((yaw.abs() <= 1.).all())
        self.assertTrue(((vx - grid.vx_centers[vx_index]).abs() <= .25).all())
        self.assertTrue(((yaw - grid.yaw_centers[yaw_index]).abs() <= .25).all())
        # 中心格直接覆盖正负低速，不再只靠相邻中心点泛化。
        center = (vx_index == 2) & (yaw_index == 2)
        self.assertTrue(((vx[center] > .15) & (vx[center] < .25)).any())
        self.assertTrue(((vx[center] < -.15) & (vx[center] > -.25)).any())
        # 边界按截断区间均匀采样，而非先越界再 clamp 造成端点堆积。
        edge = vx[vx_index == 4]
        self.assertAlmostEqual(edge.mean().item(), .875, delta=.005)
        self.assertLess((edge == 1.).float().mean().item(), .002)
        counts = grid.samples.clone()
        grid.record_interval(cell[:100], torch.zeros(100), torch.zeros(100))
        touched = torch.bincount(cell[:100], minlength=grid.num_cells) > 0
        torch.testing.assert_close(grid.samples - counts, touched.long())

    def test_non_aligned_asymmetric_limits_have_no_sampling_gaps_at_edges(self):
        grid = make_scheduler(continuous_sampling=True, max_vx_range=(-.9, 1.1),
                              max_yaw_range=(.1, .9))
        _, vx, yaw = grid.sample(10000, generator=torch.Generator().manual_seed(8))
        self.assertTrue(((vx >= -.9) & (vx <= 1.1)).all())
        self.assertTrue(((yaw >= .1) & (yaw <= .9)).all())
        self.assertLess(vx.min().item(), -.85)
        self.assertGreater(vx.max().item(), 1.05)
        self.assertLess(yaw.min().item(), .15)
        self.assertGreater(yaw.max().item(), .85)

    def test_low_speed_mixture_matches_exported_probabilities(self):
        grid = make_scheduler(continuous_sampling=True, low_speed_sampling_prob=.3,
                              initial_vx_range=(-1., 1.), initial_yaw_range=(-1., 1.))
        cell, vx, yaw, low = grid.sample_with_low_speed_mask(
            40000, generator=torch.Generator().manual_seed(9))
        self.assertAlmostEqual(low.float().mean().item(), .3, delta=.01)
        self.assertTrue((cell[low] == 12).all())
        self.assertTrue((vx[low].abs() <= .25).all())
        self.assertTrue((yaw[low].abs() <= .25).all())
        self.assertTrue((vx[low] > .15).any())
        self.assertTrue((vx[low] < -.15).any())
        frequencies = torch.bincount(cell, minlength=grid.num_cells).float() / cell.numel()
        torch.testing.assert_close(frequencies, grid.probs, atol=.01, rtol=0.)
        self.assertAlmostEqual(grid.probs[12].item(), .3 + .7 / 25, places=5)
        self.assertAlmostEqual(grid.active_vx_range, 1.)
        self.assertAlmostEqual(grid.active_yaw_range, 1.)

    def test_low_speed_branch_does_not_bypass_activation_or_cell_bounds(self):
        grid = make_scheduler(continuous_sampling=True, low_speed_sampling_prob=1.,
                              low_speed_vx_range=(-.6, .6), low_speed_yaw_range=(-.4, .4))
        cell, vx, yaw = grid.sample(10000, generator=torch.Generator().manual_seed(10))
        self.assertTrue(grid.activated.flatten()[cell].all())
        self.assertTrue((vx.abs() <= .6).all())
        self.assertTrue((yaw.abs() <= .4).all())
        self.assertTrue(((vx - grid.vx_centers[cell // grid.num_yaw]).abs() <= .25).all())
        self.assertTrue(((yaw - grid.yaw_centers[cell % grid.num_yaw]).abs() <= .25).all())

        grid = make_scheduler(continuous_sampling=True, low_speed_sampling_prob=1.,
                              initial_vx_range=(.5, .6), initial_yaw_range=(.5, .6))
        _, _, _, low = grid.sample_with_low_speed_mask(100)
        self.assertFalse(low.any())  # 零速区域尚未激活时，退回常规课程。

    def test_seed_controls_cell_branch_and_continuous_draws(self):
        grid = make_scheduler(continuous_sampling=True, low_speed_sampling_prob=.3)
        first = grid.sample_with_low_speed_mask(100, generator=torch.Generator().manual_seed(42))
        second = grid.sample_with_low_speed_mask(100, generator=torch.Generator().manual_seed(42))
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0., atol=0.)

    def test_invalid_sampling_configuration_is_rejected(self):
        for args in [dict(low_speed_sampling_prob=-.1), dict(low_speed_sampling_prob=1.1),
                     dict(command_jitter=.3), dict(command_jitter=float('nan')),
                     dict(low_speed_vx_range=(.2, -.2)), dict(max_vx_range=(float('-inf'), 1.))]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                make_scheduler(**args)


if __name__ == "__main__":
    unittest.main()
