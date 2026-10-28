"""CENet 单元测试：维度、前向/反向、辅助损失与梯度（测试 1/2）。"""

import sys
import unittest

import torch

sys.path.insert(0, "src")

from bpx_dreamwaq.tasks.agents.cenet import CENet
from bpx_dreamwaq.tasks.obs_contract import DEFAULT_HISTORY_LENGTH, history_obs_dim, single_obs_dim


class TestCENet(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.history_dim = history_obs_dim(DEFAULT_HISTORY_LENGTH)
        self.obs_dim = single_obs_dim()
        self.net = CENet(num_history_obs=self.history_dim, num_next_obs=self.obs_dim)

    def test_output_dimensions(self):
        batch = 8
        vel_hat, mu, logvar, z = self.net(torch.randn(batch, self.history_dim))
        self.assertEqual(vel_hat.shape, (batch, 3))
        self.assertEqual(mu.shape, (batch, self.net.latent_dim))
        self.assertEqual(logvar.shape, (batch, self.net.latent_dim))
        self.assertEqual(z.shape, (batch, self.net.latent_dim))
        self.assertEqual(self.net.latent_dim, 16)

    def test_deterministic_latent(self):
        history = torch.randn(4, self.history_dim)
        vel_a, mu_a = self.net.deterministic_latent(history)
        vel_b, mu_b = self.net.deterministic_latent(history)
        self.assertTrue(torch.allclose(vel_a, vel_b))
        self.assertTrue(torch.allclose(mu_a, mu_b))

    def test_decoder_output_dim(self):
        z = torch.randn(4, self.net.latent_dim)
        out = self.net.decode(z, torch.randn(4, 3))
        self.assertEqual(out.shape, (4, self.obs_dim))

    def test_prediction_trains_velocity_head(self):
        losses = self.net.compute_aux_losses(torch.randn(8, self.history_dim),
                                             torch.randn(8, self.obs_dim), torch.randn(8, 3))
        losses["pred_loss"].backward()
        self.assertGreater(self.net.vel_head[0].weight.grad.abs().sum().item(), 0)
        self.assertEqual(self.net.decoder[0].in_features, self.net.latent_dim + 3)

    def test_aux_losses_finite_and_grads_flow(self):
        history = torch.randn(16, self.history_dim)
        next_obs = torch.randn(16, self.obs_dim)
        true_vel = torch.randn(16, 3)
        losses = self.net.compute_aux_losses(history, next_obs, true_vel)
        for key in ("vel_loss", "pred_loss", "kl_loss"):
            self.assertTrue(torch.isfinite(losses[key]).item(), f"{key} 非有限值")
        total = losses["vel_loss"] + losses["pred_loss"] + losses["kl_loss"]
        total.backward()
        for name, param in self.net.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} 未获得梯度")
            self.assertTrue(torch.isfinite(param.grad).all(), f"{name} 梯度含 NaN/Inf")

    def test_done_mask_excludes_invalid_transitions(self):
        history = torch.randn(6, self.history_dim)
        next_obs = torch.randn(6, self.obs_dim)
        true_vel = torch.randn(6, 3)
        # 前 3 个转移有效，后 3 个为 episode 边界（done）。
        mask = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        good_next = next_obs.clone()
        torch.manual_seed(123)  # 固定 VAE 重参数化噪声，使两次前向可比
        losses_good = self.net.compute_aux_losses(history, good_next, true_vel, mask)
        # 修改被屏蔽样本的下一观测目标不应影响预测损失。
        bad_next = next_obs.clone()
        bad_next[3:] += 100.0
        torch.manual_seed(123)
        losses_bad = self.net.compute_aux_losses(history, bad_next, true_vel, mask)
        self.assertTrue(torch.allclose(losses_good["pred_loss"], losses_bad["pred_loss"]))
        # 全部屏蔽时损失有限（分母保护）。
        losses_all_masked = self.net.compute_aux_losses(history, next_obs, true_vel, torch.zeros(6))
        self.assertTrue(torch.isfinite(losses_all_masked["pred_loss"]).item())

    def test_independent_masks_and_original_reductions(self):
        history = torch.randn(4, self.history_dim)
        target = torch.randn(4, self.obs_dim)
        velocity = torch.randn(4, 3)
        torch.manual_seed(12)
        vel, mu, logvar, z = self.net(history)
        prediction = self.net.decode(z, vel)
        vel_each = (vel - velocity).square().sum(-1)
        pred_each = (prediction - target).square().mean(-1)
        kl_each = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1)
        for mask in (torch.ones(4), torch.tensor([1, 0, 1, 0])):
            torch.manual_seed(12)
            losses = self.net.compute_aux_losses(history, target, velocity, mask)
            torch.testing.assert_close(losses["vel_loss"], vel_each.mean())
            torch.testing.assert_close(losses["kl_loss"], kl_each.mean())
            torch.testing.assert_close(losses["pred_loss"], (pred_each * mask).sum() / mask.sum())
        # A real terminal observation can be included when the caller knows it is continuous.
        torch.manual_seed(12)
        terminal = self.net.compute_aux_losses(history, target, velocity, torch.ones(4, 1))
        torch.testing.assert_close(terminal["pred_loss"], pred_each.mean())

        current = torch.tensor([1, 0, 1, 0])
        torch.manual_seed(12)
        masked = self.net.compute_aux_losses(history, target, velocity, torch.ones(4), current)
        torch.testing.assert_close(masked["vel_loss"], (vel_each * current).sum() / current.sum())
        torch.testing.assert_close(masked["kl_loss"], (kl_each * current).sum() / current.sum())
        torch.testing.assert_close(masked["pred_loss"], (pred_each * current).sum() / current.sum())

    def test_mask_shapes_and_zero_prediction_gradient(self):
        for batch in (1, 3, 7):
            for column in (False, True):
                self.net.zero_grad(set_to_none=True)
                mask = torch.zeros(batch, 1) if column else torch.zeros(batch)
                losses = self.net.compute_aux_losses(
                    torch.randn(batch, self.history_dim), torch.randn(batch, self.obs_dim),
                    torch.randn(batch, 3), mask,
                )
                self.assertEqual(losses["pred_loss"].item(), 0.0)
                losses["pred_loss"].backward()
                for param in self.net.parameters():
                    if param.grad is not None:
                        self.assertTrue(torch.isfinite(param.grad).all())
                        self.assertEqual(param.grad.count_nonzero().item(), 0)
                self.assertTrue(all(torch.isfinite(losses[k]) for k in ("vel_loss", "pred_loss", "kl_loss")))
        with self.assertRaises(ValueError):
            self.net.compute_aux_losses(torch.randn(3, self.history_dim), torch.randn(3, self.obs_dim),
                                        torch.randn(3, 3), torch.ones(3, 3))

    def test_latent_dim_configurable(self):
        net = CENet(num_history_obs=32, num_next_obs=8, latent_dim=5, vel_dim=2)
        vel_hat, mu, logvar, z = net(torch.randn(3, 32))
        self.assertEqual(vel_hat.shape, (3, 2))
        self.assertEqual(mu.shape, (3, 5))

    def test_logvar_clamped(self):
        net = CENet(num_history_obs=16, num_next_obs=8)
        history = torch.zeros(2, 16) + 50.0  # 极端输入
        _, _, logvar, _ = net(history)
        self.assertTrue((logvar <= 10.0).all() and (logvar >= -10.0).all())


if __name__ == "__main__":
    unittest.main()
