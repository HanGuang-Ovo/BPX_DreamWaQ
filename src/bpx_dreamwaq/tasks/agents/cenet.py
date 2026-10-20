"""DreamWaQ 风格的 CENet（Context-Aided Estimator Network）。

参考 Nahrendra et al., "DreamWaQ: Learning Robust Quadrupedal Locomotion With
Implicit Terrain Imagination via Deep Reinforcement Learning" (ICRA 2023,
arXiv:2301.10602)：

- 单个共享编码器读取历史本体观测 ``o^H_t``；
- 速度估计头输出当前机体线速度估计 ``v̂_t``（3 维，机体坐标系）；
- β-VAE 头输出潜在分布 ``(μ, log σ²)``，经重参数化得到隐变量 ``z_t``；
- Decoder 仅根据 ``z_t`` 预测下一时刻本体观测 ``o_{t+1}``（不是重构当前观测）；
- 辅助损失 ``L_CE = L_est + L_VAE``，其中
  ``L_est = MSE(v̂_t, v_t)``，``L_VAE = MSE(ô_{t+1}, o_{t+1}) + β·KL``。

本模块只依赖 torch，可在不启动 Isaac Sim 的情况下单元测试。
网络规模、潜在维度、历史长度全部可配置。
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..obs_contract import BASE_LIN_VEL_DIM


class CENet(nn.Module):
    """历史本体观测编码器：共享编码 + 速度估计头 + VAE 隐变量头 + 未来观测 Decoder。

    前向输出::

        vel_hat, mu, logvar, z = cenet(history_obs)

    - ``vel_hat``: (B, vel_dim) 机体线速度估计；
    - ``mu``/``logvar``: (B, latent_dim) 后验分布参数；
    - ``z``: (B, latent_dim) 重参数化采样得到的隐变量（训练时使用）。
    """

    def __init__(
        self,
        num_history_obs: int,
        num_next_obs: int,
        history_length: int = 10,
        encoder_hidden_dims: Tuple[int, ...] = (256, 128),
        vel_head_hidden_dims: Tuple[int, ...] = (128,),
        vae_head_hidden_dims: Tuple[int, ...] = (128,),
        decoder_hidden_dims: Tuple[int, ...] = (128, 128),
        latent_dim: int = 16,
        vel_dim: int = BASE_LIN_VEL_DIM,
        activation: str = "elu",
        logvar_bounds: Tuple[float, float] = (-10.0, 10.0),
    ):
        super().__init__()
        if num_history_obs <= 0 or num_next_obs <= 0:
            raise ValueError("num_history_obs 与 num_next_obs 必须为正")
        if latent_dim <= 0:
            raise ValueError("latent_dim 必须为正")
        self.num_history_obs = num_history_obs
        self.num_next_obs = num_next_obs
        self.history_length = history_length
        self.latent_dim = latent_dim
        self.vel_dim = vel_dim
        self.logvar_bounds = logvar_bounds

        act = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}.get(activation, nn.ELU)

        def mlp(dims_in, hidden, dims_out):
            layers: list[nn.Module] = []
            dim = dims_in
            for h in hidden:
                layers += [nn.Linear(dim, h), act()]
                dim = h
            layers.append(nn.Linear(dim, dims_out))
            return nn.Sequential(*layers)

        shared_out = encoder_hidden_dims[-1] if encoder_hidden_dims else num_history_obs
        # 共享编码器：速度估计与 VAE 共用同一特征。
        self.encoder = mlp(num_history_obs, encoder_hidden_dims, shared_out)
        # 速度估计头。
        self.vel_head = mlp(shared_out, vel_head_hidden_dims, vel_dim)
        # VAE 编码头：输出 2*latent_dim（均值与 log variance）。
        self.vae_head = mlp(shared_out, vae_head_hidden_dims, 2 * latent_dim)
        # Decoder：仅根据隐变量预测下一时刻本体观测。
        self.decoder = mlp(latent_dim, decoder_hidden_dims, num_next_obs)

    def forward(self, history_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(history_obs)
        vel_hat = self.vel_head(features)
        vae_out = self.vae_head(features)
        mu, logvar = torch.chunk(vae_out, 2, dim=-1)
        # 限制 logvar 防止数值发散（不影响梯度方向，仅裁剪极端值）。
        logvar = torch.clamp(logvar, self.logvar_bounds[0], self.logvar_bounds[1])
        z = self.reparameterize(mu, logvar)
        return vel_hat, mu, logvar, z

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = μ + σ·ε, ε ~ N(0, I)。"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def deterministic_latent(self, history_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (vel_hat, mu)：用于 Actor 输入和部署推理的确定性特征。

        部署与 PPO 概率比复现均使用确定性均值 μ，避免随机重采样导致
        新旧策略概率比计算不一致。
        """
        features = self.encoder(history_obs)
        vel_hat = self.vel_head(features)
        vae_out = self.vae_head(features)
        mu, logvar = torch.chunk(vae_out, dim=-1, chunks=2)
        logvar = torch.clamp(logvar, self.logvar_bounds[0], self.logvar_bounds[1])
        return vel_hat, mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """根据隐变量预测下一时刻本体观测。"""
        return self.decoder(z)

    def compute_aux_losses(
        self,
        history_obs: torch.Tensor,
        next_obs: torch.Tensor,
        true_vel: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """计算 CENet 辅助损失。

        Args:
            history_obs: (B, num_history_obs) 当前时刻历史本体观测。
            next_obs: (B, num_next_obs) 下一时刻本体观测（VAE Decoder 目标）。
            true_vel: (B, vel_dim) 仿真器真实机体线速度（与估计同坐标系、同时刻）。
            valid_mask: (B,) 或 (B,1) 的 0/1 掩码；0 表示该转移无效
                （episode 终止或 reset 后的下一观测），其预测损失与 KL 损失被屏蔽。

        Returns:
            字典：``vel_loss``（MSE）、``pred_loss``（未来观测预测 MSE）、
            ``kl_loss``（KL 散度）、``vel_error``（未加权平均误差，用于日志）。
        """
        if valid_mask is None:
            valid_mask = torch.ones(history_obs.shape[0], 1, device=history_obs.device)
        else:
            valid_mask = valid_mask.to(history_obs.device).reshape(-1, 1)
        mask_sum = valid_mask.sum().clamp_min(1.0)

        vel_hat, mu, logvar, z = self.forward(history_obs)

        # 速度监督：真实速度仅在训练中使用（标签，detached）。
        vel_loss = ((vel_hat - true_vel.detach()) ** 2 * valid_mask).sum() / mask_sum

        # 未来观测预测损失：屏蔽 episode 边界。
        next_obs_hat = self.decode(z)
        pred_loss = (((next_obs_hat - next_obs.detach()) ** 2).mean(dim=-1, keepdim=True) * valid_mask).sum() / mask_sum

        # KL(q(z|o) || N(0,I))，同样按有效样本归一化。
        kl_per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1, keepdim=True)
        kl_loss = (kl_per_sample * valid_mask).sum() / mask_sum

        vel_error = (vel_hat - true_vel).abs().detach()

        return {
            "vel_loss": vel_loss,
            "pred_loss": pred_loss,
            "kl_loss": kl_loss,
            "vel_error": vel_error,
        }
