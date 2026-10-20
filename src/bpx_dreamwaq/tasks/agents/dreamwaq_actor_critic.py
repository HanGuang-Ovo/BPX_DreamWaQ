"""DreamWaQ Actor-Critic：CENet 与非对称 Actor-Critic 的连接。

设计要点（依据 DreamWaQ 论文 arXiv:2301.10602 与任务要求的核验结论）：

- Actor 输入 ``concat(policy_obs, vel_hat, latent)``：
  - ``policy_obs``: 当前实机可获取的 45 维单帧观测（不含真实线速度）；
  - ``vel_hat``: CENet 估计的机体线速度（3 维）；
  - ``latent``: CENet VAE 隐变量特征（默认 16 维，Actor 使用确定性均值 μ）。
- Critic 输入特权观测（真实线速度、高度扫描等），与 Actor 完全隔离。
- 默认 ``detach_cenet_from_actor=True``：Actor 的 PPO 梯度不回传至 CENet，
  CENet 仅由辅助损失（速度监督 + VAE）更新。论文未明确该连接方式，此处
  选择与任务伪代码一致的 detach 实现，并保持可配置。
- VAE 隐变量送入 Actor 时默认使用确定性均值 μ（``latent_actor_mode="mean"``），
  避免 PPO 更新阶段重采样隐变量导致新旧策略概率比计算不一致；
  可配置为 ``"sample"``，此时采样结果在 rollout 期间保存并在更新时复用。
- PPO 更新时 Actor 输入使用 rollout 期间保存的 CENet 特征
  （``act_with_cenet_features``），保证与采样时刻完全一致。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from rsl_rl.modules import ActorCritic

from ..obs_contract import DEFAULT_HISTORY_LENGTH, history_obs_dim, newest_frame_slices, single_obs_dim
from .cenet import CENet


class DreamWaQActorCritic(ActorCritic):
    """带 CENet 状态估计的非对称 Actor-Critic（兼容 rsl-rl 2.3.3 接口）。"""

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        # ---- CENet 配置 ----
        history_length: int = DEFAULT_HISTORY_LENGTH,
        cenet_encoder_hidden_dims: Tuple[int, ...] = (256, 128),
        cenet_vel_head_hidden_dims: Tuple[int, ...] = (128,),
        cenet_vae_head_hidden_dims: Tuple[int, ...] = (128,),
        cenet_decoder_hidden_dims: Tuple[int, ...] = (128, 128),
        cenet_latent_dim: int = 16,
        cenet_vel_dim: int = 3,
        cenet_activation: str = "elu",
        # ---- 梯度与特征策略 ----
        detach_cenet_from_actor: bool = True,
        latent_actor_mode: str = "mean",  # "mean" | "sample"
        # ---- 基类参数 ----
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if latent_actor_mode not in ("mean", "sample"):
            raise ValueError(f"latent_actor_mode 必须为 'mean' 或 'sample'，当前为 {latent_actor_mode}")
        self.history_length = history_length
        self.latent_actor_mode = latent_actor_mode
        self.detach_cenet_from_actor = detach_cenet_from_actor

        self.num_history_obs = history_obs_dim(history_length)
        # 单帧历史 = 下一时刻观测目标维度。
        self.num_next_obs = num_actor_obs
        self.cenet_vel_dim = cenet_vel_dim
        self.cenet_latent_dim = cenet_latent_dim
        # Actor 实际输入 = 单帧观测 + 估计速度 + 隐变量特征。
        num_actor_input = num_actor_obs + cenet_vel_dim + cenet_latent_dim
        super().__init__(
            num_actor_input,
            num_critic_obs,
            num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )

        self.cenet = CENet(
            num_history_obs=self.num_history_obs,
            num_next_obs=self.num_next_obs,
            history_length=history_length,
            encoder_hidden_dims=tuple(cenet_encoder_hidden_dims),
            vel_head_hidden_dims=tuple(cenet_vel_head_hidden_dims),
            vae_head_hidden_dims=tuple(cenet_vae_head_hidden_dims),
            decoder_hidden_dims=tuple(cenet_decoder_hidden_dims),
            latent_dim=cenet_latent_dim,
            vel_dim=cenet_vel_dim,
            activation=cenet_activation,
        )
        # 最近一次送入 Actor 的 CENet 特征 (B, vel_dim + latent_dim)。
        self._cenet_features: torch.Tensor | None = None

    """
    CENet 特征接口
    """

    @torch.no_grad()
    def encode_cenet(self, history_obs: torch.Tensor) -> torch.Tensor:
        """从历史观测计算送入 Actor 的特征 ``[vel_hat, latent]``。

        默认情况下 rollout 与更新使用确定性 μ，并按配置决定是否 detach。
        本方法在 rollout 阶段调用（外层通常处于 ``torch.inference_mode()``）。
        """
        if history_obs.shape[-1] != self.num_history_obs:
            raise ValueError(
                f"历史观测维度 {history_obs.shape[-1]} 与网络配置 {self.num_history_obs} 不一致"
            )
        if self.latent_actor_mode == "mean":
            vel_hat, latent = self.cenet.deterministic_latent(history_obs)
        else:
            vel_hat, _, _, latent = self.cenet(history_obs)
        if self.detach_cenet_from_actor:
            vel_hat, latent = vel_hat.detach(), latent.detach()
        self._cenet_features = torch.cat([vel_hat, latent], dim=-1)
        return self._cenet_features

    def set_cenet_features(self, cenet_features: torch.Tensor):
        """直接设置 Actor 使用的 CENet 特征（更新阶段复用 rollout 保存值）。"""
        if cenet_features.shape[-1] != self.cenet_vel_dim + self.cenet_latent_dim:
            raise ValueError("cenet_features 维度与 vel_dim + latent_dim 不一致")
        self._cenet_features = cenet_features.detach()

    """
    Actor-Critic 接口（覆盖 rsl_rl.ActorCritic）
    """

    def _actor_input(self, observations: torch.Tensor) -> torch.Tensor:
        if self._cenet_features is None or self._cenet_features.shape[0] != observations.shape[0]:
            raise RuntimeError(
                "必须在调用 act/update_distribution 前设置 CENet 特征"
                "（调用 encode_cenet 或 set_cenet_features）"
            )
        return torch.cat([observations, self._cenet_features], dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(self._actor_input(observations))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"未知的噪声类型: {self.noise_std_type}")
        self.distribution = torch.distributions.Normal(mean, std)

    def act(self, observations, history_obs: torch.Tensor | None = None, **kwargs):
        """采样动作。提供 ``history_obs`` 时先编码 CENet，否则复用已设置的特征。"""
        if history_obs is not None:
            self.encode_cenet(history_obs)
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_with_cenet_features(self, observations, cenet_features: torch.Tensor, **kwargs):
        """使用保存的 CENet 特征采样（PPO 更新阶段保证概率比一致性）。"""
        self.set_cenet_features(cenet_features)
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations, history_obs: torch.Tensor | None = None):
        """确定性推理：给定历史观测时输出 CENet + Actor 的确定性动作。"""
        if history_obs is not None:
            self.encode_cenet(history_obs)
        return self.actor(self._actor_input(observations))


class DeployableDreamWaQPolicy(nn.Module):
    """导出/部署用的融合策略：450 维历史观测 → 12 维动作。

    输入为展平的历史观测（term-major、项内从最旧到最新），内部：
    提取最新一帧 → CENet（确定性 μ）→ 拼接单帧观测、估计速度与隐变量 → Actor。
    部署时不需要 Critic、真实速度、高度扫描或 VAE Decoder。
    """

    def __init__(self, policy: DreamWaQActorCritic):
        super().__init__()
        self.policy = policy
        self._slices = newest_frame_slices(policy.history_length)
        self.num_single_obs = single_obs_dim()

    def extract_current_frame(self, history_obs: torch.Tensor) -> torch.Tensor:
        """从展平历史中提取每个观测项的最新一帧，拼接为单帧观测。"""
        frames = [history_obs[..., start:end] for start, end in self._slices]
        return torch.cat(frames, dim=-1)

    def forward(self, history_obs: torch.Tensor) -> torch.Tensor:
        current_obs = self.extract_current_frame(history_obs)
        vel_hat, latent = self.policy.cenet.deterministic_latent(history_obs)
        actor_input = torch.cat([current_obs, vel_hat, latent], dim=-1)
        return self.policy.actor(actor_input)
