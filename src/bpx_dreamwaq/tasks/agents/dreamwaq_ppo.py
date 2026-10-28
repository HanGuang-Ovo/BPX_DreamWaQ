"""DreamWaQ PPO：辅助损失、扩展 Rollout 存储与训练 Runner。

在 rsl-rl 2.3.3 的 PPO 基础上做最小扩展：

1. ``DreamWaQRolloutStorage``：在标准 rollout 字段之外额外保存
   - ``history_observations``: CENet 输入（当前时刻历史本体观测）；
   - ``cenet_features``: rollout 时实际送入 Actor 的 CENet 特征
     （含 AdaBoot 混合后的速度），更新阶段原样复用以保证 PPO 概率比一致；
   - ``true_velocities``: 速度估计监督标签（仿真真实机体线速度）；
   - ``next_observations``: 下一时刻本体观测（VAE Decoder 目标）。
2. ``DreamWaQPPO``：每个 minibatch 在 PPO 损失上叠加
   ``lambda_vel * L_vel + lambda_pred * L_pred + beta_kl * L_kl``，
   CENet 与 PPO 使用独立 Adam、学习率和梯度裁剪；默认 PPO 不回传 CENet。
3. ``DreamWaQRunner``：适配 ``OnPolicyRunner``——解析自定义策略/算法类，
   在 rollout 中传递历史观测与真实速度标签，记录 CENet、AdaBoot、课程指标。

AdaBoot（论文公式 p = 1 - tanh(CV(R))，CV 为回合奖励变异系数）在此实现为
可选机制，支持四种模式：``error``（默认，按速度估计误差过渡，任务伪代码风格）、
``reward_cv``（论文公式）、``linear``（线性爬升）与 ``off``（关闭即 p=1，
始终使用估计速度）。默认关闭（基础 CENet + PPO 联合训练）。
"""

from __future__ import annotations

import math
import os
import statistics
import time
from collections import deque
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage

from .dreamwaq_actor_critic import DeployableDreamWaQPolicy, DreamWaQActorCritic


def _compute_reward_cv(rewards) -> float | None:
    """有限回合奖励的 std/|mean|；样本不足、非有限或均值近零时不可用。

    先缩放再求标准差，避免极大但有限的回合奖励导致中间结果溢出。
    不过滤坏样本，以免把无效统计误认为可信的 AdaBoot 输入。
    """
    values = [float(value) for value in rewards]
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        return None
    mean = statistics.mean(values)
    if abs(mean) <= 1e-6:
        return None
    scale = max(abs(value) for value in values)
    scaled_mean = abs(mean) / scale
    if scaled_mean == 0.0:
        return None
    cv = statistics.pstdev(value / scale for value in values) / scaled_mean
    return cv if math.isfinite(cv) else None


class DreamWaQRolloutStorage(RolloutStorage):
    """标准 rollout 存储 + CENet 辅助监督数据。

    所有扩展字段的时间索引与标准字段一致：第 t 行对应第 t 次环境步
    （t 时刻的观测、历史、真实速度、动作，以及执行动作后的 done 与
    下一时刻观测）。``dones[t]=1`` 时 ``next_observations[t]`` 是 reset 后
    的新 episode 初始观测，训练时必须用 done 掩码屏蔽。
    """

    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.history_observations = None
            self.cenet_features = None
            self.true_velocities = None
            self.next_observations = None
            self.use_estimated_velocity = None
            self.numerical_invalid = None

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        history_obs_shape,
        cenet_feature_shape,
        true_vel_shape,
        rnd_state_shape=None,
        device="cpu",
    ):
        super().__init__(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs_shape,
            privileged_obs_shape,
            actions_shape,
            rnd_state_shape,
            device,
        )
        self.history_observations = torch.zeros(num_transitions_per_env, num_envs, *history_obs_shape, device=device)
        self.cenet_features = torch.zeros(num_transitions_per_env, num_envs, *cenet_feature_shape, device=device)
        self.true_velocities = torch.zeros(num_transitions_per_env, num_envs, *true_vel_shape, device=device)
        self.use_estimated_velocity = torch.ones(num_transitions_per_env, num_envs, 1, dtype=torch.bool, device=device)
        self.next_observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=device)
        self.valid_transitions = torch.ones(num_transitions_per_env, num_envs, 1, dtype=torch.bool, device=device)

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")
        # 标准字段。
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)
        # CENet 扩展字段。
        self.history_observations[self.step].copy_(transition.history_observations)
        self.cenet_features[self.step].copy_(transition.cenet_features)
        self.true_velocities[self.step].copy_(transition.true_velocities)
        self.next_observations[self.step].copy_(transition.next_observations)
        if transition.use_estimated_velocity is not None:
            self.use_estimated_velocity[self.step].copy_(transition.use_estimated_velocity)
        self.valid_transitions[self.step] = True
        if transition.numerical_invalid is not None:
            affected = transition.numerical_invalid.reshape(-1, 1).bool().clone()
            self.valid_transitions[self.step] &= ~affected
            # Drop this episode's prefix still in the rollout, stopping at real episode boundaries.
            for t in reversed(range(self.step)):
                affected &= ~self.dones[t].bool()
                self.valid_transitions[t] &= ~affected
        self.step += 1

    def compute_returns(self, last_values, gamma, lam, normalize_advantage=True):
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.num_transitions_per_env)):
            valid = self.valid_transitions[step]
            next_values = last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
            # Avoid NaN * 0: invalid transitions are excluded before arithmetic.
            value = torch.where(valid, self.values[step], 0.0)
            reward = torch.where(valid, self.rewards[step], 0.0)
            continuing = valid & ~self.dones[step].bool()
            next_value = torch.where(continuing, next_values, 0.0)
            delta = reward + gamma * next_value - value
            advantage = delta + gamma * lam * torch.where(continuing, advantage, 0.0)
            self.returns[step] = advantage + value
            self.advantages[step] = advantage
        if normalize_advantage:
            good = self.advantages[self.valid_transitions]
            if good.numel():
                # Match the original sample std, but keep singleton batches finite.
                std = good.std() if good.numel() > 1 else good.new_zeros(())
                self.advantages[self.valid_transitions] = (good - good.mean()) / (std + 1e-8)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """与基类相同的抽样逻辑，额外产出 CENet 相关字段。"""
        if self.training_type != "rl":
            raise ValueError("该函数仅用于强化学习训练。")
        valid_indices = self.valid_transitions.flatten().nonzero(as_tuple=False).flatten()
        batch_size = valid_indices.numel()
        if batch_size == 0:
            return
        indices = valid_indices[torch.randperm(batch_size, device=self.device)]
        batches = torch.tensor_split(indices, min(num_mini_batches, batch_size))

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            privileged_observations = self.privileged_observations.flatten(0, 1)
        else:
            privileged_observations = observations
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        # CENet 字段。
        history_observations = self.history_observations.flatten(0, 1)
        cenet_features = self.cenet_features.flatten(0, 1)
        true_velocities = self.true_velocities.flatten(0, 1)
        next_observations = self.next_observations.flatten(0, 1)
        dones = self.dones.flatten(0, 1)

        for _ in range(num_epochs):
            for batch_idx in batches:
                yield (
                    observations[batch_idx],
                    privileged_observations[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                    (None, None),
                    None,
                    None,
                    history_observations[batch_idx],
                    cenet_features[batch_idx],
                    true_velocities[batch_idx],
                    next_observations[batch_idx],
                    dones[batch_idx],
                    self.use_estimated_velocity.flatten(0, 1)[batch_idx],
                )


class DreamWaQPPO(PPO):
    """带 CENet 辅助损失与可选 AdaBoot 引导的 PPO。"""

    def __init__(
        self,
        policy: DreamWaQActorCritic,
        # ---- CENet 辅助损失权重 ----
        lambda_vel: float = 1.0,
        lambda_pred: float = 1.0,
        beta_kl: float = 1.0e-3,
        cenet_learning_rate: float = 1.0e-3,
        cenet_max_grad_norm: float = 1.0,
        # ---- 梯度连接 ----
        ppo_grad_to_cenet: bool = False,
        # ---- AdaBoot ----
        adaboot_enabled: bool = False,
        adaboot_mode: str = "error",  # "error" | "reward_cv" | "linear"
        adaboot_error_threshold: float = 0.5,
        adaboot_ema_alpha: float = 0.9,
        adaboot_ramp_iterations: int = 500,
        adaboot_p_min: float = 0.0,
        # ---- 标准 PPO 参数 ----
        **kwargs,
    ):
        super().__init__(policy=policy, **kwargs)
        if cenet_learning_rate <= 0 or cenet_max_grad_norm <= 0:
            raise ValueError("CENet learning rate and gradient norm must be positive")
        self.cenet_parameters = list(policy.cenet.parameters())
        cenet_ids = {id(p) for p in self.cenet_parameters}
        self.ppo_parameters = [p for p in policy.parameters() if id(p) not in cenet_ids]
        self.optimizer = torch.optim.Adam(self.ppo_parameters, lr=self.learning_rate)
        self.cenet_optimizer = torch.optim.Adam(self.cenet_parameters, lr=cenet_learning_rate)
        self.cenet_max_grad_norm = cenet_max_grad_norm
        self.lambda_vel = lambda_vel
        self.lambda_pred = lambda_pred
        self.beta_kl = beta_kl
        self.ppo_grad_to_cenet = ppo_grad_to_cenet
        # The algorithm flag is authoritative for update-time gradient flow.
        if ppo_grad_to_cenet and policy.latent_actor_mode != "mean":
            raise ValueError("ppo_grad_to_cenet requires latent_actor_mode='mean'")

        self.adaboot_enabled = adaboot_enabled
        self.adaboot_mode = adaboot_mode
        self.adaboot_error_threshold = adaboot_error_threshold
        self.adaboot_ema_alpha = adaboot_ema_alpha
        self.adaboot_ramp_iterations = max(1, adaboot_ramp_iterations)
        self.adaboot_p_min = adaboot_p_min
        # 当前 AdaBoot 概率（使用估计速度的概率）。关闭时恒为 1。
        self.adaboot_p = adaboot_p_min if adaboot_enabled else 1.0
        self._adaboot_iteration = 0
        self._vel_error_ema: float | None = None
        self._reward_cv: float | None = None
        # rollout 期间估计速度使用比例统计。
        self._use_est_total = 0
        self._use_est_count = 0

        # 每个 PPO 迭代后刷新的 CENet 指标（由 Runner 写入日志）。
        self.cenet_metrics: Dict[str, float] = {}

    """
    初始化与 rollout
    """

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape
    ):
        policy: DreamWaQActorCritic = self.policy
        self.storage = DreamWaQRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            history_obs_shape=[policy.num_history_obs],
            cenet_feature_shape=[policy.cenet_vel_dim + policy.cenet_latent_dim],
            true_vel_shape=[policy.cenet_vel_dim],
            device=self.device,
        )
        # 使用扩展 Transition（基类构造了标准 Transition）。
        self.transition = DreamWaQRolloutStorage.Transition()

    def act(self, obs: torch.Tensor, history_obs: torch.Tensor, critic_obs: torch.Tensor, true_vel: torch.Tensor):
        """rollout 采样：编码 CENet、可选 AdaBoot 混合、保存一致性所需数据。"""
        if self.policy.is_recurrent:
            raise NotImplementedError("DreamWaQ 暂不支持循环策略")
        features = self.policy.encode_cenet(history_obs)
        vel_dim = self.policy.cenet_vel_dim
        if self.adaboot_enabled:
            # 每个环境独立抽样是否使用估计速度；真实速度仅用于构造 Actor 输入
            # （detached，不进入部署路径）。
            use_est = torch.rand(true_vel.shape[0], 1, device=true_vel.device) < self.adaboot_p
            vel_actor = torch.where(use_est, features[:, :vel_dim], self.policy.scale_velocity(true_vel.detach()))
            features_actor = torch.cat([vel_actor, features[:, vel_dim:]], dim=-1)
            self._use_est_total += use_est.numel()
            self._use_est_count += use_est.sum().item()
        else:
            use_est = torch.ones(true_vel.shape[0], 1, device=true_vel.device, dtype=torch.bool)
            features_actor = features
            self._use_est_total += features.shape[0]
            self._use_est_count += features.shape[0]
        # rollout 时实际送入 Actor 的特征被保存，更新阶段原样复用。
        self.policy.set_cenet_features(features_actor)
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        self.transition.history_observations = history_obs
        self.transition.cenet_features = features_actor
        self.transition.use_estimated_velocity = use_est
        self.transition.true_velocities = true_vel.detach()
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_obs: torch.Tensor):
        """记录转移；``next_obs`` 为执行动作后的下一时刻策略观测。"""
        self.transition.rewards = rewards.clone()
        invalid = infos.get("numerical_invalid", torch.zeros_like(dones, dtype=torch.bool)).to(self.device).bool()
        self.transition.numerical_invalid = invalid
        self.transition.dones = dones.bool() | invalid
        self.transition.rewards[invalid] = 0.0
        # 无限时域任务的 timeout 自举（沿用基类逻辑）。
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * (infos["time_outs"].to(self.device).bool() & ~invalid).unsqueeze(1), 1
            )
        self.transition.next_observations = next_obs
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    """
    AdaBoot 概率调度
    """

    def set_reward_cv(self, reward_cv: float | None):
        """由 Runner 提供：最近回合奖励的变异系数 std/|mean|（reward_cv 模式用）。"""
        self._reward_cv = (
            reward_cv if reward_cv is not None and math.isfinite(reward_cv) and reward_cv >= 0.0 else None
        )

    def _update_adaboot(self):
        if not self.adaboot_enabled:
            self.adaboot_p = 1.0
            return
        self._adaboot_iteration += 1
        if self.adaboot_mode == "error":
            # 任务伪代码风格：估计误差越大越依赖真实速度；误差趋零则完全使用估计。
            if self._vel_error_ema is not None:
                ratio = min(self._vel_error_ema / max(self.adaboot_error_threshold, 1e-6), 1.0)
                self.adaboot_p = 1.0 - ratio
            else:
                self.adaboot_p = self.adaboot_p_min
        elif self.adaboot_mode == "reward_cv":
            # 论文公式：p = 1 - tanh(CV(R))。
            if self._reward_cv is not None:
                self.adaboot_p = 1.0 - float(torch.tanh(torch.tensor(self._reward_cv)))
            else:
                self.adaboot_p = self.adaboot_p_min
        elif self.adaboot_mode == "linear":
            frac = min(self._adaboot_iteration / self.adaboot_ramp_iterations, 1.0)
            self.adaboot_p = self.adaboot_p_min + (1.0 - self.adaboot_p_min) * frac
        else:
            raise ValueError(f"未知 AdaBoot 模式: {self.adaboot_mode}")
        self.adaboot_p = float(min(max(self.adaboot_p, self.adaboot_p_min), 1.0))

    """
    更新
    """

    def update(self):  # noqa: C901
        self._update_adaboot()
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_vel_loss = 0.0
        mean_pred_loss = 0.0
        mean_kl_loss = 0.0
        mean_vel_mse = 0.0
        vel_err_axis = torch.zeros(self.policy.cenet_vel_dim, device=self.device)
        # 分速度档与隐变量诊断（见 update 末尾的 cenet_metrics）。
        vel_err_low_sum = 0.0
        vel_err_low_count = 0
        vel_err_high_sum = 0.0
        vel_err_high_count = 0
        latent_std_sum = 0.0

        num_updates = 0
        excluded_fraction = 1.0 - self.storage.valid_transitions.float().mean().item()
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            _hid_states_batch,
            _masks_batch,
            _rnd_state_batch,
            history_batch,
            cenet_features_batch,
            true_vel_batch,
            next_obs_batch,
            dones_batch,
            use_estimated_velocity_batch,
        ) in generator:
            num_updates += 1
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / ((advantages_batch.std() if advantages_batch.numel() > 1 else 0.0) + 1e-8)

            # Actor：默认复用 rollout 保存的 CENet 特征，保证新旧策略概率比
            # 在相同输入上计算。可选 ppo_grad_to_cenet=True 时改为带梯度重算。
            if self.ppo_grad_to_cenet:
                vel_hat, mu, _, _ = self.policy.cenet(self.policy.normalize_history(history_batch))
                vel_dim = self.policy.cenet_vel_dim
                vel_actor = torch.where(use_estimated_velocity_batch, vel_hat, self.policy.scale_velocity(true_vel_batch))
                cenet_features_batch = torch.cat([vel_actor, mu], dim=-1)
                self.policy.set_cenet_features(cenet_features_batch, detach=False)
                self.policy.update_distribution(obs_batch)
            else:
                self.policy.act_with_cenet_features(obs_batch, cenet_features_batch)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(critic_obs_batch)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # 自适应学习率（沿用基类 KL 规则）。
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # PPO 损失。
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Release PPO saved activations before building the auxiliary graph.
            # Even when PPO trains CENet, these are separate forward graphs:
            # accumulate both gradients before clipping or stepping either optimizer.
            self.optimizer.zero_grad(set_to_none=True)
            self.cenet_optimizer.zero_grad(set_to_none=True)
            ppo_loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )
            ppo_loss.backward()

            # CENet 辅助损失：done 转移的下一观测属于新 episode，必须屏蔽。
            # 数值异常转移已由 mini_batch_generator 排除；当前样本均有效。
            next_obs_mask = ~dones_batch.bool()
            aux = self.policy.cenet.compute_aux_losses(
                self.policy.normalize_history(history_batch),
                self.policy.obs_normalizer(next_obs_batch), self.policy.scale_velocity(true_vel_batch), next_obs_mask
            )
            vel_loss = aux["vel_loss"]
            pred_loss = aux["pred_loss"]
            kl_loss = aux["kl_loss"]
            aux_loss = self.lambda_vel * vel_loss + self.lambda_pred * pred_loss + self.beta_kl * kl_loss

            aux_loss.backward()
            nn.utils.clip_grad_norm_(self.ppo_parameters, self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.cenet_parameters, self.cenet_max_grad_norm)
            self.optimizer.step()
            self.cenet_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_vel_loss += vel_loss.item()
            mean_pred_loss += pred_loss.item()
            mean_kl_loss += kl_loss.item()
            # Keep diagnostic metrics and AdaBoot error thresholds in physical m/s.
            physical_vel_error = aux["vel_error"] / self.policy.velocity_scale
            mean_vel_mse += (physical_vel_error ** 2).mean().item()
            vel_err_axis += physical_vel_error.mean(dim=0)
            # 按 |vx 指令| 分桶的平均估计误差（观测为原始 m/s，指令位于第 7 维），
            # 验证“低速段估计相对误差更高”的假设；同时监控 latent μ 的平均
            # 标准差——若趋零说明隐变量对 Actor 不再携带信息。
            vx_cmd = obs_batch[:, 6].abs()
            low_mask = vx_cmd < 0.3
            if low_mask.any():
                vel_err_low_sum += physical_vel_error[low_mask].mean().item()
                vel_err_low_count += 1
            high_mask = vx_cmd >= 0.5
            if high_mask.any():
                vel_err_high_sum += physical_vel_error[high_mask].mean().item()
                vel_err_high_count += 1
            latent_std_sum += cenet_features_batch[:, self.policy.cenet_vel_dim:].std(dim=0).mean().item()

        # Entirely quarantined rollouts do not step either optimizer.
        num_updates = max(num_updates, 1)
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_vel_loss /= num_updates
        mean_pred_loss /= num_updates
        mean_kl_loss /= num_updates
        mean_vel_mse /= num_updates
        vel_err_axis /= num_updates

        # AdaBoot 误差 EMA（按 m/s 计的平均绝对误差）。
        vel_mae = float(vel_err_axis.mean())
        if excluded_fraction == 1.0:
            pass
        elif self._vel_error_ema is None:
            self._vel_error_ema = vel_mae
        else:
            self._vel_error_ema = (
                self.adaboot_ema_alpha * self._vel_error_ema + (1.0 - self.adaboot_ema_alpha) * vel_mae
            )

        self.storage.clear()

        self.cenet_metrics = {
            "numerical/excluded_transition_fraction": excluded_fraction,
            "cenet/learning_rate": self.cenet_optimizer.param_groups[0]["lr"],
            "cenet/velocity_mse": mean_vel_mse,
            "cenet/velocity_error_x": float(vel_err_axis[0]),
            "cenet/velocity_error_y": float(vel_err_axis[1]),
            "cenet/velocity_error_z": float(vel_err_axis[2]),
            "cenet/velocity_mae": vel_mae,
            "cenet/velocity_error_low_vx": vel_err_low_sum / max(vel_err_low_count, 1),
            "cenet/velocity_error_high_vx": vel_err_high_sum / max(vel_err_high_count, 1),
            "cenet/latent_mu_std": latent_std_sum / num_updates,
            "cenet/prediction_loss": mean_pred_loss,
            "cenet/kl_loss": mean_kl_loss,
            "cenet/vel_loss_weighted": mean_vel_loss,
            "cenet/total_auxiliary_loss": self.lambda_vel * mean_vel_loss
            + self.lambda_pred * mean_pred_loss
            + self.beta_kl * mean_kl_loss,
            "cenet/adaboot_p_boot": self.adaboot_p,
            "cenet/adaboot_est_fraction": (
                self._use_est_count / max(self._use_est_total, 1) if self._use_est_total > 0 else 1.0
            ),
        }
        self._use_est_total = 0
        self._use_est_count = 0

        return {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }


class DreamWaQRunner(OnPolicyRunner):
    """DreamWaQ 训练入口：OnPolicyRunner 的自定义策略/算法解析与数据流适配。

    相对 rsl_rl ``OnPolicyRunner`` 的修改：

    - 从本模块注册表解析 ``DreamWaQActorCritic`` / ``DreamWaQPPO`` 类名；
    - 训练类型固定为 RL（不使用蒸馏分支）；
    - rollout 循环携带 ``prop_history`` 观测组与真实速度标签
      （从 Critic 特权观测的固定切片读取，默认前 3 维）；
    - 迭代级写入 ``cenet/*``、``policy/*`` 与课程 ``curriculum/*`` 指标；
    - checkpoint 保存完整的策略（含 CENet）与优化器状态。
    """

    # class_name -> 类 的解析表（替代基类的 eval()）。
    POLICY_CLASSES = {"DreamWaQActorCritic": DreamWaQActorCritic}
    ALG_CLASSES = {"DreamWaQPPO": DreamWaQPPO}

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu", true_vel_slice=None):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        # 真实速度在 Critic 特权观测中的切片：优先取 train_cfg，缺省取前 3 维。
        if true_vel_slice is None:
            true_vel_slice = train_cfg.get("true_vel_slice", (0, 3))
        self.true_vel_slice = tuple(true_vel_slice)
        if len(self.true_vel_slice) != 2:
            raise ValueError("true_vel_slice 必须为 (start, end)")

        self.training_type = "rl"
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        if "critic" in extras["observations"]:
            self.privileged_obs_type = "critic"
            num_privileged_obs = extras["observations"]["critic"].shape[1]
        else:
            self.privileged_obs_type = None
            num_privileged_obs = num_obs
        # CENet 需要历史观测组。
        if "prop_history" not in extras["observations"]:
            raise ValueError("环境缺少 'prop_history' 观测组，DreamWaQ 训练需要历史本体观测")
        self.num_history_obs = extras["observations"]["prop_history"].shape[1]
        if self.true_vel_slice[1] > num_privileged_obs:
            raise ValueError("true_vel_slice 超出 Critic 特权观测维度")

        # 解析策略与算法类。
        policy_cfg = dict(self.policy_cfg)
        policy_cfg["observation_normalization"] = self.cfg["empirical_normalization"]
        policy_class_name = policy_cfg.pop("class_name")
        if policy_class_name not in self.POLICY_CLASSES:
            raise ValueError(f"未注册的策略类: {policy_class_name}（可用: {list(self.POLICY_CLASSES)}）")
        policy_class = self.POLICY_CLASSES[policy_class_name]
        self.policy: DreamWaQActorCritic = policy_class(
            num_obs, num_privileged_obs, self.env.num_actions, **policy_cfg
        ).to(self.device)

        alg_cfg = dict(self.alg_cfg)
        alg_class_name = alg_cfg.pop("class_name")
        if alg_class_name not in self.ALG_CLASSES:
            raise ValueError(f"未注册的算法类: {alg_class_name}（可用: {list(self.ALG_CLASSES)}）")
        alg_class = self.ALG_CLASSES[alg_class_name]
        self.alg: DreamWaQPPO = alg_class(self.policy, device=self.device, **alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        # Running moments live inside the policy and are exported with it.
        self.obs_normalizer = torch.nn.Identity().to(self.device)
        self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)

        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
        )

        self.disable_logs = False
        self.logger_type = "tensorboard"
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        # 单进程训练属性（基类 log() 需要；多 GPU 训练暂不支持）。
        self.gpu_world_size = 1
        self.gpu_global_rank = 0
        self.is_distributed = False

    def _extract_true_vel(self, privileged_obs: torch.Tensor) -> torch.Tensor:
        start, end = self.true_vel_slice
        return privileged_obs[:, start:end]

    def get_inference_policy(self, device=None):
        """Return deterministic CENet + Actor; input is raw prop_history (B, 450).

        Never reuse rollout AdaBoot features or the base runner's single-frame
        inference path. Normalization is already inside the returned module.
        """
        self.eval_mode()
        if device is not None:
            self.policy.to(device)
        return DeployableDreamWaQPolicy(self.policy).eval()

    def load_for_finetuning(self, path: str):
        """Load model and running moments into a fresh runner, keeping its optimizer.

        Fine-tuning starts iteration numbering and AdaBoot state from this run's
        configuration. Use load() for an optimizer-preserving resume instead.
        """
        if self.current_learning_iteration != 0 or self.alg.optimizer.state or self.alg.cenet_optimizer.state:
            raise RuntimeError("微调权重必须加载到尚未训练的新 Runner")
        infos = self.load(path, load_optimizer=False)
        self.current_learning_iteration = 0
        return infos

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        fixed = "obs_normalizer.scale" in checkpoint["model_state_dict"]
        if fixed != self.policy.fixed_observation_scaling:
            raise ValueError("checkpoint 固定缩放模式与当前配置不一致；旧权重请关闭 fixed_observation_scaling，新模式请从头训练")
        decoder_input = checkpoint["model_state_dict"]["cenet.decoder.0.weight"].shape[1]
        if decoder_input != self.policy.cenet.decoder[0].in_features:
            raise ValueError("checkpoint 解码器结构不同；旧权重需设置 cenet_decoder_use_velocity=False，新结构请从头训练")
        aux_state = (checkpoint.get("infos") or {}).get("cenet_optimizer_state_dict")
        if load_optimizer and aux_state is None:
            raise ValueError("旧 checkpoint 没有独立 CENet 优化器；请使用权重微调模式或从头训练")
        infos = super().load(path, load_optimizer=load_optimizer)
        if load_optimizer:
            self.alg.cenet_optimizer.load_state_dict(aux_state)
            self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
        return infos

    def save(self, path: str, infos=None):
        infos = dict(infos or {})
        infos["train_cfg"] = self.cfg
        infos["cenet_optimizer_state_dict"] = self.alg.cenet_optimizer.state_dict()
        super().save(path, infos=infos)

    def _check_rollout_finite(self, iteration: int, **tensors):
        """在写入 rollout 前拒绝非有限数据，保留出错环境与奖励项线索。"""
        invalid = {
            name: ~torch.isfinite(value).reshape(self.env.num_envs, -1).all(dim=1)
            for name, value in tensors.items()
        }
        if not torch.stack(list(invalid.values())).any().item():
            return
        details = []
        for name, mask in invalid.items():
            ids = mask.nonzero(as_tuple=False).flatten().tolist()
            if ids:
                details.append(f"{name}: env_ids={ids[:16]}, count={len(ids)}")
        # RewardManager 的逐项贡献仍保留当前步，可直接定位 Inf/NaN 来源。
        manager = getattr(getattr(self.env, "unwrapped", None), "reward_manager", None)
        contributions = getattr(manager, "_step_reward", None)
        if contributions is not None:
            bad_terms = (~torch.isfinite(contributions)).any(dim=0).nonzero(as_tuple=False).flatten().tolist()
            details.append(f"reward_terms={[manager.active_terms[index] for index in bad_terms]}")
        raise FloatingPointError(
            f"PPO iteration {iteration}: rollout 出现 NaN/Inf，已在写入训练存储前停止；"
            + "; ".join(details)
        )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        if self.log_dir is not None and self.writer is None:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs, extras = self.env.get_observations()
        history_obs = extras["observations"]["prop_history"].to(self.device)
        obs = self.alg.policy.current_frame(history_obs)
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs).to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        # 回合累计采用 float64，避免逐步有限的 float32 奖励在求和时溢出。
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float64, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    true_vel = self._extract_true_vel(privileged_obs)
                    actions = self.alg.act(obs, history_obs, privileged_obs, true_vel)
                    self._check_rollout_finite(it, actions=actions)
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    history_obs = infos["observations"]["prop_history"].to(self.device)
                    obs = self.alg.policy.current_frame(history_obs)
                    next_obs = obs
                    if self.privileged_obs_type is not None:
                        privileged_obs = infos["observations"][self.privileged_obs_type].to(self.device)
                    self._check_rollout_finite(
                        it, rewards=rewards, obs=obs, history_obs=history_obs, privileged_obs=privileged_obs,
                    )
                    self.alg.process_env_step(rewards, dones, infos, next_obs=next_obs)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    invalid = infos.get("numerical_invalid", torch.zeros_like(dones, dtype=torch.bool)).to(self.device).bool()
                    new_ids = ((dones > 0) & ~invalid).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    ended = (dones > 0) | invalid
                    cur_reward_sum[ended] = 0
                    cur_episode_length[ended] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(privileged_obs)

            # AdaBoot reward_cv 模式：提供回合奖励变异系数。
            if isinstance(self.alg, DreamWaQPPO) and self.alg.adaboot_enabled and self.alg.adaboot_mode == "reward_cv":
                # 不可用时显式清除旧 CV，回退到 p_min，避免沿用过时统计。
                self.alg.set_reward_cv(_compute_reward_cv(rewbuffer))

            loss_dict = self.alg.update()
            # clear() only resets the storage cursor. Update moments from the completed
            # rollout after optimization, so the next rollout uses one frozen transform.
            # The first rollout uses identity statistics instead of reset-only variance.
            valid = self.alg.storage.valid_transitions.flatten()
            if self.alg.policy.observation_normalization and valid.any():
                self.alg.policy.update_normalization(
                    self.alg.storage.history_observations.flatten(0, 1)[valid],
                    self.alg.storage.privileged_observations.flatten(0, 1)[valid],
                )

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                # CENet 指标（任务要求的 cenet/* 标签）。
                for key, value in getattr(self.alg, "cenet_metrics", {}).items():
                    self.writer.add_scalar(key, value, it)
                # 任务要求的 policy/* 标签（与 Train/* 同源）。
                if len(rewbuffer) > 0:
                    self.writer.add_scalar("policy/mean_reward", statistics.mean(rewbuffer), it)
                    self.writer.add_scalar("policy/episode_length", statistics.mean(lenbuffer), it)
                # 课程指标。
                self._log_curriculum(it)
                self._log_gait(it)
                # 标准日志（Loss/*、Train/*、Perf/*、Episode/*）。
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
                    self._export_curriculum_data(os.path.join(self.log_dir, f"curriculum_{it}.npz"))

            ep_infos.clear()

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self._export_curriculum_data(
                os.path.join(self.log_dir, f"curriculum_{self.current_learning_iteration}.npz")
            )

    """
    课程学习日志
    """

    def _log_gait(self, it: int):
        """逐脚记录触地占空比和长时间悬空比例，避免平均奖励掩盖跛行。"""
        if self.writer is None:
            return
        env = self.env.unwrapped
        if "feet_contact_balance" not in env.reward_manager.active_terms:
            return
        term = env.reward_manager.get_term_cfg("feet_contact_balance").func
        maximum_air_time = env.reward_manager.get_term_cfg("feet_air_time_excess").params["maximum_air_time"]
        for index, leg in enumerate(term.foot_names):
            self.writer.add_scalar(
                f"gait/{leg}/contact_fraction", term.contact_fraction[:, index].mean().item(), it,
            )
            self.writer.add_scalar(
                f"gait/{leg}/air_time_max_s", term.air_time[:, index].max().item(), it,
            )
            self.writer.add_scalar(
                f"gait/{leg}/long_air_fraction",
                (term.air_time[:, index] > maximum_air_time).float().mean().item(), it,
            )

    def _log_curriculum(self, it: int):
        if self.writer is None:
            return
        try:
            import torch as _torch

            env = self.env.unwrapped
            # ---- 地形课程 ----
            terrain = env.scene.terrain
            if terrain is not None and getattr(terrain, "terrain_levels", None) is not None:
                levels = terrain.terrain_levels.float()
                self.writer.add_scalar("curriculum/terrain/mean_level", levels.mean().item(), it)
                max_level = int(terrain.max_terrain_level) if hasattr(terrain, "max_terrain_level") else 0
                if max_level > 0:
                    for level in range(max_level):
                        frac = (terrain.terrain_levels == level).float().mean().item()
                        self.writer.add_scalar(f"curriculum/terrain/level_{level:02d}_frac", frac, it)
            stats = getattr(env, "terrain_curriculum_stats", None)
            if stats is not None:
                self.writer.add_scalar("curriculum/terrain/move_up_total", float(stats["move_up_count"]), it)
                self.writer.add_scalar("curriculum/terrain/move_down_total", float(stats["move_down_count"]), it)
                self.writer.add_scalar("curriculum/terrain/hold_total", float(stats["hold_count"]), it)
                for level in range(stats["num_levels"]):
                    if stats["count"][level] > 0:
                        self.writer.add_scalar(
                            f"curriculum/terrain/level_{level:02d}_mean_reward",
                            stats["reward_sum"][level] / stats["count"][level],
                            it,
                        )
                        self.writer.add_scalar(
                            f"curriculum/terrain/level_{level:02d}_mean_length",
                            stats["length_sum"][level] / stats["count"][level],
                            it,
                        )
            # ---- 指令课程 ----
            command_term = env.command_manager.get_term("base_velocity")
            grid = getattr(command_term, "grid", None)
            if grid is not None:
                self.writer.add_scalar("curriculum/command/activated_cells", int(grid.num_activated), it)
                self.writer.add_scalar("curriculum/command/vx_range", float(grid.active_vx_range), it)
                self.writer.add_scalar("curriculum/command/yaw_range", float(grid.active_yaw_range), it)
                self.writer.add_scalar("curriculum/command/expansions", int(grid.expansion_count), it)
                for key, value in command_term.metrics.items():
                    if isinstance(value, _torch.Tensor) and value.numel() == 1:
                        self.writer.add_scalar(f"curriculum/command/{key}", float(value), it)
                    elif isinstance(value, _torch.Tensor) and value.ndim == 1:
                        mean_val = value[value > 0].mean()
                        if mean_val.numel() > 0:
                            self.writer.add_scalar(f"curriculum/command/{key}_mean", float(mean_val), it)
        except Exception as exc:  # noqa: BLE001
            print(f"[DreamWaQRunner] 课程日志记录失败（不影响训练）: {exc}")

    def _export_curriculum_data(self, path: str):
        """导出指令课程网格数据（中心、概率、采样数、成功率），便于离线可视化。"""
        try:
            import numpy as np

            command_term = self.env.unwrapped.command_manager.get_term("base_velocity")
            grid = getattr(command_term, "grid", None)
            if grid is None:
                return
            np.savez(
                path,
                vx_centers=grid.vx_centers.cpu().numpy(),
                yaw_centers=grid.yaw_centers.cpu().numpy(),
                probs=grid.probs.cpu().numpy(),
                activated=grid.activated.cpu().numpy(),
                samples=grid.samples.cpu().numpy(),
                success_rate=grid.success_ema.cpu().numpy(),
                err_vx_ema=grid.err_vx_ema.cpu().numpy(),
                err_yaw_ema=grid.err_yaw_ema.cpu().numpy(),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[DreamWaQRunner] 课程数据导出失败（不影响训练）: {exc}")
