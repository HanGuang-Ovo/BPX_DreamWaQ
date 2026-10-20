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
   与 PPO 共用同一优化器（CENet 是策略的子模块，参数已注册）。
3. ``DreamWaQRunner``：适配 ``OnPolicyRunner``——解析自定义策略/算法类，
   在 rollout 中传递历史观测与真实速度标签，记录 CENet、AdaBoot、课程指标。

AdaBoot（论文公式 p = 1 - tanh(CV(R))，CV 为回合奖励变异系数）在此实现为
可选机制，支持四种模式：``error``（默认，按速度估计误差过渡，任务伪代码风格）、
``reward_cv``（论文公式）、``linear``（线性爬升）与 ``off``（关闭即 p=1，
始终使用估计速度）。默认关闭（基础 CENet + PPO 联合训练）。
"""

from __future__ import annotations

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

from .dreamwaq_actor_critic import DreamWaQActorCritic


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
        self.next_observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=device)

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
        self.step += 1

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """与基类相同的抽样逻辑，额外产出 CENet 相关字段。"""
        if self.training_type != "rl":
            raise ValueError("该函数仅用于强化学习训练。")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

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
            for i in range(num_mini_batches):
                batch_idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
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
        self.lambda_vel = lambda_vel
        self.lambda_pred = lambda_pred
        self.beta_kl = beta_kl
        self.ppo_grad_to_cenet = ppo_grad_to_cenet

        self.adaboot_enabled = adaboot_enabled
        self.adaboot_mode = adaboot_mode
        self.adaboot_error_threshold = adaboot_error_threshold
        self.adaboot_ema_alpha = adaboot_ema_alpha
        self.adaboot_ramp_iterations = max(1, adaboot_ramp_iterations)
        self.adaboot_p_min = adaboot_p_min
        # 当前 AdaBoot 概率（使用估计速度的概率）。关闭时恒为 1。
        self.adaboot_p = 1.0
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
            vel_actor = torch.where(use_est, features[:, :vel_dim], true_vel.detach())
            features_actor = torch.cat([vel_actor, features[:, vel_dim:]], dim=-1)
            self._use_est_total += use_est.numel()
            self._use_est_count += use_est.sum().item()
        else:
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
        self.transition.true_velocities = true_vel.detach()
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_obs: torch.Tensor):
        """记录转移；``next_obs`` 为执行动作后的下一时刻策略观测。"""
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # 无限时域任务的 timeout 自举（沿用基类逻辑）。
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
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
        self._reward_cv = reward_cv

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
        ) in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Actor：默认复用 rollout 保存的 CENet 特征，保证新旧策略概率比
            # 在相同输入上计算。可选 ppo_grad_to_cenet=True 时改为带梯度重算。
            if self.ppo_grad_to_cenet:
                vel_hat, mu, _, _ = self.policy.cenet(history_batch)
                vel_dim = self.policy.cenet_vel_dim
                cenet_features_batch = torch.cat([vel_hat, mu], dim=-1)
                self.policy.set_cenet_features(cenet_features_batch)
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

            # CENet 辅助损失：done 转移的下一观测属于新 episode，必须屏蔽。
            valid_mask = 1.0 - dones_batch.float()
            aux = self.policy.cenet.compute_aux_losses(
                history_batch, next_obs_batch, true_vel_batch, valid_mask
            )
            vel_loss = aux["vel_loss"]
            pred_loss = aux["pred_loss"]
            kl_loss = aux["kl_loss"]
            aux_loss = self.lambda_vel * vel_loss + self.lambda_pred * pred_loss + self.beta_kl * kl_loss

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + aux_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_vel_loss += vel_loss.item()
            mean_pred_loss += pred_loss.item()
            mean_kl_loss += kl_loss.item()
            mean_vel_mse += (aux["vel_error"] ** 2).mean().item()
            vel_err_axis += aux["vel_error"].mean(dim=0)

        num_updates = self.num_learning_epochs * self.num_mini_batches
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
        if self._vel_error_ema is None:
            self._vel_error_ema = vel_mae
        else:
            self._vel_error_ema = (
                self.adaboot_ema_alpha * self._vel_error_ema + (1.0 - self.adaboot_ema_alpha) * vel_mae
            )

        self.storage.clear()

        self.cenet_metrics = {
            "cenet/velocity_mse": mean_vel_mse,
            "cenet/velocity_error_x": float(vel_err_axis[0]),
            "cenet/velocity_error_y": float(vel_err_axis[1]),
            "cenet/velocity_error_z": float(vel_err_axis[2]),
            "cenet/velocity_mae": vel_mae,
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
        if self.empirical_normalization:
            raise ValueError("DreamWaQ 部署契约不包含观测归一化，请保持 empirical_normalization=False")
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

    def save(self, path: str, infos=None):
        infos = dict(infos or {})
        infos["train_cfg"] = self.cfg
        super().save(path, infos=infos)

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
        obs = obs.to(self.device)
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs).to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    true_vel = self._extract_true_vel(privileged_obs)
                    actions = self.alg.act(obs, history_obs, privileged_obs, true_vel)
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    next_obs = obs
                    obs = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    history_obs = infos["observations"]["prop_history"].to(self.device)
                    if self.privileged_obs_type is not None:
                        privileged_obs = infos["observations"][self.privileged_obs_type].to(self.device)
                    self.alg.process_env_step(rewards, dones, infos, next_obs=next_obs)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(privileged_obs)

            # AdaBoot reward_cv 模式：提供回合奖励变异系数。
            if isinstance(self.alg, DreamWaQPPO) and self.alg.adaboot_enabled and self.alg.adaboot_mode == "reward_cv":
                if len(rewbuffer) >= 2:
                    mean_reward = statistics.mean(rewbuffer)
                    if abs(mean_reward) > 1e-6:
                        self.alg.set_reward_cv(statistics.pstdev(rewbuffer) / abs(mean_reward))

            loss_dict = self.alg.update()

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
