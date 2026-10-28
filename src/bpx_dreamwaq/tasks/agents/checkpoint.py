"""Shared checkpoint loader for deterministic deployment export and acceptance."""

import torch

from .dreamwaq_actor_critic import DeployableDreamWaQPolicy, DreamWaQActorCritic
from ..obs_contract import ACTION_DIM, single_obs_dim


def build_policy_from_checkpoint(path: str) -> tuple[DeployableDreamWaQPolicy, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    train_cfg = (checkpoint.get("infos") or {}).get("train_cfg") or {}
    policy_cfg = dict(train_cfg.get("policy") or {})
    policy_cfg.pop("class_name", None)
    decoder_input = state["cenet.decoder.0.weight"].shape[1]
    latent_dim = policy_cfg.get("cenet_latent_dim", 16)
    vel_dim = policy_cfg.get("cenet_vel_dim", 3)
    if decoder_input not in (latent_dim, latent_dim + vel_dim):
        raise ValueError("checkpoint decoder input dimension is invalid")
    policy_cfg.setdefault("cenet_decoder_use_velocity", decoder_input == latent_dim + vel_dim)
    fixed_in_state = "obs_normalizer.scale" in state
    policy_cfg.setdefault("fixed_observation_scaling", fixed_in_state)
    if policy_cfg["fixed_observation_scaling"] != fixed_in_state:
        raise ValueError("checkpoint 固定缩放配置与权重不一致")
    policy_cfg["observation_normalization"] = train_cfg.get(
        "empirical_normalization", "obs_normalizer.mean" in state
    )
    if not policy_cfg["observation_normalization"] and "obs_normalizer.mean" in state:
        raise ValueError("checkpoint 配置禁用归一化，但权重包含归一化统计；请核对 checkpoint 元数据")

    # 部署只需要 cenet.* 与 actor.* 权重；Critic 维度与导出无关，
    # 过滤掉 critic.*（strict=False 不容忍形状不匹配，必须先剔除）。
    policy = DreamWaQActorCritic(
        num_actor_obs=single_obs_dim(),
        num_critic_obs=1,  # 占位：Critic 权重不参与部署
        num_actions=ACTION_DIM,
        **policy_cfg,
    )
    deploy_state = {key: value for key, value in state.items() if not key.startswith(("critic.", "critic_normalizer."))}
    # 注意：rsl_rl 的 ActorCritic.load_state_dict 覆写后返回 bool，
    # 这里直接用 nn.Module 的实现以获得 (missing, unexpected) 元组。
    missing, unexpected = torch.nn.Module.load_state_dict(policy, deploy_state, strict=False)
    needed = [key for key in missing if key.startswith(("cenet.", "actor.", "obs_normalizer."))]
    if needed:
        raise RuntimeError(f"checkpoint 缺少部署必需参数: {needed[:5]} ...")
    deployable = DeployableDreamWaQPolicy(policy)
    deployable.eval()
    return deployable, {
        "iteration": checkpoint.get("iter"),
        "history_length": policy.history_length,
        "history_dim": policy.num_history_obs,
        "actor_input_dim": single_obs_dim() + policy.cenet_vel_dim + policy.cenet_latent_dim,
        "latent_dim": policy.cenet_latent_dim,
        "fixed_observation_scaling": policy.fixed_observation_scaling,
        "unused_checkpoint_keys": [key for key in unexpected],
    }

