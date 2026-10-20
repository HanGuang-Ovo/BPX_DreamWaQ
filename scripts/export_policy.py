#!/usr/bin/env python
"""从训练 checkpoint 导出部署策略（450 维历史观测 → 12 维动作）。

导出的融合模型 = CENet（确定性 μ）+ Actor：

- 输入：展平历史观测 (B, 450)，term-major、项内从最旧到最新，
  与 ``bpx_dreamwaq.runtime.history.ObservationHistory`` 输出一致；
- 输出：12 维关节位置动作（乘 0.5 加默认角后送 PD）；
- 不包含 Critic、真实速度、高度扫描或 VAE Decoder——部署推理不依赖
  任何特权信息。

用法：

    bash scripts/training_python.sh scripts/export_policy.py \
        --checkpoint outputs/training/bpx_dreamwaq/<run>/model_100.pt \
        --format both

不启动 Isaac Sim（仅 torch + rsl_rl）。
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from bpx_dreamwaq.tasks.agents.dreamwaq_actor_critic import (  # noqa: E402
    DeployableDreamWaQPolicy,
    DreamWaQActorCritic,
)
from bpx_dreamwaq.tasks.obs_contract import ACTION_DIM, history_obs_dim, single_obs_dim  # noqa: E402


def build_policy_from_checkpoint(path: str) -> tuple[DeployableDreamWaQPolicy, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    train_cfg = (checkpoint.get("infos") or {}).get("train_cfg") or {}
    policy_cfg = dict(train_cfg.get("policy") or {})
    policy_cfg.pop("class_name", None)

    # 部署只需要 cenet.* 与 actor.* 权重；Critic 维度与导出无关，
    # 过滤掉 critic.*（strict=False 不容忍形状不匹配，必须先剔除）。
    policy = DreamWaQActorCritic(
        num_actor_obs=single_obs_dim(),
        num_critic_obs=1,  # 占位：Critic 权重不参与部署
        num_actions=ACTION_DIM,
        **policy_cfg,
    )
    deploy_state = {key: value for key, value in state.items() if not key.startswith("critic.")}
    # 注意：rsl_rl 的 ActorCritic.load_state_dict 覆写后返回 bool，
    # 这里直接用 nn.Module 的实现以获得 (missing, unexpected) 元组。
    missing, unexpected = torch.nn.Module.load_state_dict(policy, deploy_state, strict=False)
    needed = [key for key in missing if key.startswith(("cenet.", "actor."))]
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
        "unused_checkpoint_keys": [key for key in unexpected],
    }


def main():
    parser = argparse.ArgumentParser(description="Export deployable DreamWaQ policy.")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练 checkpoint (.pt) 路径。")
    parser.add_argument("--out", type=str, default=None, help="输出文件前缀（默认 policies/policy）。")
    parser.add_argument(
        "--format", type=str, default="both", choices=["onnx", "torchscript", "both"], help="导出格式。"
    )
    args = parser.parse_args()

    deployable, info = build_policy_from_checkpoint(args.checkpoint)
    out_prefix = args.out or os.path.join("policies", "policy")
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    history_dim = history_obs_dim(info["history_length"])
    dummy = torch.zeros(1, history_dim)

    with torch.no_grad():
        actions = deployable(dummy)
    if actions.shape != (1, ACTION_DIM):
        raise RuntimeError(f"导出自检失败：期望输出 (1, {ACTION_DIM})，实际 {tuple(actions.shape)}")

    if args.format in ("torchscript", "both"):
        traced = torch.jit.trace(deployable, dummy)
        ts_path = f"{out_prefix}.pt"
        traced.save(ts_path)
        print(f"[export] TorchScript: {ts_path}")
    if args.format in ("onnx", "both"):
        onnx_path = f"{out_prefix}.onnx"
        # 部署契约要求固定形状 (1, 450) → (1, 12)：sim2real 诊断与
        # sim2sim 运行时均按单机器人 batch=1 校验。
        torch.onnx.export(
            deployable,
            dummy,
            onnx_path,
            input_names=["history_obs"],
            output_names=["actions"],
            opset_version=17,
        )
        print(f"[export] ONNX: {onnx_path}")

    print(f"[export] checkpoint 迭代: {info['iteration']}")
    print(f"[export] 输入: 展平历史观测 ({history_dim},) = {info['history_length']} 帧 × {single_obs_dim()} 维")
    print(f"[export] Actor 输入维度: {info['actor_input_dim']}（单帧 {single_obs_dim()} + vel 3 + latent {info['latent_dim']}）")
    print(f"[export] 输出: {ACTION_DIM} 维关节位置动作（部署侧 q = q_default + 0.5 × a）")


if __name__ == "__main__":
    main()
