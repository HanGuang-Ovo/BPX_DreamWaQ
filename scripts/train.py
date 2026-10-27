#!/usr/bin/env python
"""BPX DreamWaQ 训练入口（Isaac Lab 2.2.1 + rsl-rl 2.3.3）。

用法示例（在仓库根目录；配置默认值见 tasks/rough_env_cfg.py 与
tasks/agents/rsl_rl_ppo_cfg.py，命令行覆盖优先）：

    # 默认配置训练（当前默认：双课程开 + AdaBoot 开 reward_cv 模式）
    bash scripts/training_python.sh scripts/train.py --headless --num_envs 512

    # 短时冒烟测试
    bash scripts/training_python.sh scripts/train.py --headless --num_envs 16 --max_iterations 3

    # 课程/AdaBoot 开关（双向可配，覆盖配置文件默认值）
    #   地形课程:  --disable_terrain_curriculum
    #   指令课程:  --enable_command_curriculum / --disable_command_curriculum
    #   AdaBoot:   --adaboot / --disable_adaboot [--adaboot_mode error|reward_cv|linear]
    bash scripts/training_python.sh scripts/train.py --headless --num_envs 512 \
        --disable_command_curriculum --disable_adaboot

    # 关闭全部课程与 AdaBoot（原始 PPO + CENet 基线）
    bash scripts/training_python.sh scripts/train.py --headless --num_envs 512 \
        --disable_terrain_curriculum --disable_command_curriculum --disable_adaboot

    # 恢复训练
    bash scripts/training_python.sh scripts/train.py --headless --num_envs 512 \
        --resume --checkpoint outputs/training/bpx_dreamwaq/<run>/model_100.pt
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import random
import sys
import traceback
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train the BPX DreamWaQ policy with RSL-RL PPO.")
parser.add_argument("--task", type=str, default="BPX-Velocity-Rough-DreamWaQ-v0", help="Registered task id.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Number of PPO iterations.")
parser.add_argument("--disable_terrain_curriculum", action="store_true", help="Turn off terrain curriculum.")
parser.add_argument("--enable_command_curriculum", action="store_true", help="Turn on Grid-Adaptive command curriculum.")
parser.add_argument("--disable_command_curriculum", action="store_true", help="Turn off Grid-Adaptive command curriculum.")
parser.add_argument("--num_terrain_levels", type=int, default=None, help="Terrain difficulty levels.")
parser.add_argument("--adaboot", action="store_true", help="Enable AdaBoot velocity bootstrapping.")
parser.add_argument("--disable_adaboot", action="store_true", help="Disable AdaBoot velocity bootstrapping.")
parser.add_argument("--adaboot_mode", type=str, default=None, choices=["error", "reward_cv", "linear"])
loading = parser.add_mutually_exclusive_group()
loading.add_argument("--resume", action="store_true", help="Resume model, optimizer and iteration from a checkpoint.")
loading.add_argument("--finetune", action="store_true", help="Load weights/normalization with a fresh optimizer and iteration counter.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to load.")
parser.add_argument("--learning_rate", type=float, default=None, help="Initial learning rate (fine-tuning default: 1e-4).")
parser.add_argument("--run_name", type=str, default="", help="Sub-directory name for this run.")
# append AppLauncher cli args and default to headless training
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli, _ = parser.parse_known_args()
if bool(args_cli.checkpoint) != bool(args_cli.resume or args_cli.finetune):
    parser.error("--checkpoint 必须与 --resume 或 --finetune 一起使用")
if args_cli.checkpoint and not os.path.isfile(args_cli.checkpoint):
    parser.error(f"checkpoint 不存在: {args_cli.checkpoint}")
if args_cli.learning_rate is not None and (
    not math.isfinite(args_cli.learning_rate) or args_cli.learning_rate <= 0
):
    parser.error("--learning_rate 必须为有限正数")
if args_cli.resume and args_cli.learning_rate is not None:
    parser.error("--resume 保留优化器；指定新学习率请使用 --finetune")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows after the simulator is launched."""

import gymnasium as gym
import numpy as np
import torch

import bpx_dreamwaq.tasks  # noqa: F401  — registers BPX gym environments
from bpx_dreamwaq.tasks.agents.dreamwaq_ppo import DreamWaQRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.io import dump_yaml


def _load_class(entry_point: str):
    module_path, _, attr = entry_point.partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def main():
    spec = gym.spec(args_cli.task)
    env_cfg_entry = spec.kwargs["env_cfg_entry_point"]
    agent_cfg_entry = spec.kwargs["rsl_rl_cfg_entry_point"]
    env_cfg = _load_class(env_cfg_entry)()
    agent_cfg = _load_class(agent_cfg_entry)()

    # ---- 命令行覆盖 ----
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.num_terrain_levels is not None:
        env_cfg.num_terrain_levels = args_cli.num_terrain_levels
    if args_cli.disable_terrain_curriculum:
        env_cfg.enable_terrain_curriculum = False
    if args_cli.enable_command_curriculum:
        env_cfg.enable_grid_adaptive_command_curriculum = True
    if args_cli.disable_command_curriculum:
        env_cfg.enable_grid_adaptive_command_curriculum = False
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed
    if args_cli.finetune:
        agent_cfg.algorithm.learning_rate = 1.0e-4
        # 迁移微调默认让 Actor 全程使用估计速度；可显式 --adaboot 覆盖。
        agent_cfg.algorithm.adaboot_enabled = False
    if args_cli.learning_rate is not None:
        agent_cfg.algorithm.learning_rate = args_cli.learning_rate
    if args_cli.adaboot:
        agent_cfg.algorithm.adaboot_enabled = True
    if args_cli.disable_adaboot:
        agent_cfg.algorithm.adaboot_enabled = False
    if args_cli.adaboot_mode is not None:
        agent_cfg.algorithm.adaboot_mode = args_cli.adaboot_mode

    # 课程开关在 __post_init__ 中接线，覆盖后需要重建内部配置。
    env_cfg.__post_init__()
    env_cfg.seed = agent_cfg.seed

    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    # ---- 随机种子 ----
    if agent_cfg.seed is not None:
        random.seed(agent_cfg.seed)
        np.random.seed(agent_cfg.seed)
        torch.manual_seed(agent_cfg.seed)

    # ---- 日志目录 ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args_cli.run_name}" if args_cli.run_name else timestamp
    log_root = os.environ.get("BPX_LOG_DIR", os.path.join("outputs", "training"))
    log_dir = os.path.join(log_root, agent_cfg.experiment_name, run_name)
    os.makedirs(log_dir, exist_ok=True)

    # ---- 环境、包装器与 Runner ----
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    train_cfg = agent_cfg.to_dict()
    train_cfg["true_vel_slice"] = tuple(agent_cfg.true_vel_slice)
    actuator_cfg = env_cfg.scene.robot.actuators["legs"]
    train_cfg["control_contract"] = {
        "actuator": actuator_cfg.class_type.__name__,
        "stiffness": actuator_cfg.stiffness, "damping": actuator_cfg.damping,
        "torque_limit": actuator_cfg.effort_limit,
        "velocity_limit_sim": actuator_cfg.velocity_limit_sim,
        "action_scale": env_cfg.actions.joint_pos.scale, "physics_dt": env_cfg.sim.dt,
        "decimation": env_cfg.decimation,
    }
    train_cfg["source_checkpoint"] = os.path.abspath(args_cli.checkpoint) if args_cli.checkpoint else None
    train_cfg["load_mode"] = "finetune" if args_cli.finetune else "resume" if args_cli.resume else "new"
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), train_cfg)
    runner = DreamWaQRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)

    try:
        if args_cli.finetune:
            print(f"[train] 显式 PD 微调，加载权重/归一化，重建优化器: {args_cli.checkpoint}")
            runner.load_for_finetuning(args_cli.checkpoint)
        elif args_cli.resume:
            print(f"[train] 从 checkpoint 恢复: {args_cli.checkpoint}")
            runner.load(args_cli.checkpoint, load_optimizer=True)

        print(f"[train] 日志目录: {log_dir}")
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
    finally:
        if runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.app.post_quit(1 if sys.exc_info()[0] is not None else 0)
        simulation_app.close()
