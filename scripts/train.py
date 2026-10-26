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
import os
import random
import sys
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
parser.add_argument("--resume", action="store_true", help="Resume from a checkpoint.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to load.")
parser.add_argument("--run_name", type=str, default="", help="Sub-directory name for this run.")
# append AppLauncher cli args and default to headless training
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows after the simulator is launched."""

import gymnasium as gym
import numpy as np
import torch

import bpx_dreamwaq.tasks  # noqa: F401  — registers BPX gym environments
from bpx_dreamwaq.tasks.agents.dreamwaq_ppo import DreamWaQRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


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
    if args_cli.adaboot:
        agent_cfg.algorithm.adaboot_enabled = True
    if args_cli.disable_adaboot:
        agent_cfg.algorithm.adaboot_enabled = False
    if args_cli.adaboot_mode is not None:
        agent_cfg.algorithm.adaboot_mode = args_cli.adaboot_mode

    # 课程开关在 __post_init__ 中接线，覆盖后需要重建内部配置。
    env_cfg.__post_init__()

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
    runner = DreamWaQRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)

    if args_cli.resume:
        ckpt = args_cli.checkpoint
        if ckpt is None:
            raise ValueError("--resume 需要同时指定 --checkpoint")
        print(f"[train] 从 checkpoint 恢复: {ckpt}")
        runner.load(ckpt, load_optimizer=True)

    print(f"[train] 日志目录: {log_dir}")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
