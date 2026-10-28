#!/usr/bin/env python
"""验收显式 PD 下的纯 CENet 确定性策略；默认平地零指令和 vx=0.3。

返回码：0 全部通过，2 有案例未达标；加载/仿真异常为非零。
训练时 AdaBoot 配置不影响本入口；Actor 只读取原始 prop_history。
"""

import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    # AppLauncher probes parse_known_args before adding its own --help handling.
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--duration", type=float, default=20.0, help="每个指令案例的秒数")
    parser.add_argument("--warmup", type=float, default=1.0, help="只排除跟踪统计；期间终止仍算失败")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--command", type=float, nargs=3, action="append", metavar=("VX", "VY", "WZ"),
                        help="可重复指定；默认 [0,0,0] 和 [0.3,0,0]")
    parser.add_argument("--linear_tolerance", type=float, help="覆盖平面速度 RMSE 门槛；站立默认 .05，移动 .15")
    parser.add_argument("--yaw_tolerance", type=float, help="覆盖偏航角速度 RMSE 门槛；站立默认 .10，移动 .20")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/acceptance.json"))
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    args = parser.parse_args()
    commands = args.command or [(0., 0., 0.), (.3, 0., 0.)]
    if args.checkpoint is None or not args.checkpoint.is_file():
        parser.error(f"checkpoint 不存在: {args.checkpoint}")
    if args.num_envs < 1 or not (math.isfinite(args.duration) and math.isfinite(args.warmup)
                               and 0 <= args.warmup < args.duration):
        parser.error("num_envs 必须为正；duration 必须大于非负 warmup，且二者有限")
    if any(not math.isfinite(x) for cmd in commands for x in cmd):
        parser.error("command 必须为有限数")
    for tolerance in (args.linear_tolerance, args.yaw_tolerance):
        if tolerance is not None and (not math.isfinite(tolerance) or tolerance <= 0):
            parser.error("验收门槛必须为有限正数")

    launcher = AppLauncher(args)
    env = None
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "policy": "deterministic_cenet_actor", "velocity_source": "cenet_only",
        "control": "explicit_pd", "terrain": "flat", "seed": args.seed,
        "num_envs": args.num_envs, "duration": args.duration, "warmup": args.warmup,
        "passed": False, "cases": [],
    }
    try:
        import gymnasium as gym
        import torch
        import bpx_dreamwaq.tasks  # noqa: F401
        from isaaclab.envs.mdp import UniformVelocityCommandCfg
        from bpx_dreamwaq.tasks.rough_env_cfg import BPXRoughEnvCfg_PLAY
        from bpx_dreamwaq.tasks.agents.checkpoint import build_policy_from_checkpoint
        from bpx_dreamwaq.tasks.evaluation import AcceptanceMetrics

        cfg = BPXRoughEnvCfg_PLAY()
        # PLAY 与训练复用同一显式 PD 配置，避免验收参数独立漂移。
        cfg.seed = args.seed
        cfg.scene.num_envs = args.num_envs
        cfg.scene.env_spacing = 3.0
        cfg.episode_length_s = args.duration + 10.0
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
        cfg.curriculum.terrain_levels = None
        if args.device is not None:
            cfg.sim.device = args.device
        cfg.commands.base_velocity = UniformVelocityCommandCfg(
            asset_name="robot", resampling_time_range=(args.duration + 10., args.duration + 10.),
            rel_standing_envs=0., heading_command=False,
            ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(0., 0.), lin_vel_y=(0., 0.), ang_vel_z=(0., 0.)),
        )
        cfg.events.reset_base.params["pose_range"] = {}
        cfg.events.reset_robot_joints.params.update(position_range=(0., 0.), velocity_range=(0., 0.))
        deployable, info = build_policy_from_checkpoint(str(args.checkpoint))
        report["checkpoint_iteration"] = info["iteration"]
        env = gym.make("BPX-Velocity-Rough-DreamWaQ-Play-v0", cfg=cfg)
        base = env.unwrapped
        policy = deployable.to(base.device).eval()
        robot = base.scene["robot"]
        if any(actuator.is_implicit_model for actuator in robot.actuators.values()):
            raise RuntimeError("部署验收要求全部受控关节使用显式力矩执行器")
        dt = cfg.sim.dt * cfg.decimation
        total_steps = math.ceil(args.duration / dt)
        warmup_steps = math.ceil(args.warmup / dt)
        if warmup_steps >= total_steps:
            raise ValueError("duration/warmup 取整后没有可评分的策略步")
        report["physics_dt"] = cfg.sim.dt
        report["policy_dt"] = dt
        report["control_parameters"] = {
            "kp": cfg.scene.robot.actuators["legs"].stiffness,
            "kd": cfg.scene.robot.actuators["legs"].damping,
            "torque_limit": cfg.scene.robot.actuators["legs"].effort_limit,
            "velocity_limit_sim": cfg.scene.robot.actuators["legs"].velocity_limit_sim,
        }
        for command in commands:
            ranges = base.command_manager.get_term("base_velocity").cfg.ranges
            ranges.lin_vel_x = (command[0], command[0])
            ranges.lin_vel_y = (command[1], command[1])
            ranges.ang_vel_z = (command[2], command[2])
            obs, _ = env.reset(seed=args.seed)
            standing = all(x == 0 for x in command)
            metrics = AcceptanceMetrics(
                args.num_envs, command,
                args.linear_tolerance or (.05 if standing else .15),
                args.yaw_tolerance or (.10 if standing else .20),
            )
            steps = 0
            # Keep simulator buffers normal tensors so reset between cases is legal.
            with torch.no_grad():
                while steps < total_steps and launcher.app.is_running():
                    history = obs["prop_history"]
                    action = policy(history)
                    if not torch.isfinite(action).all():
                        raise FloatingPointError("确定性策略输出 NaN/Inf")
                    estimate, _ = policy.policy.cenet.deterministic_latent(policy.policy.normalize_history(history))
                    estimate = estimate / policy.policy.velocity_scale  # Report physical m/s.
                    # Snapshot before env.step: terminal states may be auto-reset inside step.
                    velocity = robot.data.root_lin_vel_b.cpu().numpy().copy()
                    yaw_rate = robot.data.root_ang_vel_b[:, 2].cpu().numpy().copy()
                    tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1., 1.)).cpu().numpy()
                    obs, _, terminated, truncated, _ = env.step(action)
                    metrics.update(velocity, yaw_rate, estimate.cpu().numpy(), tilt,
                                   terminated.cpu().numpy(), truncated.cpu().numpy(), score=steps >= warmup_steps)
                    steps += 1
            result = metrics.result(completed=steps == total_steps)
            result["simulated_seconds"] = steps * dt
            report["cases"].append(result)
            print(f"[acceptance] command={command}: {'PASS' if result['passed'] else 'FAIL'}, "
                  f"terminations={result['terminated_count']}, "
                  f"max_linear_rmse={result['max_linear_rmse']:.4f}, "
                  f"max_yaw_rmse={result['max_yaw_rmse']:.4f}", flush=True)
        report["passed"] = all(case["passed"] for case in report["cases"])
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        raise
    finally:
        exit_code = 1 if sys.exc_info()[0] is not None else (0 if report["passed"] else 2)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        print(f"[acceptance] report: {args.output}", flush=True)
        if env is not None:
            env.close()
        sys.stdout.flush()
        sys.stderr.flush()
        # Native fast shutdown can exit before Python returns. Set Kit's code first;
        # disabling fast shutdown causes native teardown crashes on this Sim stack.
        launcher.app.app.post_quit(exit_code)
        launcher.app.close()
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
