#!/usr/bin/env python3
"""在 MuJoCo 中运行 BPX 的 Isaac Lab 策略。"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import sys

from bpx_dreamwaq.paths import CONFIGS_DIR

from bpx_dreamwaq.sim2sim.config import load_config
from bpx_dreamwaq.runtime.gamepad import (
    BTN_TL,
    BTN_TR,
    GamepadCommandSource,
    GamepadMapping,
    enumerate_gamepads,
    select_gamepad,
)
from bpx_dreamwaq.runtime.policy import create_policy
from bpx_dreamwaq.sim2sim.runner import Sim2SimRunner
from bpx_dreamwaq.runtime.supervisor import BehaviorSupervisor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "sim2sim" / "bpx_history.toml")
    parser.add_argument("--backend", choices=("torchscript", "onnx", "zero"), default="zero",
                        help="默认 zero，仅检查模型和控制框架；推理时显式选择 onnx/torchscript")
    parser.add_argument("--policy", type=Path, help="覆盖所选 ONNX 或 TorchScript 策略文件")
    parser.add_argument("--device", default="cpu", help="TorchScript 设备，例如 cpu 或 cuda:0")
    parser.add_argument("--vx", type=float, default=None, help="机体 x 方向目标线速度，m/s")
    parser.add_argument("--vy", type=float, default=None, help="机体 y 方向目标线速度，m/s")
    parser.add_argument("--wz", type=float, default=None, help="目标偏航角速度，rad/s")
    parser.add_argument("--duration", type=float, default=None, help="仿真时长，秒")
    parser.add_argument("--viewer", action="store_true", help="打开 MuJoCo 可视化窗口")
    parser.add_argument("--velocity-plot", action="store_true", help="打开 command 与实际速度追踪曲线窗口")
    parser.add_argument("--velocity-window", type=float, default=10.0, help="速度曲线时间窗，1–30 秒（默认 10）")
    parser.add_argument("--torque-plot", action="store_true", help="打开 12 关节实际输出力矩曲线窗口")
    parser.add_argument("--torque-window", type=float, default=10.0, help="力矩曲线时间窗，1–30 秒（默认 10）")
    parser.add_argument("--no-realtime", action="store_true", help="关闭实时限速，尽快完成仿真")
    parser.add_argument("--gamepad", action="store_true", help="用 Linux 游戏手柄实时生成速度指令")
    parser.add_argument("--list-gamepads", action="store_true", help="列出检测到的游戏手柄后退出")
    gamepad_selection = parser.add_mutually_exclusive_group()
    gamepad_selection.add_argument("--gamepad-device", type=Path, help="显式指定 /dev/input/eventN")
    gamepad_selection.add_argument("--gamepad-index", type=int, help="选择 --list-gamepads 输出的设备序号")
    parser.add_argument("--gamepad-deadzone", type=float, default=0.1, help="摇杆死区（默认 0.1）")
    parser.add_argument("--gamepad-max-vx", type=float, default=1.0, help="前后最大速度 m/s")
    parser.add_argument("--gamepad-max-vy", type=float, default=0.5, help="横向最大速度 m/s")
    parser.add_argument("--gamepad-max-wz", type=float, default=1.0, help="最大偏航角速度 rad/s")
    parser.add_argument("--gamepad-vx-axis", type=int, default=1, help="vx 对应的 Linux ABS 轴（默认 1/ABS_Y）")
    parser.add_argument("--gamepad-vy-axis", type=int, default=0, help="vy 对应的 Linux ABS 轴（默认 0/ABS_X）")
    parser.add_argument("--gamepad-wz-axis", type=int, default=3, help="wz 对应的 Linux ABS 轴（默认 3/ABS_RX）")
    parser.add_argument(
        "--gamepad-vx-sign", type=float, choices=(-1.0, 1.0), default=-1.0, help="vx 轴方向（默认 -1）"
    )
    parser.add_argument(
        "--gamepad-vy-sign", type=float, choices=(-1.0, 1.0), default=-1.0, help="vy 轴方向（默认 -1）"
    )
    parser.add_argument(
        "--gamepad-wz-sign", type=float, choices=(-1.0, 1.0), default=-1.0, help="wz 轴方向（默认 -1）"
    )
    parser.add_argument(
        "--gamepad-deadman",
        action="store_true",
        help=f"仅按住 LB（Linux 按钮码 {BTN_TL}）时接受非零指令",
    )
    parser.add_argument(
        "--supervisor",
        action="store_true",
        help="启用 WAITING_INIT/INIT/STAND/WALK/STOPPING/DISABLED 上层状态机",
    )
    parser.add_argument(
        "--auto-init",
        action="store_true",
        help="无手柄时自动启用 Supervisor 并触发 Init（摔倒回到趴姿会再次自动起身）",
    )
    termination_group = parser.add_mutually_exclusive_group()
    termination_group.add_argument(
        "--terminate-on-fall",
        action="store_true",
        help="机器人高度过低、倾角过大或躯干触地时提前结束（默认不提前结束）",
    )
    termination_group.add_argument(
        "--no-terminate",
        dest="terminate_on_fall",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(terminate_on_fall=False)
    parser.add_argument("--log", type=Path, default=None, help="保存策略频率的 CSV 日志")
    return parser.parse_args(argv)


def _run(args: argparse.Namespace) -> int:
    if args.list_gamepads:
        devices = enumerate_gamepads()
        if not devices:
            print("没有找到 Linux input 游戏手柄。")
            return 1
        for index, device in enumerate(devices):
            print(f"[{index}] {device.label}")
        return 0

    config = load_config(args.config)
    if args.policy is not None:
        if args.backend == "zero":
            raise ValueError("--policy 需要显式选择 --backend onnx 或 torchscript")
        config = replace(config, paths=replace(config.paths, policy=args.policy.expanduser().resolve()))
    if not config.paths.mjcf.is_file():
        raise FileNotFoundError(f"MJCF 文件不存在：{config.paths.mjcf}")
    if args.backend != "zero" and not config.paths.policy.is_file():
        backend_name = "ONNX" if args.backend == "onnx" else "TorchScript"
        raise FileNotFoundError(f"{backend_name} 文件不存在：{config.paths.policy}")

    command_defaults = config.command.as_tuple()
    command = (
        command_defaults[0] if args.vx is None else args.vx,
        command_defaults[1] if args.vy is None else args.vy,
        command_defaults[2] if args.wz is None else args.wz,
    )
    use_gamepad = args.gamepad or args.gamepad_device is not None or args.gamepad_index is not None
    policy = create_policy(
        backend=args.backend,
        path=config.paths.policy,
        observation_dimension=config.observation.dimension,
        action_dimension=config.observation.action_dimension,
        device=args.device,
    )
    use_supervisor = args.supervisor or args.auto_init or use_gamepad
    supervisor = BehaviorSupervisor(config.supervisor) if use_supervisor else None
    if use_supervisor:
        print(f"Init: 足端五次轨迹 + PD，标称时长 {config.init_controller.duration:.1f}s")
    gamepad_context = nullcontext(None)
    if use_gamepad:
        try:
            device = select_gamepad(args.gamepad_device, args.gamepad_index)
        except (RuntimeError, ValueError) as exc:
            print(f"手柄错误：{exc}", file=sys.stderr)
            return 1
        mapping = GamepadMapping(
            vx_axis=args.gamepad_vx_axis,
            vy_axis=args.gamepad_vy_axis,
            wz_axis=args.gamepad_wz_axis,
            vx_sign=args.gamepad_vx_sign,
            vy_sign=args.gamepad_vy_sign,
            wz_sign=args.gamepad_wz_sign,
        )
        gamepad_context = GamepadCommandSource(
            device=device,
            maximum_command=(args.gamepad_max_vx, args.gamepad_max_vy, args.gamepad_max_wz),
            deadzone=args.gamepad_deadzone,
            mapping=mapping,
            deadman_button=BTN_TL if args.gamepad_deadman else None,
            init_button=BTN_TR if use_supervisor else None,
        )

    runner = Sim2SimRunner(
        config,
        policy,
        supervisor=supervisor,
    )
    try:
        with gamepad_context as gamepad:
            source_label = f"gamepad ({gamepad.device.label})" if gamepad is not None else "fixed"
            print(
                f"BPX sim2sim: backend={args.backend}, source={source_label}, "
                f"command=[{command[0]}, {command[1]}, {command[2]}], "
                f"supervisor={'on' if use_supervisor else 'off'}, "
                f"physics={1.0 / config.simulation.timestep:.1f}Hz, "
                f"policy={1.0 / config.simulation.policy_dt:.1f}Hz"
            )
            if gamepad is not None and use_supervisor:
                print(f"控制流程：机器人以趴姿等待；按下 RB（Linux 按钮码 {BTN_TR}）启动 Init。")
            result = runner.run(
                command=command,
                command_source=gamepad.read if gamepad is not None else None,
                init_request_source=(
                    gamepad.consume_init_request if gamepad is not None and use_supervisor
                    else (lambda: True) if args.auto_init and use_supervisor
                    else None
                ),
                duration=args.duration,
                viewer=args.viewer,
                realtime=False if args.no_realtime else None,
                log_path=args.log,
                terminate_on_fall=args.terminate_on_fall,
                velocity_plot=args.velocity_plot,
                velocity_window=args.velocity_window,
                torque_plot=args.torque_plot,
                torque_window=args.torque_window,
            )
    except (OSError, ValueError) as exc:
        if not use_gamepad:
            raise
        print(f"手柄错误：{exc}", file=sys.stderr)
        print("请检查设备节点、轴编号以及 /dev/input/event* 的读取权限。", file=sys.stderr)
        return 1
    print(
        f"结束：simulated={result.simulated_seconds:.3f}s, policy_steps={result.policy_steps}, "
        f"reason={result.termination_reason}"
    )
    return 0 if result.termination_reason in ("duration", "viewer_closed") else 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return _run(args)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
