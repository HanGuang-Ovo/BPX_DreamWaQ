"""分阶段命令行。check/analyze/replay 不实例化 SDK。"""

import argparse
from contextlib import ExitStack
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

from bpx_dreamwaq.runtime.joints import SDK_JOINT_NAMES
from bpx_dreamwaq.runtime.gamepad import GamepadCommandSource, select_gamepad, BTN_TL, BTN_TR
from bpx_dreamwaq.paths import CONFIGS_DIR, OUTPUTS_DIR
from .config import load_config, positive
from .recording import JsonlWriter, load_samples, summarize
from .runner import replay, run_live
from .sdk_robot import SDKSource
from .safety import require_ready

DEFAULT_CONFIG = CONFIGS_DIR / "sim2real/runtime.toml"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def check(config):
    models = {}
    for mode in ("walk", "stand"):
        p = config.model_path(mode)
        models[mode] = {
            "path": str(p),
            "exists": p.is_file(),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None,
        }
    dependencies = {name: importlib.util.find_spec(name) is not None for name in ("numpy", "onnxruntime", "bpx_sdk")}
    return {
        "config": str(config.path),
        "calibration": str(config.calibration_path),
        "dependencies": dependencies,
        "models": models,
        "observe_blockers": [] if dependencies["bpx_sdk"] else ["当前 Python 未安装 bpx_sdk"],
        "replay_blockers": [] if dependencies["onnxruntime"] else ["当前 Python 未安装 onnxruntime"],
        "deploy_blockers": config.blockers("deploy"),
        "mapping": [
            {
                "policy_index": i,
                "name": j.name,
                "sdk_index": j.sdk_index,
                "sdk_documented_name": SDK_JOINT_NAMES[j.sdk_index],
                "sign": j.sign,
                "offset": j.offset,
                "alignment_verified": config.alignment_verified,
            }
            for i, j in enumerate(config.joints)
        ],
    }


def capture(config, output, duration, poll_hz, *, source_factory=SDKSource):
    duration = positive(duration, "duration")
    period = 1.0 / positive(poll_hz, "poll_hz")
    samples = []
    with JsonlWriter(
        output,
        {
            "mode": "observe",
            "config": str(config.path),
            "runtime": config.runtime,
            "calibration": config.calibration,
            "created_at": datetime.now().astimezone().isoformat(),
            "note": "RequestRobotState; SDK connect may perform automatic time synchronization",
        },
    ) as log:
        with source_factory(config, joint_control=False) as source:
            start = time.monotonic()
            next_print = start
            tick = 0
            while time.monotonic() - start < duration:
                sample = source.read()
                samples.append(sample)
                log.write({"type": "sample", "sample": sample.to_dict()})
                now = time.monotonic()
                if now >= next_print:
                    print(
                        json.dumps(
                            {
                                "elapsed_s": round(now - start, 3),
                                "connected": sample.connected,
                                "identity": sample.identity,
                                "sdk_q": sample.q,
                                "joint_age_s": sample.joint_age_s,
                                "imu_age_s": sample.imu_age_s,
                            },
                            ensure_ascii=False,
                        )
                    )
                    next_print = now + 1.0
                tick += 1
                time.sleep(max(0.0, start + tick * period - time.monotonic()))
    return summarize(samples)


def parser():
    p = argparse.ArgumentParser(description="BPX 分阶段 sim2real 工具")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = p.add_subparsers(dest="command", required=True)
    monitor = sub.add_parser("monitor", help="只读反馈、策略预览与三维姿态窗口")
    monitor.add_argument("--demo", action="store_true", help="模拟反馈，不连接 SDK")
    monitor.add_argument("--recording", type=Path, help="回放 inspect 录制，不连接 SDK")
    monitor.add_argument("--no-policy", action="store_true", help="只看反馈，不加载模型")
    monitor.add_argument("--output", type=Path, help="保存原始反馈 JSONL")
    monitor.add_argument("--duration", type=float, help="窗口自动关闭时间，秒")
    from .monitor import add_gamepad_arguments

    add_gamepad_arguments(monitor)
    sub.add_parser("check", help="离线检查配置、模型和依赖；不连接机器人")
    observe = sub.add_parser("inspect", help="只读状态录制；不发送运动命令")
    observe.add_argument("--output", type=Path, required=True)
    observe.add_argument("--duration", type=float, default=10.0)
    observe.add_argument("--poll-hz", type=float, default=100.0)
    analyze = sub.add_parser("analyze", help="分析只读录制")
    analyze.add_argument("recording", type=Path)
    analyze.add_argument("--output", type=Path)
    off = sub.add_parser("replay", help="离线策略推理；不连接 SDK")
    off.add_argument("recording", type=Path)
    off.add_argument("--output", type=Path, required=True)
    off.add_argument("--vx", type=float, default=0.0)
    off.add_argument("--vy", type=float, default=0.0)
    off.add_argument("--wz", type=float, default=0.0)
    run = sub.add_parser("run", help="默认仅检查运行资源；--live 执行实机控制")
    run.add_argument("--live", action="store_true")
    run.add_argument("--arm", action="store_true", help=argparse.SUPPRESS)
    run.add_argument("--duration", type=float, help="可选运行秒数；省略则运行至 Ctrl+C 或故障")
    run.add_argument("--output", type=Path, help="日志路径；省略则自动创建带时间戳的 JSONL")
    run.add_argument("--vx", type=float, default=0.0)
    run.add_argument("--vy", type=float, default=0.0)
    run.add_argument("--wz", type=float, default=0.0)
    run.add_argument("--command-source", choices=("gamepad", "constant"), default="gamepad",
                     help="默认手柄：RB 起身，摇杆行走；constant 仅提供固定速度，无起身请求")
    device = run.add_mutually_exclusive_group()
    device.add_argument("--gamepad-device", type=Path)
    device.add_argument("--gamepad-index", type=int)
    run.add_argument("--gamepad-deadman", action="store_true", help="按住 LB 才接受摇杆速度，RB 独立生效")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "monitor":
            from .monitor import launch, list_gamepads

            if args.list_gamepads:
                return list_gamepads()
            return launch(
                config,
                demo=args.demo,
                recording=args.recording,
                no_policy=args.no_policy,
                output=args.output,
                duration=args.duration,
                gamepad_device=args.gamepad_device,
                gamepad_index=args.gamepad_index,
            )
        if args.command == "check":
            result = check(config)
        elif args.command == "inspect":
            result = capture(config, args.output, args.duration, args.poll_hz)
        elif args.command == "analyze":
            _, samples = load_samples(args.recording)
            result = summarize(samples)
            if args.output:
                write_json(args.output, result)
        elif args.command == "replay":
            count = replay(
                config, args.recording, args.output, (args.vx, args.vy, args.wz)
            )
            result = {"policy_steps": count, "output": str(args.output), "mode": "offline"}
        elif args.command == "run":
            if not args.live:
                result = {
                    "mode": "preflight",
                    "blockers": config.blockers("deploy"),
                    "note": "未创建 SDK 对象；--live 才进入关节控制；仅检查有限数与四元数模长，退出尝试阻尼",
                }
            else:
                if args.output is None:
                    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S-%f")
                    args.output = OUTPUTS_DIR / "sim2real" / f"live-{stamp}.jsonl"
                require_ready(config, "deploy")
                duration = positive(args.duration, "duration") if args.duration is not None else None
                with ExitStack() as stack:
                    gamepad = None
                    if args.command_source == "gamepad":
                        if any((args.vx, args.vy, args.wz)):
                            raise ValueError("手柄模式不接受固定速度 --vx/--vy/--wz")
                        device = select_gamepad(args.gamepad_device, args.gamepad_index)
                        gamepad = stack.enter_context(GamepadCommandSource(
                            device, tuple(config.control["command_limit"]), init_button=BTN_TR,
                            deadman_button=BTN_TL if args.gamepad_deadman else None))
                        print("手柄已连接：等待 RB 起身；摇杆控制速度。退出尝试 kp=0、kd=1 阻尼。", file=sys.stderr)
                    elif args.gamepad_device is not None or args.gamepad_index is not None or args.gamepad_deadman:
                        raise ValueError("constant 模式不能指定手柄参数")
                    count = run_live(
                        config, args.output, (args.vx, args.vy, args.wz), duration, armed=args.arm,
                        command_source=gamepad.read if gamepad else None,
                        init_request_source=gamepad.consume_init_request if gamepad else None,
                        status_callback=lambda mode, reason: print(f"状态：{mode} ({reason})", file=sys.stderr),
                    )
                result = {"policy_steps": count, "output": str(args.output)}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("操作已中断，SDK 上下文执行退出处理。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
