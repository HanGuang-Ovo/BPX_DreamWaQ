"""只读监控入口：窗口启动、参数解析和手柄枚举。"""

import argparse
from pathlib import Path

from .config import load_config, positive
from .monitor_window import MonitorWindow
from .monitor_worker import MonitorWorker
from .monitor_geometry import leg_points

# Public inspection helpers are also exported for offline diagnostics.
__all__ = ["main", "launch", "MonitorWindow", "MonitorWorker", "leg_points"]


def launch(
    config,
    *,
    demo=False,
    recording=None,
    no_policy=False,
    output=None,
    duration=None,
    gamepad_device=None,
    gamepad_index=None,
):
    if demo and recording:
        raise ValueError("--demo 与 --recording 不能同时使用")
    if duration is not None:
        positive(duration, "duration")
    try:
        import tkinter as tk

        root = tk.Tk()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("窗口需要 Tkinter 和桌面显示；在 bpx-real 环境中检查 python -m tkinter") from exc
    except tk.TclError as exc:
        raise RuntimeError("无法打开桌面窗口；请在有图形桌面的本机终端执行") from exc
    window = MonitorWindow(
        root,
        config,
        demo=demo,
        recording=recording,
        no_policy=no_policy,
        output=output,
        gamepad_device=gamepad_device,
        gamepad_index=gamepad_index,
    )
    if duration is not None:
        root.after(int(duration * 1000), window.close)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        window.close()
        root.mainloop()
    return 0


def main(argv=None):
    from .cli import DEFAULT_CONFIG

    local = DEFAULT_CONFIG.with_name("local.runtime.toml")
    parser = argparse.ArgumentParser(description="BPX 只读反馈、策略输出与姿态对齐窗口")
    parser.add_argument("--config", type=Path, default=local if local.exists() else DEFAULT_CONFIG)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--no-policy", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--duration", type=float)
    add_gamepad_arguments(parser)
    args = parser.parse_args(argv)
    try:
        if args.list_gamepads:
            return list_gamepads()
        return launch(
            load_config(args.config),
            demo=args.demo,
            recording=args.recording,
            no_policy=args.no_policy,
            output=args.output,
            duration=args.duration,
            gamepad_device=args.gamepad_device,
            gamepad_index=args.gamepad_index,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


def add_gamepad_arguments(parser):
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--gamepad-device", type=Path, help="只读手柄设备，例如 /dev/input/eventN")
    selector.add_argument("--gamepad-index", type=int, help="多个手柄时指定列表索引")
    parser.add_argument("--list-gamepads", action="store_true", help="列出电脑手柄，不连接机器人或打开窗口")


def list_gamepads():
    from bpx_dreamwaq.runtime.gamepad import enumerate_gamepads

    devices = enumerate_gamepads()
    for index, device in enumerate(devices):
        print(f"[{index}] {device.label}")
    if not devices:
        print("没有找到电脑 USB/蓝牙手柄")
    return 0
