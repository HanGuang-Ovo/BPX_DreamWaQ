"""Read a Linux gamepad for BPX simulation, deployment or read-only monitoring.

This module intentionally uses the kernel ``input-event`` interface directly,
so the runtime does not need python-evdev or hidapi. It never sends robot commands.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import struct
import sys

import numpy as np
import numpy.typing as npt

EV_KEY = 0x01
EV_ABS = 0x03
EV_SYN = 0x00
SYN_DROPPED = 3
EVENT_STRUCT = struct.Struct("llHHi")
ABS_INFO_STRUCT = struct.Struct("iiiiii")

# Standard Linux gamepad axis codes.
ABS_X = 0
ABS_Y = 1
ABS_RX = 3
BTN_TL = 310
BTN_TR = 311
KEY_STATE_BYTES = 64


@dataclass(frozen=True)
class InputDevice:
    path: Path
    name: str
    vendor_id: int
    product_id: int
    physical_path: str

    @property
    def label(self) -> str:
        return (
            f"VID=0x{self.vendor_id:04X} PID=0x{self.product_id:04X}  "
            f"{self.name}  node={self.path} phys={self.physical_path}"
        )


@dataclass(frozen=True)
class AxisCalibration:
    minimum: int
    maximum: int
    flat: int

    def normalize(self, value: int, deadzone: float) -> float:
        """Map an asymmetric Linux axis range to [-1, 1] with a deadzone."""

        center = 0.5 * (self.minimum + self.maximum)
        span = self.maximum - center if value >= center else center - self.minimum
        normalized = 0.0 if span <= 0.0 else (value - center) / span
        normalized = float(np.clip(normalized, -1.0, 1.0))

        # Some drivers report a useful hardware deadzone through ``flat``.
        hardware_deadzone = 0.0
        full_span = 0.5 * (self.maximum - self.minimum)
        if full_span > 0.0:
            hardware_deadzone = self.flat / full_span
        effective_deadzone = min(max(deadzone, hardware_deadzone), 0.99)
        magnitude = abs(normalized)
        if magnitude <= effective_deadzone:
            return 0.0
        scaled = (magnitude - effective_deadzone) / (1.0 - effective_deadzone)
        return float(np.copysign(scaled, normalized))


@dataclass(frozen=True)
class GamepadMapping:
    """Map Linux absolute axes to ``[vx, vy, wz]`` commands."""

    vx_axis: int = ABS_Y
    vy_axis: int = ABS_X
    wz_axis: int = ABS_RX
    vx_sign: float = -1.0
    vy_sign: float = -1.0
    wz_sign: float = -1.0

    @property
    def axes(self) -> tuple[int, int, int]:
        return (self.vx_axis, self.vy_axis, self.wz_axis)

    @property
    def signs(self) -> tuple[float, float, float]:
        return (self.vx_sign, self.vy_sign, self.wz_sign)


def _read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return default


def _read_hex_bitmap(path: Path) -> int:
    try:
        words = _read_text(path, "0").split()
        bits_per_word = struct.calcsize("L") * 8
        result = 0
        for word in words:
            result = (result << bits_per_word) | int(word, 16)
        return result
    except ValueError:
        return 0


def enumerate_gamepads() -> list[InputDevice]:
    """Return Linux input-event nodes that expose gamepad axes and buttons."""

    if not sys.platform.startswith("linux"):
        return []
    devices: list[InputDevice] = []
    event_directories = sorted(
        Path("/sys/class/input").glob("event*"),
        key=lambda item: int(item.name.removeprefix("event")),
    )
    for event_directory in event_directories:
        info_directory = event_directory / "device"
        event_bits = _read_hex_bitmap(info_directory / "capabilities/ev")
        key_bits = _read_hex_bitmap(info_directory / "capabilities/key")
        abs_bits = _read_hex_bitmap(info_directory / "capabilities/abs")
        has_absolute_axes = bool(event_bits & (1 << EV_ABS))
        has_stick_or_dpad = any(abs_bits & (1 << code) for code in (0, 1, 3, 4, 16, 17))
        has_gamepad_button = any(key_bits & (1 << code) for code in range(288, 319))
        if not (has_absolute_axes and has_stick_or_dpad and has_gamepad_button):
            continue
        devices.append(
            InputDevice(
                path=Path("/dev/input") / event_directory.name,
                name=_read_text(info_directory / "name", "unknown device"),
                vendor_id=int(_read_text(info_directory / "id/vendor", "0"), 16),
                product_id=int(_read_text(info_directory / "id/product", "0"), 16),
                physical_path=_read_text(info_directory / "phys", "-"),
            )
        )
    return devices


def select_gamepad(device_path: Path | None, index: int | None) -> InputDevice:
    """Resolve an explicit event path or one of the discovered gamepads."""

    devices = enumerate_gamepads()
    if device_path is not None:
        requested = device_path.expanduser().resolve()
        return next(
            (device for device in devices if device.path.resolve() == requested),
            InputDevice(requested, "specified input device", 0, 0, "-"),
        )
    if not devices:
        raise RuntimeError(
            "没有找到 Linux input 游戏手柄；请连接手柄，或用 --gamepad-device /dev/input/eventN 显式指定"
        )
    if index is not None:
        if not 0 <= index < len(devices):
            raise ValueError(f"gamepad index {index} 超出范围 0..{len(devices) - 1}")
        return devices[index]
    if len(devices) > 1:
        choices = "\n".join(f"  [{i}] {device.label}" for i, device in enumerate(devices))
        raise RuntimeError(f"找到多个手柄，请用 --gamepad-index 选择：\n{choices}")
    return devices[0]


def _ioc_read(number: int, size: int) -> int:
    # Linux _IOR('E', number, size), valid for the input-event ABI.
    return (2 << 30) | (size << 16) | (ord("E") << 8) | number


def _read_axis_info(file_descriptor: int, axis: int) -> tuple[int, AxisCalibration]:
    buffer = bytearray(ABS_INFO_STRUCT.size)
    try:
        fcntl.ioctl(file_descriptor, _ioc_read(0x40 + axis, len(buffer)), buffer, True)
    except OSError as exc:
        raise ValueError(f"手柄不支持 ABS 轴 {axis}") from exc
    value, minimum, maximum, _fuzz, flat, _resolution = ABS_INFO_STRUCT.unpack(buffer)
    if maximum <= minimum:
        raise ValueError(f"手柄 ABS 轴 {axis} 的范围无效：[{minimum}, {maximum}]")
    return value, AxisCalibration(minimum=minimum, maximum=maximum, flat=flat)


def _button_is_pressed(file_descriptor: int, button: int) -> bool:
    if button < 0 or button >= KEY_STATE_BYTES * 8:
        raise ValueError(f"手柄按钮码必须在 0..{KEY_STATE_BYTES * 8 - 1} 范围内")
    buffer = bytearray(KEY_STATE_BYTES)
    fcntl.ioctl(file_descriptor, _ioc_read(0x18, len(buffer)), buffer, True)
    return bool(buffer[button // 8] & (1 << (button % 8)))


class GamepadCommandSource:
    """Non-blocking gamepad source sampled by the policy loop."""

    def __init__(
        self,
        device: InputDevice,
        maximum_command: tuple[float, float, float],
        deadzone: float = 0.1,
        mapping: GamepadMapping | None = None,
        deadman_button: int | None = None,
        init_button: int | None = None,
    ):
        if not 0.0 <= deadzone < 1.0:
            raise ValueError("手柄死区必须在 [0, 1) 范围内")
        maximum = np.asarray(maximum_command, dtype=np.float32)
        if maximum.shape != (3,) or not np.all(np.isfinite(maximum)) or np.any(maximum < 0.0):
            raise ValueError("手柄最大速度必须是三个有限的非负数 [vx, vy, wz]")
        self.device = device
        self.maximum_command = maximum
        self.deadzone = deadzone
        self.mapping = mapping or GamepadMapping()
        self.deadman_button = deadman_button
        self.init_button = init_button
        self._file_descriptor: int | None = None
        self._pending = b""
        self._values: dict[int, int] = {}
        self._calibrations: dict[int, AxisCalibration] = {}
        self._buttons: dict[int, bool] = {}
        self._pressed_edges: set[int] = set()

    def __enter__(self) -> "GamepadCommandSource":
        if not sys.platform.startswith("linux"):
            raise RuntimeError("当前手柄 command source 仅支持 Linux input-event")
        try:
            self._file_descriptor = os.open(self.device.path, os.O_RDONLY | os.O_NONBLOCK)
            self._pressed_edges.clear()
            for axis in set(self.mapping.axes):
                value, calibration = _read_axis_info(self._file_descriptor, axis)
                self._values[axis] = value
                self._calibrations[axis] = calibration
            for button in {self.deadman_button, self.init_button} - {None}:
                self._buttons[button] = _button_is_pressed(self._file_descriptor, button)
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self._file_descriptor is not None:
            os.close(self._file_descriptor)
            self._file_descriptor = None

    def _drain_events(self) -> None:
        if self._file_descriptor is None:
            raise RuntimeError("手柄 command source 尚未打开")
        while True:
            try:
                packet = os.read(self._file_descriptor, EVENT_STRUCT.size * 64)
            except BlockingIOError:
                break
            if not packet:
                raise OSError(f"手柄已断开：{self.device.path}")
            self._pending += packet
            while len(self._pending) >= EVENT_STRUCT.size:
                raw_event, self._pending = (
                    self._pending[: EVENT_STRUCT.size],
                    self._pending[EVENT_STRUCT.size :],
                )
                _seconds, _microseconds, event_type, code, value = EVENT_STRUCT.unpack(raw_event)
                if event_type == EV_SYN and code == SYN_DROPPED:
                    raise OSError("手柄事件丢失（SYN_DROPPED）；需重新检查输入")
                if event_type == EV_ABS and code in self._values:
                    self._values[code] = value
                elif event_type == EV_KEY:
                    pressed = value != 0
                    if pressed and not self._buttons.get(code, False):
                        self._pressed_edges.add(code)
                    self._buttons[code] = pressed

    def read(self) -> npt.NDArray[np.float32]:
        """Return the latest ``[vx, vy, wz]`` without blocking simulation."""

        self._drain_events()
        if self.deadman_button is not None and not self._buttons.get(self.deadman_button, False):
            return np.zeros(3, dtype=np.float32)
        normalized = np.asarray(
            [self._calibrations[axis].normalize(self._values[axis], self.deadzone) for axis in self.mapping.axes],
            dtype=np.float32,
        )
        signs = np.asarray(self.mapping.signs, dtype=np.float32)
        return normalized * signs * self.maximum_command

    def consume_init_request(self) -> bool:
        """RB 每次由松开变为按下时返回一次 True。应在 ``read`` 之后调用。"""

        if self.init_button is None or self.init_button not in self._pressed_edges:
            return False
        self._pressed_edges.remove(self.init_button)
        return True
