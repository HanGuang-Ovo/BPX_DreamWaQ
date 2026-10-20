"""只读电脑 USB/蓝牙手柄，缺失和断开不能伪装成零输入。"""

import time

from bpx_dreamwaq.runtime.gamepad import GamepadCommandSource, select_gamepad


class MonitorGamepad:
    def __init__(
        self,
        config,
        *,
        device=None,
        index=None,
        clock=time.monotonic,
        selector=select_gamepad,
        factory=GamepadCommandSource,
    ):
        self.config, self.device, self.index = config, device, index
        self.clock, self.selector, self.factory = clock, selector, factory
        self.source = None
        self.retry_at = 0.0
        self.error = "等待电脑手柄"

    def read(self):
        if self.source is None and self.clock() >= self.retry_at:
            try:
                device = self.selector(self.device, self.index)
                source = self.factory(device, tuple(self.config.control["command_limit"]))
                source.__enter__()
                self.source = source
                self.error = ""
            except (OSError, ValueError, RuntimeError) as exc:
                self.error = str(exc)
                self.retry_at = self.clock() + 1.0
        command = None
        if self.source is not None:
            try:
                command = self.source.read().tolist()
            except (OSError, ValueError, RuntimeError) as exc:
                self.error = str(exc)
                self.close()
                self.retry_at = self.clock() + 1.0
        return {
            "remote_command": command,
            "remote_command_source": f"电脑手柄 · {self.source.device.name}" if self.source else "电脑手柄未就绪",
            "remote_command_error": self.error,
        }

    def close(self):
        if self.source is not None:
            self.source.close()
            self.source = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
