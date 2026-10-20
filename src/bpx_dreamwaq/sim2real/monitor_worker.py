"""只读监控采样线程：反馈采集、录制回放、策略预览和有界队列发布。"""

from contextlib import ExitStack
from dataclasses import replace
import queue
import threading
import time

from .diagnostics import PolicyPreview, build_snapshot, demo_sample
from .recording import JsonlWriter, load_samples, resample
from .sdk_robot import SDKSource
from .monitor_input import MonitorGamepad


class MonitorWorker(threading.Thread):
    """在单个线程中管理只读会话，向 UI 发布最新快照和会话状态。"""

    def __init__(
        self,
        config,
        messages,
        *,
        demo=False,
        recording=None,
        no_policy=False,
        output=None,
        source_factory=SDKSource,
        policies=None,
        gamepad_device=None,
        gamepad_index=None,
        gamepad_factory=MonitorGamepad,
    ):
        super().__init__(name="bpx-readonly-monitor", daemon=False)
        self.config, self.messages = config, messages
        self.demo, self.recording, self.no_policy = demo, recording, no_policy
        self.output, self.source_factory, self.policies = output, source_factory, policies
        self.stop_event = threading.Event()
        self.gamepad_device, self.gamepad_index = gamepad_device, gamepad_index
        self.gamepad_factory = gamepad_factory

    def publish(self, kind, value):
        # 有界队列：慢速绘图不会积累过时反馈。
        try:
            self.messages.put_nowait((kind, value))
        except queue.Full:
            try:
                self.messages.get_nowait()
            except queue.Empty:
                pass
            self.messages.put_nowait((kind, value))

    def run(self):
        try:
            self.publish("status", "正在加载策略；SDK 尚未连接…")
            probe = PolicyPreview(self.config, policies=self.policies, enabled=not self.no_policy)
            with ExitStack() as stack:
                writer = (
                    stack.enter_context(
                        JsonlWriter(
                            self.output,
                            {
                                "mode": "monitor",
                                "config": str(self.config.path),
                                "runtime": self.config.runtime,
                                "calibration": self.config.calibration,
                                "source": "demo" if self.demo else "recording" if self.recording else "sdk_read_only",
                                "note": "RequestRobotState only; policy preview never sends commands",
                            },
                        )
                    )
                    if self.output
                    else None
                )
                recorded = None
                source = None
                gamepad = None
                if self.recording:
                    _, samples = load_samples(self.recording)
                    if not samples:
                        raise ValueError("录制没有 sample 数据")
                    recorded = iter(resample(samples, self.config.period))
                elif not self.demo:
                    if self.stop_event.is_set():
                        return
                    self.publish("status", "正在建立只读 SDK 连接…")
                    source = stack.enter_context(self.source_factory(self.config, joint_control=False))
                    gamepad = stack.enter_context(
                        self.gamepad_factory(
                            self.config,
                            device=self.gamepad_device,
                            index=self.gamepad_index,
                        )
                    )
                start = time.monotonic()
                next_tick = start
                last_time = None
                session = None
                tokens, updates = {}, {"joint": 0, "imu": 0}
                tick = 0
                while not self.stop_event.is_set():
                    if self.stop_event.wait(max(0.0, next_tick - time.monotonic())):
                        break
                    now = time.monotonic()
                    if self.demo:
                        sample = demo_sample(self.config, now - start, now, tick)
                    elif recorded is not None:
                        try:
                            sample = next(recorded)
                        except StopIteration:
                            self.publish("status", "录制播放结束；最后一帧已停用策略预览")
                            break
                    else:
                        sample = source.read()
                        sample = replace(sample, telemetry={**sample.telemetry, **gamepad.read()})
                    discontinuity = session != sample.session or not sample.connected
                    if discontinuity or (last_time is not None and now - last_time > self.config.period * 1.5):
                        probe.reset()
                    if discontinuity:
                        tokens, updates = {}, {"joint": 0, "imu": 0}
                    session, last_time = sample.session, now
                    for group in ("joint", "imu"):
                        token = getattr(sample, f"{group}_token")
                        if token is not None and token != tokens.get(group):
                            updates[group] += 1
                            tokens[group] = token
                    snapshot = build_snapshot(
                        self.config,
                        sample,
                        probe,
                        ready=self.demo or self.recording is not None or min(updates.values()) >= 2,
                    )
                    snapshot["received_at"] = time.monotonic()
                    snapshot["source"] = (
                        "DEMO · 模拟反馈" if self.demo else "录制回放" if self.recording else "SDK · 只读反馈"
                    )
                    snapshot["model_info"] = {m: str(self.config.model_path(m)) for m in ("stand", "walk")}
                    self.publish("snapshot", snapshot)
                    if writer:
                        writer.write({
                            "type": "sample",
                            "sample": sample.to_dict(),
                            "preview": snapshot["policies"],
                            "control": snapshot["control"],
                            "warnings": snapshot["warnings"],
                        })
                    tick += 1
                    next_tick += self.config.period
                    # 不补跑历史；推理卡顿后重建历史，等待下一时刻。
                    if time.monotonic() > next_tick + self.config.period * 0.5:
                        probe.reset()
                        next_tick = time.monotonic() + self.config.period
        except Exception as exc:
            self.publish("error", f"{type(exc).__name__}: {exc}")
        finally:
            self.publish("stopped", "只读会话已结束")
