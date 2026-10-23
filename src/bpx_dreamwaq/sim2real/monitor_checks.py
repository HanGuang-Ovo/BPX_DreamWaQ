"""启动前只读检查结果；不包含机器人运动或标零操作。"""

import threading

import numpy as np

from .preflight import check


class PreflightWorker(threading.Thread):
    """模型加载和预热在后台执行，通过独立队列回传，避免阻塞 Tk。"""

    def __init__(self, config, messages, *, no_policy=False):
        super().__init__(name="bpx-offline-preflight", daemon=False)
        self.config, self.messages, self.no_policy = config, messages, no_policy

    def run(self):
        try:
            result = check(self.config, load_model=not self.no_policy)
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        self.messages.put(result)


def startup_rows(config, snapshot=None, report=None, *, active=False, elapsed=0., no_policy=False):
    """分别显示软件和当前反馈状态，不把 DEMO / 旧帧解释为实机就绪。"""
    rows = []

    def add(key, label, status, detail):
        rows.append((key, label, status, detail))

    source = snapshot["source"] if snapshot else "未连接"
    sample = snapshot["sample"] if snapshot else None
    stale_after = config.monitor.get("display_stale_after_s", .10)
    current = bool(sample is not None and active and sample.connected and elapsed <= stale_after)
    source_status = source if current else "反馈待更新" if active and sample else "未连接 / 会话结束"
    add("source", "数据来源", source_status, source)
    if no_policy:
        add("model", "模型检查", "已关闭", "--no-policy：不加载权重，仅检查反馈与姿态")
    elif report is None:
        add("model", "模型检查", "检查中", str(config.model_path()))
    elif report.get("error"):
        add("model", "模型检查", "异常", report["error"])
    elif report.get("model_checked"):
        model = report["models"]["policy"]
        add("model", "模型检查", "通过",
            f"{model['input_shape']} → {model['output_shape']} float32；"
            f"预热 {model['warmup_ms']:.2f} ms；SHA256 {model['sha256']}")
    else:
        add("model", "模型检查", "异常", "；".join(report.get("replay_blockers", [])) or "模型尚未校验")
    dependencies = (report or {}).get("dependencies", {})
    sdk = dependencies.get("bpx_sdk")
    add("sdk", "SDK 依赖", "已安装" if sdk else "未安装" if sdk is False else "检查中",
        "只读会话仅使用 RequestRobotState；DEMO / 回放不连接机器人")
    add("contract", "观测与控制契约", "已读取",
        f"{config.policy['observation_dimension']} 维 / H={config.policy['history_length']}；"
        f"策略 {config.control['policy_rate_hz']} Hz / 发送 {config.control['send_rate_hz']} Hz（本窗口不发送）")
    add("mapping", "策略 → SDK 通道", "已读取", str([j.sdk_index for j in config.joints]))

    def fresh(group):
        age = getattr(sample, f"{group}_age_s") if sample else None
        if not current:
            return False, "等待当前只读会话的有效反馈"
        if age is None:
            return False, "尚未观察到数据更新"
        age = age + elapsed
        return age <= stale_after, f"数据年龄 {age * 1000:.0f} ms；提示阈值 {stale_after * 1000:.0f} ms"

    valid, detail = fresh("joint")
    complete = bool(snapshot and len(snapshot["rows"]) == 12 and all(
        row["q_model"] is not None and row["dq_model"] is not None
        and np.isfinite([row["q_model"], row["dq_model"]]).all() for row in snapshot["rows"]))
    add("joint", "12 关节反馈", "有效" if valid and complete else "待检查", detail)
    valid, detail = fresh("imu")
    attitude = snapshot.get("attitude") if snapshot else None
    complete = bool(attitude and "omega_body" in attitude)
    if complete:
        detail += f"；|q|={attitude['norm']:.4f}；RPY[°]=" + str([round(v, 1) for v in attitude["body_rpy_deg"]])
    add("imu", "IMU 与姿态", "有效" if valid and complete else "待检查", detail)
    control = snapshot.get("control", {}) if snapshot else {}
    command = control.get("remote_command")
    valid_command = command is not None and np.shape(command) == (3,) and np.isfinite(command).all()
    add("gamepad", "电脑手柄", "有效" if current and valid_command else "待检查",
        f"vx/vy/wz={command}；仅预览，不发送" if current and valid_command else
        control.get("input_source", "等待电脑手柄；DEMO 使用模拟输入"))
    errors = snapshot.get("policy_errors", {}) if snapshot else {}
    add("preview", "策略预览", "已关闭" if no_policy else
        "有效" if current and snapshot.get("policies") else "待检查",
        "；".join(errors.values()) or control.get("reason", "连接反馈后，按固件状态和手柄输入自动预览"))
    return rows
