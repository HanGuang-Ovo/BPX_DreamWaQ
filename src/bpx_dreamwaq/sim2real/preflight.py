"""CLI 与只读窗口共用的软件预检；不导入或实例化机器人 SDK。"""

import hashlib
import importlib.util

from bpx_dreamwaq.runtime.joints import SDK_JOINT_NAMES
from bpx_dreamwaq.runtime.policy import OnnxPolicy
from .inference import warmup_policy


def check(config, *, load_model=True):
    p = config.model_path()
    models = {"policy": {
        "path": str(p),
        "exists": p.is_file(),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None,
    }}
    dependencies = {name: importlib.util.find_spec(name) is not None for name in ("numpy", "onnxruntime", "bpx_sdk")}
    model_blockers = config.blockers("deploy")
    if not dependencies["onnxruntime"]:
        model_blockers.append("当前 Python 未安装 onnxruntime")
    if load_model and not model_blockers:
        try:
            policy = OnnxPolicy(p, config.policy["observation_dimension"], 12)
            models["policy"].update(warmup_policy(policy, config))
        except Exception as exc:
            models["policy"]["error"] = str(exc)
            model_blockers.append(f"策略加载或试推理失败：{exc}")
    observe_blockers = [] if dependencies["bpx_sdk"] else ["当前 Python 未安装 bpx_sdk"]
    deploy_blockers = model_blockers + observe_blockers
    return {
        "config": str(config.path),
        "calibration": str(config.calibration_path),
        "dependencies": dependencies,
        "models": models,
        "model_checked": load_model and not model_blockers,
        "observe_blockers": observe_blockers,
        "replay_blockers": model_blockers,
        "deploy_blockers": deploy_blockers,
        "ready": load_model and not deploy_blockers,
        "timing": {"policy_rate_hz": config.control["policy_rate_hz"],
                   "send_rate_hz": config.control["send_rate_hz"],
                   "history_length": config.policy["history_length"]},
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
