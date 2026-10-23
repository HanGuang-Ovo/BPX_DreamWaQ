"""部署前验证模型接口并预热；仅使用名义本体观测，不创建 SDK 对象。"""

import time

import numpy as np

from bpx_dreamwaq.runtime.history import ObservationHistory
from bpx_dreamwaq.runtime.observation import actor_frame
from .config import vector


def validate_model(policy, config):
    session = getattr(policy, "session", None)
    if session is None:
        return
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or len(outputs) != 1
            or inputs[0].shape != [1, config.policy["observation_dimension"]]
            or outputs[0].shape != [1, 12]):
        raise ValueError("模型的输入输出尺寸不符合实机配置：要求 [1, observation_dimension] → [1, 12]")
    if inputs[0].type != "tensor(float)" or outputs[0].type != "tensor(float)":
        raise ValueError("实机 ONNX 输入和输出必须为 float32")


def warmup_policy(policy, config):
    """独立历史上的名义站姿试推理，不写入控制器的历史或上一帧动作。"""
    validate_model(policy, config)
    settings = config.policy
    zeros = np.zeros(12, dtype=np.float32)
    frame = actor_frame(
        [0., 0., 0.], [0., 0., -1.], [0., 0., 0.],
        settings["default_joint_position"], zeros, zeros, settings["default_joint_position"],
        [0., 0., 0.] if settings["include_base_lin_vel"] else None,
    )
    history = ObservationHistory(settings["history_length"],
                                 include_base_lin_vel=settings["include_base_lin_vel"])
    start = time.perf_counter()
    action = vector(policy(history.append(frame)), 12, "预热 Actor 输出")
    elapsed = time.perf_counter() - start
    return {
        "input_shape": [1, settings["observation_dimension"]],
        "output_shape": [1, 12],
        "dtype": "float32",
        "warmup_ms": elapsed * 1000.,
        "nominal_action": action.tolist(),
    }
