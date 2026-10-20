"""可回放的实机 Actor 管线：观测、共享历史、统一策略、接管与动作混合。"""

import numpy as np

from bpx_dreamwaq.runtime.history import ObservationHistory
from bpx_dreamwaq.runtime.observation import actor_frame
from bpx_dreamwaq.runtime.policy import OnnxPolicy
from bpx_dreamwaq.runtime.transition import ActionBlender
from bpx_dreamwaq.runtime.routing import PolicyRouter
from .calibration import Calibration


class ActorController:
    def __init__(self, config, *, policy=None):
        self.config = config
        self.calibration = Calibration(config)
        self.policy = policy if policy is not None else OnnxPolicy(
            config.model_path(), config.policy["observation_dimension"], 12
        )
        session = getattr(self.policy, "session", None)
        if session is not None:
            inputs, outputs = session.get_inputs(), session.get_outputs()
            if (
                len(inputs) != 1
                or len(outputs) != 1
                or inputs[0].shape != [1, config.policy["observation_dimension"]]
                or outputs[0].shape != [1, 12]
            ):
                raise ValueError("模型的输入输出尺寸不符合实机配置")
        settings = config.policy
        self.history = ObservationHistory(
            settings["history_length"], include_base_lin_vel=settings["include_base_lin_vel"]
        )
        self.router = PolicyRouter(config.control, config.period)
        self.blender = ActionBlender(12, config.control["action_blend_duration_s"])
        self.last_action = None
        self.initial_target = None
        self.started_at = None
        self.mode = "stand"

    def _route(self, command):
        filtered = self.router.step(command)
        self.mode = self.router.mode
        return filtered

    def step(self, sample, command, *, decision=None):
        cal = self.calibration
        q = cal.position_from_sdk(sample.q)
        dq = cal.velocity_from_sdk(sample.dq)
        gravity, angular = cal.imu(sample.quat_xyzw, sample.omega)
        if self.started_at is None:
            self.started_at = sample.host_time
            self.initial_target = q.copy()
            self.last_action = ((q - cal.default) / self.config.policy["action_scale"]).astype(np.float32)
            self.blender.start(self.last_action)
        command = np.asarray(command, dtype=float)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("command 必须为三个有限值")
        command_limit = np.asarray(self.config.control["command_limit"])
        command = np.clip(command, -command_limit, command_limit)
        if decision is None:
            filtered = self._route(command)
        else:
            filtered = np.asarray(decision.command, dtype=float)
            if filtered.shape != (3,) or not np.isfinite(filtered).all():
                raise ValueError("supervisor command 必须为三个有限值")
            # 外部 decision 也必须遵守 Actor 的最终 command 范围。
            filtered = np.clip(filtered, -command_limit, command_limit)
            self.mode = decision.mode.value
        linear = sample.velocity_body if self.config.policy["include_base_lin_vel"] else None
        if self.mode == "stand":
            filtered = np.zeros(3, dtype=np.float32)
        frame = actor_frame(angular, gravity, filtered, q, dq, self.last_action, cal.default, linear)
        observation = self.history.append(frame)
        output = np.asarray(self.policy(observation), dtype=np.float32)
        if output.shape != (12,) or not np.isfinite(output).all():
            raise RuntimeError("Actor 输出无效")
        blended = self.blender.update(output, self.config.period)
        target = cal.default + self.config.policy["action_scale"] * blended
        # 接管插值从实测姿态开始，不增加目标范围、限速或限矩处理。
        progress = np.clip((sample.host_time - self.started_at) / self.config.control["takeover_duration_s"], 0.0, 1.0)
        alpha = progress * progress * (3 - 2 * progress)
        target = self.initial_target + alpha * (target - self.initial_target)
        return {
            "mode": self.mode,
            "command": filtered.tolist(),
            "observation": observation.tolist(),
            "raw_action": output.tolist(),
            "target_model": target.tolist(),
            "q_model": q.tolist(),
            "dq_model": dq.tolist(),
            "takeover_alpha": float(alpha),
        }

    def commit_target(self, target):
        # 下一周期观测用成功发送（或离线虚拟发送）的最终目标所对应动作。
        self.last_action = (
            (np.asarray(target) - self.calibration.default) / self.config.policy["action_scale"]
        ).astype(np.float32)
