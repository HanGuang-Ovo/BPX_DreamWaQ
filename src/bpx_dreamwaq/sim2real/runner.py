"""实机执行和离线回放：仅验证有限数与四元数，不限幅、限速或限矩。"""

import time

import numpy as np

from .config import vector
from .controller import ActorController
from .behavior import HardwareController
from .recording import JsonlWriter, load_samples, resample
from .safety import StateGuard
from .sdk_robot import SDKSource


def replay(config, input_path, output_path, command, *, policies=None):
    metadata, samples = load_samples(input_path)
    first_complete = next((i for i, s in enumerate(samples)
                           if all(v is not None for v in (s.q, s.dq, s.quat_xyzw, s.omega))
                           and (not config.policy["include_base_lin_vel"] or s.velocity_body is not None)), None)
    if first_complete is None:
        raise ValueError("录制没有完整 Actor 状态")
    samples = samples[first_complete:]
    guard = StateGuard(config)
    controller = ActorController(config, policies=policies)
    count = 0
    with JsonlWriter(output_path, {"mode": "replay", "input": str(input_path), "input_metadata": metadata,
                                 "config": str(config.path), "skipped_initial_samples": first_complete,
                                 "feedback": "recorded trajectory; policy targets do not alter recorded states"}) as log:
        for sample in resample(samples, config.period):
            guard.check(sample)
            result = controller.step(sample, command)
            target = vector(result["target_model"], 12, "target")
            controller.commit_target(target)
            result.update(target_model=target.tolist(), target_sdk=controller.calibration.position_to_sdk(target).tolist(),
                          host_time=sample.host_time, sample=sample.to_dict())
            log.write({"type": "policy", **result})
            count += 1
    return count


def wait_for_feedback(source, guard, timeout, *, clock=time.monotonic, sleep=time.sleep):
    deadline = clock() + timeout
    while clock() < deadline:
        sample = source.read()
        needed = [sample.q, sample.dq, sample.quat_xyzw, sample.omega]
        if guard.config.policy["include_base_lin_vel"]:
            needed.append(sample.velocity_body)
        if all(v is not None for v in needed):
            guard.check(sample)
            return sample
        sleep(.005)
    raise TimeoutError("等待首个完整 Actor 反馈超时")


def run_live(config, output_path, command, duration, *, armed=False, policies=None, source_factory=SDKSource,
             clock=time.monotonic, sleep=time.sleep, command_source=None, init_request_source=None,
             status_callback=None):
    # armed 仅兼容旧调用；--live 已经明确选择实机执行。
    if duration is not None and (not np.isfinite(duration) or duration <= 0):
        raise ValueError("duration 必须为有限正数")
    command = vector(command, 3, "command")
    controller = HardwareController(config, policies=policies)
    guard = StateGuard(config)
    cal = controller.calibration
    zeros = np.zeros(12)
    send_period = 1. / config.control["send_rate_hz"]
    with JsonlWriter(output_path, {"mode": "live", "config": str(config.path), "runtime": config.runtime,
                                 "calibration": config.calibration, "duration_s": duration,
                                 "command_source": "dynamic" if command_source is not None else "constant"}) as log:
        with source_factory(config, joint_control=True) as source:
            sample = wait_for_feedback(source, guard, config.network["connect_timeout_s"], clock=clock, sleep=sleep)
            controller.initialize(sample)
            start = next_send = next_policy = clock()
            policy_count = 0
            previous_mode = None
            while True:
                if next_send > clock():
                    sleep(next_send - clock())
                now = clock()
                if duration is not None and now - start + 1e-9 >= duration:
                    break
                sample = source.read()
                guard.check(sample)
                policy_due = now + 1e-9 >= next_policy
                init_requested = False
                if policy_due:
                    if command_source is not None:
                        command = vector(command_source(), 3, "command")
                    init_requested = bool(init_request_source()) if init_request_source is not None else False
                    policy_count += 1
                    next_policy += config.period
                result = controller.step(sample, command, now=now, policy_due=policy_due,
                                         init_requested=init_requested)
                target = vector(result["target_model"], 12, "target")
                source.send(cal.gains_to_sdk(controller.kp), cal.position_to_sdk(target),
                            cal.gains_to_sdk(controller.kd), zeros, zeros)
                controller.commit_target(target)
                after = clock()
                if controller.mode != previous_mode:
                    if status_callback is not None:
                        status_callback(controller.mode, result["reason"])
                    previous_mode = controller.mode
                log.write({"type": "command", "host_time": after, "mode": controller.mode,
                           "raw_command": np.asarray(command).tolist(), "init_requested": init_requested,
                           "target_model": target.tolist(), "target_sdk": cal.position_to_sdk(target).tolist(),
                           "kp_model": controller.kp.tolist(), "kd_model": controller.kd.tolist(),
                           "sample": sample.to_dict(), "policy": result})
                next_send += send_period
                if after > next_send:
                    next_send = after + send_period
                if after > next_policy:
                    next_policy = after + config.period
    return policy_count
