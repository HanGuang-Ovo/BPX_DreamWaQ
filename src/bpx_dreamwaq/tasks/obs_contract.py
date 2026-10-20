"""训练与部署共用的观测契约常量。

本模块只依赖 Python 标准库，训练侧（Isaac Lab）与导出/部署侧（ONNX、TorchScript）
共用同一组维度定义，保证两边观测布局一致：

- 单帧策略观测按固定顺序拼接：IMU 角速度、投影重力、速度指令、相对关节角、
  关节速度、上一时刻动作。
- 历史观测按「观测项分组」排列：每个观测项连续存放自己的 H 帧，项内从最旧到
  最新。该布局与 Isaac Lab ``ObservationGroupCfg(history_length=H,
  flatten_history_dim=True)`` 的展平结果一致，也与部署端
  ``bpx_dreamwaq.runtime.history.ObservationHistory`` 的输出一致。
- Actor 不输入仿真器提供的真实机体线速度；真实速度仅进入 Critic 特权观测和
  CENet 速度估计的训练标签。
"""

from __future__ import annotations

from typing import Tuple

# 每个观测项的单帧宽度，顺序即拼接顺序（与 runtime.observation.actor_frame 一致）。
POLICY_OBS_TERM_WIDTHS: Tuple[int, ...] = (3, 3, 3, 12, 12, 12)
# 各观测项名称，仅用于文档与调试输出。
POLICY_OBS_TERM_NAMES: Tuple[str, ...] = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
)

# 默认历史长度（策略 50 Hz，对应 0.2 s 历史窗口）。
DEFAULT_HISTORY_LENGTH: int = 10

# 动作维度：12 个关节位置目标。
ACTION_DIM: int = 12

# 机体线速度维度（CENet 速度估计头输出与真实速度标签共用）。
BASE_LIN_VEL_DIM: int = 3


def single_obs_dim() -> int:
    """单帧策略观测维度。"""
    return sum(POLICY_OBS_TERM_WIDTHS)


def history_obs_dim(history_length: int = DEFAULT_HISTORY_LENGTH) -> int:
    """展平后的历史观测维度。"""
    if history_length < 1:
        raise ValueError(f"history_length 必须为正数，当前为 {history_length}")
    return single_obs_dim() * history_length


def newest_frame_slices(history_length: int = DEFAULT_HISTORY_LENGTH):
    """返回从展平历史中提取「最新一帧」各项切片的 (start, end) 列表。

    展平布局为 term-major：第 i 项占 ``[i*H*d_i, (i+1)*H*d_i)``，
    其中该项内部从最旧到最新排列，因此最新一帧是该区间的最后 ``d_i`` 个元素。
    """
    slices = []
    offset = 0
    for width in POLICY_OBS_TERM_WIDTHS:
        block = width * history_length
        slices.append((offset + block - width, offset + block))
        offset += block
    return slices
