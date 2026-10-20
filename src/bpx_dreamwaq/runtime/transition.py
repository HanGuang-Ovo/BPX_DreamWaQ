"""行为切换过程中使用的连续过渡工具。

上层状态是离散的，但发送给策略的 command 和发送给 PD 的 action 不应离散跳变：

- ``SlewRateLimiter`` 限制 command 每秒最多改变多少；
- ``ActionBlender`` 在策略真正发生切换时平滑混合新旧 action。

两者处理的是不同信号，不能互相替代。只平滑 command 仍可能因为换了神经网络而造成
action 跳变；只混合 action 则无法抑制手柄突然从最大值松到零产生的指令阶跃。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float32]


class SlewRateLimiter:
    """逐分量限制三维速度 command 的变化率。

    ``rates`` 的顺序固定为 ``[vx_rate, vy_rate, wz_rate]``，单位分别是
    ``m/s², m/s², rad/s²``。内部的 ``value`` 是上一策略帧真正输出的 command。
    """

    def __init__(self, rates: tuple[float, float, float]):
        self.rates = np.asarray(rates, dtype=np.float32)
        if self.rates.shape != (3,) or not np.all(np.isfinite(self.rates)) or np.any(self.rates <= 0.0):
            raise ValueError("command slew rates 必须是三个正数")
        self.value = np.zeros(3, dtype=np.float32)

    def reset(self, value: npt.ArrayLike | None = None) -> None:
        """清除过滤器历史；不传值时重置为全零。"""

        new_value = np.zeros(3, dtype=np.float32) if value is None else np.asarray(value, dtype=np.float32)
        if new_value.shape != (3,) or not np.all(np.isfinite(new_value)):
            raise ValueError("command limiter 初值必须为三个有限数")
        self.value[:] = new_value

    def update(self, target: npt.ArrayLike, dt: float) -> FloatArray:
        """向目标靠近一步，同时保证 ``|delta| <= rate * dt``。"""

        target_array = np.asarray(target, dtype=np.float32)
        if target_array.shape != (3,) or not np.all(np.isfinite(target_array)):
            raise ValueError("command limiter 目标必须为三个有限数")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("command limiter 的 dt 必须大于零")
        # np.clip 对三个轴分别裁剪，因此不会因为某一轴变化大而拖慢其他轴。
        maximum_delta = self.rates * dt
        self.value += np.clip(target_array - self.value, -maximum_delta, maximum_delta)
        return self.value.copy()


class ActionBlender:
    """从上一帧实际 action 平滑过渡到新策略输出。

    切换瞬间保存的是“实际施加给机器人”的 action，而不是旧策略此刻重新计算的输出。
    这样即使旧策略已经停止运行，切换起点仍与关节控制器上一帧收到的值连续。

    混合系数使用 smoothstep ``3p²-2p³``，在起点和终点的一阶导数均为零，比线性插值
    更不容易在切换边界制造速度突变。
    """

    def __init__(self, action_dimension: int, duration: float):
        if action_dimension <= 0 or not np.isfinite(duration) or duration < 0.0:
            raise ValueError("action blender 参数无效")
        self.duration = duration
        self.start_action = np.zeros(action_dimension, dtype=np.float32)
        self.elapsed = duration

    @property
    def alpha(self) -> float:
        """返回当前混合系数：0 表示完全旧动作，1 表示完全新动作。"""

        if self.duration == 0.0:
            return 1.0
        progress = float(np.clip(self.elapsed / self.duration, 0.0, 1.0))
        return progress * progress * (3.0 - 2.0 * progress)

    def start(self, applied_action: npt.ArrayLike) -> None:
        """通知混合器策略刚发生切换，并保存连续过渡的起点。"""

        action = np.asarray(applied_action, dtype=np.float32)
        if action.shape != self.start_action.shape or not np.all(np.isfinite(action)):
            raise ValueError("action blender 起始动作维度或数值无效")
        self.start_action[:] = action
        self.elapsed = 0.0

    def update(self, target_action: npt.ArrayLike, dt: float) -> FloatArray:
        """推进混合时间并返回这一策略帧应实际施加的 action。"""

        target = np.asarray(target_action, dtype=np.float32)
        if target.shape != self.start_action.shape or not np.all(np.isfinite(target)):
            raise ValueError("action blender 目标动作维度或数值无效")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("action blender 的 dt 必须大于零")
        self.elapsed = min(self.elapsed + dt, self.duration)
        alpha = self.alpha
        return ((1.0 - alpha) * self.start_action + alpha * target).astype(np.float32)
