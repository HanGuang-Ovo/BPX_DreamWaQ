"""把 Supervisor 的行为模式路由到对应底层策略。

路由层只保存控制器对象和回退关系，不参与状态转换。Stand 可使用 10 帧历史或旧版单帧观测；
RL Init 使用单帧观测，Locomotion 可使用完整历史。所有策略保持 12 维 action、相同关节顺序、默认角和动作缩放，
运行器才能安全地在策略输出之间进行 action 混合。PD Init 读取机器人状态，输出同一动作接口。
"""

from __future__ import annotations

from dataclasses import dataclass

from bpx_dreamwaq.runtime.policy import Policy
from bpx_dreamwaq.runtime.supervisor import BehaviorMode


@dataclass(frozen=True)
class BehaviorPolicies:
    """一组可由上层状态机选择的底层策略。

    ``locomotion`` 是必需的。``stand``、``init`` 和 ``waiting`` 允许逐步加入：没有独立站立策略
    时，STAND 会回退到 locomotion policy，并向它发送零 command；Init 控制器不存在时，
    Supervisor 不会进入 INIT，而是选择 DISABLED。
    """

    locomotion: Policy
    stand: Policy | None = None
    init: Policy | None = None
    waiting: Policy | None = None

    def resolve(self, mode: BehaviorMode) -> Policy:
        """返回某个行为真正应该执行的策略对象。"""

        if mode == BehaviorMode.STAND:
            return self.stand or self.locomotion
        if mode in (BehaviorMode.WALK, BehaviorMode.STOPPING):
            return self.locomotion
        if mode == BehaviorMode.INIT:
            if self.init is None:
                raise RuntimeError("进入 INIT，但没有配置 init policy")
            return self.init
        if mode == BehaviorMode.WAITING_INIT:
            if self.waiting is None:
                raise RuntimeError("进入 WAITING_INIT，但没有配置趴姿保持策略")
            return self.waiting
        raise RuntimeError(f"{mode.value} 状态不应执行底层策略")

    def policy_name(self, mode: BehaviorMode) -> str:
        """返回用于终端和 CSV 的可读策略名称。"""

        if mode == BehaviorMode.STAND:
            return "stand" if self.stand is not None else "locomotion_fallback"
        if mode in (BehaviorMode.WALK, BehaviorMode.STOPPING):
            return "locomotion"
        if mode == BehaviorMode.INIT:
            return getattr(self.init, "behavior_name", "init")
        if mode == BehaviorMode.WAITING_INIT:
            return "prone_hold"
        return "none"
