# 偶发仿真异常隔离

训练默认在终止管理器中检查机器人状态，时机是每个策略步的物理子步完成后、奖励计算和自动 reset 之前。
诊断阈值为机身世界系单轴线速度 20 m/s、单轴角速度 50 rad/s、关节角速度 100 rad/s；关键状态或施加力矩出现 NaN/Inf 也会触发。
这些阈值不改变正常 PD 控制，也不是电机物理限速。

触发后只终止并重置异常环境，正常环境继续：

- 异常环境当步的各项原始奖励先置零，再进入加权和回合累计；这次异常回合不进入奖励均值日志和 Runner 回合奖励统计。
- 地形升级/降级、逐等级回合统计和指令网格更新跳过这次异常回合。
- `extras["numerical_invalid"]` 将异常标记传至 RolloutStorage；该环境当前 rollout 内、从上次回合结束之后到异常步的片段全部排除。
- 被排除片段不参与 GAE 有效样本统计、优势归一化、minibatch 抽样、PPO 策略/价值/熵/KL、CENet 辅助损失和经验归一化更新。
- 数值异常不进行 timeout bootstrap。正常回合结束及 timeout 逻辑保持。
- 整个 rollout 无有效样本时，两套优化器都跳过更新。新回合的有效数据可正常学习。

TensorBoard 可检查 `numerical/excluded_transition_fraction`（本轮排除比例），以及环境日志 `Numerical/invalid_envs`、`Numerical/total_invalid_envs`。
前 3 次异常事件输出最多 4 个环境的现场，之后每 100 次异常事件再输出现场，累计计数持续记录。
需要原来的停止诊断行为时，可给 `terminations.numerical_guard.params` 设置 `fail_fast=True`。

这层保护隔离检测到的异常数据，不修复物理引擎的异常来源。检测仅在策略步边界执行，不能追溯已被先前 rollout 用于更新的样本；同一 rollout 中的回合前缀采用保守排除。
模型动作出现非有限值，或者自动 reset 后观测仍有非有限值，Runner 仍会停止，以免把不可恢复的问题当作偶发故障。

验证包括离线 NaN 污染对照（有效样本参数更新完全一致）、全排除批次、跨回合边界、课程隔离与 GPU 单环境 NaN 注入后继续 PPO/CENet 更新。
