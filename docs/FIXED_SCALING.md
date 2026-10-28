# Manaro 固定观测缩放

新训练默认 `policy.fixed_observation_scaling=True`，
`empirical_normalization=False`。两者互斥。
参考 [Manaro 配置](https://github.com/Manaro-Alpha/DreamWaQ/blob/c8a6709d9f4437d3146a0a392ca26838b87e5864/legged_gym/legged_gym/envs/base/legged_robot_config.py)
与其 `compute_observations`，仅移植尺度，不改变奖励、动作缩放或执行器。

| 量 | 倍率 |
| --- | --- |
| 角速度、yaw 指令 | 0.25 |
| 投影重力、相对关节角、上一动作 | 1 |
| vx/vy 指令、真实/估计线速度 | 2 |
| 关节速度 | 0.05 |
| Critic 高度扫描 | 5 |

环境和部署接口仍输出/接收原始物理量。缩放在网络内部完成，45 维当前帧、
450 维 term-major 历史的每一帧及 Decoder 下一帧目标使用相同尺度。
Critic 使用工程现有的 `[线速度3, 本体观测45, 高度187]` 布局；高度项先按环境
原有规则裁剪，再乘 5。未移植 Manaro 的接触力特权观测或高度偏置。

CENet 输出速度的单位是 `2 × m/s`，监督标签和 AdaBoot 真值替换分支相同；
`cenet/velocity_mse`、`velocity_error_*`、`velocity_mae` 转回物理单位，
AdaBoot error 模式的阈值仍按 m/s 解释。辅助速度损失在缩放后的单位计算。

尺度作为模型 buffer 保存并进入 TorchScript/ONNX，部署端不要重复缩放。
导出加载器根据 checkpoint 配置及 buffer 恢复模式：旧 checkpoint 无尺度 buffer
时保持原始观测或原有经验归一化行为，不受新训练默认值影响。

建议新模式从头训练。恢复/微调旧 checkpoint 前，将 policy 配置的
`fixed_observation_scaling` 设为 `False`，经验归一化设置也须匹配旧模型；
跨固定缩放模式加载会明确报错，不会静默改变旧策略的输入语义。
