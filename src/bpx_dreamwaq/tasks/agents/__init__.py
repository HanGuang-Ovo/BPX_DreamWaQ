"""DreamWaQ 训练算法模块。

- ``cenet``: CENet 网络（共享编码器、速度估计头、VAE、Decoder、辅助损失）。
- ``dreamwaq_actor_critic``: CENet + 非对称 Actor-Critic 及部署导出模块。
- ``dreamwaq_ppo``: 扩展 Rollout 存储、CENet 辅助损失 PPO 与训练 Runner。
- ``rsl_rl_ppo_cfg``: 全部可配置超参数与默认值。

``cenet`` 与 ``dreamwaq_actor_critic`` 仅依赖 torch + rsl_rl，可离线测试；
``dreamwaq_ppo`` 的 Runner 需要环境包装器但算法类本身不依赖 isaaclab。
"""
