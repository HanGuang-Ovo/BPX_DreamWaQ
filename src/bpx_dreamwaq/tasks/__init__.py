"""BPX 训练任务注册。

导入本包即在 Gymnasium 注册以下环境：

- ``BPX-Velocity-Rough-DreamWaQ-v0``: 崎岖地形 + DreamWaQ CENet + PPO
  （地形课程默认开启，Grid-Adaptive 指令课程默认关闭；两套课程均可通过
  环境配置或命令行独立启停，同时关闭即为原始 PPO 基线）。
- ``BPX-Velocity-Rough-DreamWaQ-Play-v0``: 小规模可视化变体。
"""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="BPX-Velocity-Rough-DreamWaQ-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bpx_dreamwaq.tasks.rough_env_cfg:BPXRoughEnvCfg",
        "rsl_rl_cfg_entry_point": "bpx_dreamwaq.tasks.agents.rsl_rl_ppo_cfg:BPXDreamWaQPPORunnerCfg",
    },
)

gym.register(
    id="BPX-Velocity-Rough-DreamWaQ-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bpx_dreamwaq.tasks.rough_env_cfg:BPXRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "bpx_dreamwaq.tasks.agents.rsl_rl_ppo_cfg:BPXDreamWaQPPORunnerCfg",
    },
)
