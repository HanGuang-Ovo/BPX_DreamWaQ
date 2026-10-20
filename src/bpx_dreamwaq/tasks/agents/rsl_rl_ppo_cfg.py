"""DreamWaQ PPO 训练配置（网络维度、历史长度、损失权重、AdaBoot 开关）。

所有 CENet 相关超参数均可配置；默认值为依据 DreamWaQ 论文结构与本工程
部署契约选择的初始值（论文未给出全部数值，损失权重为工程默认，见 README
「与论文的差异」一节）。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ..obs_contract import DEFAULT_HISTORY_LENGTH


@configclass
class DreamWaQActorCriticCfg:
    """DreamWaQ Actor-Critic（含 CENet）配置。"""

    class_name: str = "DreamWaQActorCritic"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    actor_hidden_dims: list = [512, 256, 128]
    critic_hidden_dims: list = [512, 256, 128]
    activation: str = "elu"

    # ---- CENet ----
    history_length: int = DEFAULT_HISTORY_LENGTH
    cenet_encoder_hidden_dims: list = [256, 128]
    cenet_vel_head_hidden_dims: list = [128]
    cenet_vae_head_hidden_dims: list = [128]
    cenet_decoder_hidden_dims: list = [128, 128]
    cenet_latent_dim: int = 16
    cenet_vel_dim: int = 3
    cenet_activation: str = "elu"

    # ---- 梯度与特征策略 ----
    # True：Actor 的 PPO 梯度不回传 CENet（CENet 仅由辅助损失更新）。
    detach_cenet_from_actor: bool = True
    # Actor 使用确定性隐变量均值（保证 PPO 概率比一致性）；可选 "sample"。
    latent_actor_mode: str = "mean"


@configclass
class DreamWaQPpoAlgorithmCfg:
    """DreamWaQ PPO 算法配置（标准 PPO 参数 + CENet 辅助损失）。"""

    class_name: str = "DreamWaQPPO"
    # ---- 标准 PPO ----
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.01
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    normalize_advantage_per_mini_batch: bool = False

    # ---- CENet 辅助损失权重 ----
    lambda_vel: float = 1.0
    lambda_pred: float = 1.0
    beta_kl: float = 1.0e-3
    # 是否允许 Actor 的 PPO 梯度回传 CENet（默认 False，见模块说明）。
    ppo_grad_to_cenet: bool = False

    # ---- AdaBoot 训练引导（默认开启，论文 reward_cv 模式）----
    adaboot_enabled: bool = True
    # "error": 按速度估计误差过渡（任务伪代码风格）；
    # "reward_cv": 论文公式 p = 1 - tanh(CV(R))；
    # "linear": 线性爬升。
    adaboot_mode: str = "reward_cv"
    adaboot_error_threshold: float = 0.5
    adaboot_ema_alpha: float = 0.9
    adaboot_ramp_iterations: int = 500
    adaboot_p_min: float = 0.0


@configclass
class BPXDreamWaQPPORunnerCfg:
    """BPX DreamWaQ 训练总配置（DreamWaQRunner 使用）。"""

    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 1500
    empirical_normalization: bool = False
    policy: DreamWaQActorCriticCfg = DreamWaQActorCriticCfg()
    algorithm: DreamWaQPpoAlgorithmCfg = DreamWaQPpoAlgorithmCfg()
    clip_actions: float | None = None
    save_interval: int = 50
    experiment_name: str = "bpx_dreamwaq"
    run_name: str = ""
    logger: str = "tensorboard"
    # Critic 特权观测中真实机体线速度的切片（前 3 维，见 rough_env_cfg.CriticCfg）。
    true_vel_slice: tuple = (0, 3)


@configclass
class BPXRoughPPORunnerCfg(BPXDreamWaQPPORunnerCfg):
    """基线 PPO（非 DreamWaQ）训练配置。

    仅在不需要 CENet 的对照实验中使用；当前仓库的 ``scripts/train.py``
    始终使用 DreamWaQRunner，基线可通过关闭两套课程获得行为对照。
    """

    experiment_name: str = "bpx_ppo_baseline"
