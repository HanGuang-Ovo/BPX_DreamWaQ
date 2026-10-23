# BPX_DreamWaQ

BPX 四足机器人强化学习工程：基于 Isaac Lab 2.2.1 + RSL-RL 2.3.3 的
DreamWaQ 训练框架（CENet 状态估计 + PPO + 双课程学习），以及配套的
MuJoCo sim2sim 与官方 SDK sim2real 部署运行时。

| 能力 | 说明 |
| --- | --- |
| 统一策略 | 450 维历史观测 → 12 维关节位置动作；站立、行走、减速共用一个模型 |
| CENet | 从本体历史中估计机体线速度与 VAE 隐变量；Actor 不读取真实速度 |
| PPO | 扩展 Rollout 存储、辅助损失、AdaBoot 与部署导出 |
| 课程 | 地形难度课程（模块 A）+ Grid-Adaptive 指令课程（模块 B），可独立启停 |
| 运行时 | Isaac Lab 训练、MuJoCo sim2sim、官方 SDK sim2real、只读 Tk 启动检查 |

---

## 1. 工程结构

```text
BPX_DreamWaQ/
├── assets/robots/bpx/              # URDF、MJCF、网格与 USD；USD 缺失时首次训练自动转换
├── configs/
│   ├── sim2sim/                    # MuJoCo 场景：平地、历史观测、固定测试地形、综合场地
│   ├── sim2real/                   # SDK 网络、控制参数与标定模板
│   └── training/versions.toml      # 训练栈期望版本（运行时核对，不作 pip 安装请求）
├── docs/                           # 架构、训练环境、sim2sim / sim2real、验证记录
├── policies/                       # 导出的 ONNX / TorchScript 放置说明；权重不入 Git
├── scripts/
│   ├── setup_env.py                # 一键环境验证与可编辑安装
│   ├── train.py                    # 训练入口（Isaac Sim headless）
│   ├── export_policy.py            # checkpoint → ONNX / TorchScript
│   ├── sim2sim.py                  # MuJoCo sim2sim 入口
│   ├── sim2real.py                 # 实机预检、录制、分析、回放与显式控制
│   ├── monitor.py / visualize.py   # 只读反馈与启动前检查窗口（同一入口）
│   ├── check_project.py            # 离线资源与配置检查（无需 Isaac / MuJoCo）
│   ├── check_training_env.py       # 训练栈检查与十步 GPU 物理冒烟
│   ├── activate_training.sh        # 激活现有 Isaac Lab 训练环境
│   ├── training_python.sh          # 在训练环境中执行任意 Python
│   └── training_bootstrap/         # 工作区 Warp 1.8.1 兼容引导
├── src/bpx_dreamwaq/
│   ├── assets.py、paths.py         # 机器人资源与工作区路径
│   ├── runtime/                    # 共用关节、观测、历史、推理、路由与状态机
│   ├── sim2sim/                    # MuJoCo 配置、地形、运行器与速度/力矩曲线
│   ├── sim2real/                   # SDK 适配、标定、控制、预检、监控、录制与回放
│   ├── tasks/
│   │   ├── obs_contract.py         # 训练/部署共用：45 维单帧、450 维历史
│   │   ├── rough_env_cfg.py        # 崎岖地形速度跟踪环境、奖励/终止/事件与课程接线
│   │   ├── rough_env.py            # 奖励总和下限扩展环境
│   │   ├── mdp/
│   │   │   ├── rewards.py          # 本工程奖励项
│   │   │   ├── terminations.py     # 过倾终止
│   │   │   ├── curriculums.py      # 模块 A：整回合地形难度课程
│   │   │   ├── command_grid.py     # 模块 B：Grid-Adaptive 指令课程（纯 torch）
│   │   │   └── commands.py         # GridAdaptiveVelocityCommand（CommandTerm）
│   │   └── agents/
│   │       ├── cenet.py            # 共享编码器 + 速度头 + VAE + Decoder
│   │       ├── dreamwaq_actor_critic.py  # 非对称 Actor-Critic 与部署导出模块
│   │       ├── dreamwaq_ppo.py     # 扩展 Rollout、辅助损失 PPO、Runner、AdaBoot
│   │       └── rsl_rl_ppo_cfg.py   # 全部训练超参数与课程/AdaBoot 开关
│   └── validation.py               # 离线资源、配置与模块检查
├── tests/
│   ├── training/                   # CENet、PPO、奖励、课程离线单元测试（136 项）
│   ├── sim2sim/                    # MuJoCo 与配置测试（42 项）
│   ├── sim2real/                   # SDK 模拟、部署、监控测试（80 项）
│   └── project/                    # 工作区边界与入口测试（9 项）
└── outputs/                        # 训练、仿真、录制与诊断输出
```

**模块职责**

| 模块 | 职责 |
| --- | --- |
| `tasks/obs_contract.py` | 定义训练与部署共用的观测顺序、维度与历史布局 |
| `tasks/agents/cenet.py` | CENet：历史编码、速度估计、β-VAE 与下一帧观测预测 |
| `tasks/agents/dreamwaq_actor_critic.py` | Actor-Critic、CENet 接入、部署导出模块 |
| `tasks/agents/dreamwaq_ppo.py` | Rollout 存储扩展、辅助损失、AdaBoot、训练 Runner |
| `tasks/mdp/command_grid.py` | 在 (vx, wz) 网格上维护采样概率与跟踪统计 |
| `tasks/mdp/curriculums.py` | 逐环境整回合地形升降级判定 |
| `runtime/` | sim2sim 与 sim2real 共用的观测历史、ONNX 推理、指令路由与起身状态机 |

---

## 2. 环境验证与安装

### 2.1 一键检查 / 安装

工程入口脚本会自动定位工作区，不依赖当前目录。先运行检查：

```bash
cd ~/BPX_DreamWaQ

# 当前解释器 + 机器人资源、配置、训练模块
python scripts/setup_env.py

# 同时探测本机已知的 bpx-sim2sim / bpx-real / isaaclab 环境
python scripts/setup_env.py --all-envs

# 额外运行当前解释器可执行的离线测试；缺少依赖的套件会跳过并说明原因
python scripts/setup_env.py --tests

# 通过 training_python.sh 检查 Isaac Lab 训练栈（含 GPU 与版本）
python scripts/setup_env.py --train-check

# 机器可读报告
python scripts/setup_env.py --json

# 也可用指定环境的解释器运行（检查会针对该环境）
~/miniconda3/envs/bpx-sim2sim/bin/python scripts/setup_env.py --tests
~/miniconda3/envs/bpx-real/bin/python scripts/setup_env.py --tests
```

安装本工程与 pip 可选依赖：

```bash
# 可编辑安装，默认 extras=deploy（MuJoCo + ONNX Runtime）
python scripts/setup_env.py --install

# 一键：安装 + 检查 + 当前解释器可执行的离线测试
python scripts/setup_env.py --install --tests

# 只装仿真 / 推理，或只装基础依赖
python scripts/setup_env.py --install --extras sim,inference
python scripts/setup_env.py --install --extras ""
```

脚本只处理本工程和 pip extras；**不会安装 Isaac Lab、Isaac Sim 或机器人 SDK**，
也不会修改已有 Conda 环境的包。训练栈通过现有 Isaac 环境检查。

检查内容：

| 检查项 | 内容 |
| --- | --- |
| Python 与必需依赖 | 版本、NumPy 是否可用 |
| 机器人资源 | URDF/MJCF 网格引用是否完整、12 个受控关节集合是否一致 |
| sim2sim 配置 | 5 份 TOML 可解析、模型路径与观测维度正确 |
| 训练模块 | CENet、PPO、课程、环境配置与训练入口是否存在 |
| 可选依赖矩阵 | MuJoCo、ONNX Runtime、Tkinter、SDK、PyTorch、RSL-RL、TensorBoard |
| 离线测试（`--tests`） | 按当前解释器能力运行 project / sim2sim / sim2real / training 套件 |
| 训练栈（`--train-check`） | Isaac Lab 2.2.1、Isaac Sim 5.0、rsl-rl 2.3.3、CUDA 与版本核对 |

退出码 `0` 表示通过，`1` 表示工作区错误、必需依赖缺失或任一测试失败。

### 2.2 三套运行环境

| 环境 | 用途 | 关键依赖 | 入口 |
| --- | --- | --- | --- |
| `bpx-sim2sim` | MuJoCo sim2sim、离线测试 | `mujoco`、`onnxruntime` | `python scripts/sim2sim.py` |
| `bpx-real` | 实机 SDK 与只读窗口 | `bpx_sdk`、`onnxruntime`、Tkinter | `python scripts/sim2real.py` |
| `isaaclab` | 训练与导出 | Isaac Lab 2.2.1、Isaac Sim 5.0、`rsl-rl-lib==2.3.3`、PyTorch 2.7 | `bash scripts/training_python.sh` |

三套环境互不复制依赖；源码通过 `PYTHONPATH` 加载，不要求预先安装本工程。

### 2.3 训练环境

本机工作区已配置为复用 Conda `isaaclab`：Python 3.11、Isaac Lab 2.2.1、
Isaac Sim 5.0、`rsl-rl-lib==2.3.3`、PyTorch 2.7.0 + CUDA 12.8。
工作区内的 Warp 1.8.1 兼容补丁通过 `.training-deps/` 提供，不修改共享 Conda 包。

```bash
# 激活当前终端；新建 VS Code 终端也会自动选择这套环境
source scripts/activate_training.sh
python scripts/check_training_env.py --require-gpu

# 从任意 Conda 环境调用，也可直接使用项目包装入口
bash scripts/training_python.sh scripts/check_training_env.py --smoke-test \
    --report outputs/training/environment_report.json
```

其他机器可通过 `BPX_CONDA_ROOT`、`BPX_TRAIN_ENV`、`BPX_ISAACLAB_ROOT`、
`BPX_ISAACSIM_ROOT` 覆盖路径。完整版本、验证结果与 Warp 兼容问题见
[训练环境说明](docs/TRAINING_ENV.md)。本机 16 GB RAM / 8 GB VRAM：
建议从 16 环境冒烟、512–1024 环境正式训练。

### 2.4 其他机器安装

需要 Python 3.11+，在工程根目录执行：

```bash
python -m pip install -e '.[deploy]'      # NumPy + MuJoCo + ONNX Runtime
python -m pip install -e '.[sim]'         # 仅 MuJoCo
python -m pip install -e '.[inference]'   # 仅 ONNX Runtime
python -m pip install -e '.[torchscript]' # 可选 PyTorch 后端
python -m pip install -e .                # 基础：仅 NumPy
```

训练还需要按 [Isaac Lab v2.2.1 安装说明](https://github.com/isaac-sim/IsaacLab/blob/v2.2.1/docs/source/setup/installation/index.rst)
准备 Isaac Lab 与 Isaac Sim，并安装与本工程匹配的 `rsl-rl-lib==2.3.3`。
实机部署还需要安装与固件匹配的官方 `bpx_sdk` wheel。

本工程按**源码工作区**使用：模型、配置和输出保留在项目目录，采用可编辑安装。
安装后可选用命令行入口 `bpx-dreamwaq-sim`、`bpx-dreamwaq-real`、
`bpx-dreamwaq-monitor` 和 `bpx-dreamwaq-check`，它们与 `scripts/` 下同名入口等价。

### 2.5 机器人 SDK

`bpx-real` 环境已包含官方 SDK。参考工程的离线 wheel 位于
`../BPX_test/bpx_sdk_open-master/wheelhouse/`；本工程运行时不导入
`BPX_test`，也不复制 SDK 二进制。训练与 sim2sim 不依赖 SDK。

---

## 3. 训练

### 3.1 算法数据流

```text
历史本体观测 prop_history (450 = 6 项 × 10 帧, term-major, 项内旧→新)
        │
        ▼
CENet 共享编码器 ──► 速度估计头 ──► vel_hat (3)          ┐
        │                                                ├─► Actor 输入 (45+3+16=64) ──► 12 维动作
        └──────────────► VAE 头 ──► μ (16, 确定性) ──────┘
                             └──► z (采样) ──► Decoder ──► 预测下一时刻 45 维观测

Critic（特权）: [真实线速度 3, 45 维单帧, 高度扫描 23] = 71 维
辅助损失: L_CE = λ_vel·MSE(vel_hat, 真实线速度) + λ_pred·MSE(ô_{t+1}, o_{t+1}) + β·KL
```

关键设计：

- **Actor 不读取真实机体线速度**；真实速度仅作为 CENet 训练标签与 Critic 特权输入。
- **PPO 概率比一致性**：rollout 时实际送入 Actor 的 CENet 特征（含 AdaBoot
  混合速度）被保存到扩展 Rollout 存储，更新阶段原样复用；Actor 使用确定性
  隐变量均值 μ，不做二次采样。
- **梯度连接**：默认 `detach_cenet_from_actor=True`——Actor 的 PPO 梯度不回传
  CENet，CENet 仅由辅助损失更新；可配置 `ppo_grad_to_cenet=True`。
- **单一优化器**：CENet 是策略的子模块，与 Actor/Critic 共用一个 Adam 优化器；
  每个 minibatch 的总损失 = PPO 损失 + CENet 辅助损失。
- **done 屏蔽**：episode 终止转移的“下一观测”是 reset 后的新 episode 观测，
  预测损失与 KL 损失按 done 掩码屏蔽。
- **AdaBoot（默认开启，`reward_cv` 模式）**：训练期间按概率在真实速度与估计
  速度间切换 Actor 输入，缓解训练初期 CENet 估计不准的问题。支持 `error`
  （按速度估计误差过渡）、`reward_cv`（论文 p = 1 − tanh(CV(R))）、`linear`
  三种模式；实际送入 Actor 的速度被保存并在更新时原样复用；部署始终使用
  估计速度。

### 3.2 训练命令

```bash
# 短时冒烟测试（首次运行自动 URDF→USD 转换，需要几分钟）
bash scripts/training_python.sh scripts/train.py --headless --num_envs 16 --max_iterations 3

# 默认配置正式训练（当前默认：双课程开 + AdaBoot 开 reward_cv 模式）
bash scripts/training_python.sh scripts/train.py --headless --num_envs 512

# 指定实验名、随机种子与环境数
bash scripts/training_python.sh scripts/train.py --headless --num_envs 512 \
    --run_name history_cv --seed 42

# 恢复训练
bash scripts/training_python.sh scripts/train.py --headless --num_envs 512 \
    --resume --checkpoint outputs/training/bpx_dreamwaq/<run>/model_100.pt

# 只训练较短迭代数（调试）
bash scripts/training_python.sh scripts/train.py --headless --num_envs 256 --max_iterations 250
```

**日志与 checkpoint 目录**：

```text
outputs/training/bpx_dreamwaq/<YYYYMMDD_HHMMSS>[_<run_name>]/
├── model_50.pt          # 每 save_interval=50 次迭代保存一次
├── model_100.pt
├── ...
├── model_<iter>.pt      # 训练结束时再保存一次
└── events.out.tfevents.*
```

`BPX_LOG_DIR` 可改日志根目录（默认 `outputs/training`）。每个 checkpoint
包含模型权重、优化器状态和训练配置，可直接用于恢复或导出。

### 3.3 配置方式：命令行与配置文件

所有课程/AdaBoot 开关都有**命令行**（临时实验）和**配置文件**（改默认值）
两种用法；命令行优先级更高。开关在训练启动时读取，不能中途切换；`--resume`
时换开关会从该点改变行为，对照实验建议独立开跑。

**命令行开关**（`scripts/train.py`，常用参数还包括 `--num_envs`、
`--max_iterations`、`--seed`、`--run_name`、`--num_terrain_levels`）：

| 参数 | 作用 | 覆盖的配置项 |
| --- | --- | --- |
| `--disable_terrain_curriculum` | 关闭地形课程（默认开） | `enable_terrain_curriculum` |
| `--enable_command_curriculum` / `--disable_command_curriculum` | 开/关指令课程 | `enable_grid_adaptive_command_curriculum` |
| `--num_terrain_levels N` | 地形难度等级数（默认 10） | `num_terrain_levels` |
| `--adaboot` / `--disable_adaboot` | 开/关 AdaBoot（默认开） | `adaboot_enabled` |
| `--adaboot_mode error\|reward_cv\|linear` | AdaBoot 概率调度模式 | `adaboot_mode` |

```bash
# 两套课程 + AdaBoot 全开（当前默认，等价于不加开关）
bash scripts/training_python.sh scripts/train.py --headless --num_envs 512

# 只关指令课程
bash scripts/training_python.sh scripts/train.py --headless --num_envs 512 --disable_command_curriculum

# 关闭全部课程与 AdaBoot（原始 PPO + CENet 基线）
bash scripts/training_python.sh scripts/train.py --headless --num_envs 512 \
    --disable_terrain_curriculum --disable_command_curriculum --disable_adaboot

# AdaBoot 切换为按估计误差过渡
bash scripts/training_python.sh scripts/train.py --headless --num_envs 512 --adaboot_mode error
```

**配置文件默认值**分两处：

1. 环境侧（课程开关与网格参数）——`tasks/rough_env_cfg.py` 的
   `BPXRoughEnvCfg` 类属性：

   ```python
   enable_terrain_curriculum = True        # 地形课程总开关
   num_terrain_levels = 10                 # 难度等级数（地形网格行数）
   terrain_curriculum_mode = "standard"    # "standard" 升降级 / "upgrade_only" 只升不降
   terrain_curriculum_debug = False        # 逐等级环境数、整回合统计等调试日志
   terrain_curriculum_min_command_speed = 0.1       # 平移指令评价下限 (m/s)
   terrain_curriculum_min_command_distance = 1.0    # 最小累计指令路程 (m)
   terrain_curriculum_min_tracking_duration_s = 2.0 # 最小移动评价时长 (s)
   terrain_curriculum_min_moving_fraction = 0.5     # 明确移动任务：移动时间占比 ≥ 50%
   terrain_curriculum_lin_vel_tolerance = 0.1       # 仅降级/日志使用：平面误差容限 (m/s)
   terrain_curriculum_yaw_tolerance = 0.5           # 仅降级/日志使用：偏航误差容限 (rad/s)
   terrain_curriculum_failure_ratio = 0.5           # 持续失败：≥ 50% 移动采样跟踪不良
   terrain_curriculum_coverage_ratio = 0.25         # 轴向空间覆盖门槛 = 地形边长 × 0.25

   enable_grid_adaptive_command_curriculum = True    # 指令课程总开关
   command_rel_standing_envs = 0.1        # 10% 精确全零站立指令
   command_grid_resolution_vx = 0.5        # vx 网格分辨率 (m/s)
   command_grid_resolution_yaw = 0.5       # wz 网格分辨率 (rad/s)
   command_curriculum_initial_vx_range = (-0.5, 0.5)  # 初始激活范围
   command_curriculum_initial_yaw_range = (-0.5, 0.5)
   command_grid_continuous_sampling = True          # 选格后在格内连续采样
   command_low_speed_sampling_prob = 0.3            # 30% 额外低速样本
   command_low_speed_vx_range = (-0.25, 0.25)        # m/s
   command_low_speed_vy_range = (-0.1, 0.1)          # m/s
   command_low_speed_yaw_range = (-0.25, 0.25)       # rad/s
   # 最大扩展范围从 command_lin_vel_x_range / command_ang_vel_z_range（±1.0）读取
   ```

   更细的门控参数（成功阈值、`min_samples`、EMA 系数、探索下限）在
   `tasks/mdp/commands.py` 的 `GridAdaptiveVelocityCommandCfg`，由
   `command_grid.py` 的 `GridAdaptiveCommandScheduler` 使用。连续模式按各轴
   半格宽与任务量程的交集均匀采样，边界格直接在截断区间采样。旧式
   `command_jitter` 仅在关闭连续模式时生效。

2. 算法侧（AdaBoot 与损失权重）——`tasks/agents/rsl_rl_ppo_cfg.py` 的
   `DreamWaQPpoAlgorithmCfg`：

   ```python
   adaboot_enabled = True        # AdaBoot 总开关
   adaboot_mode = "reward_cv"    # error / reward_cv / linear
   adaboot_error_threshold = 0.5 # error 模式：EMA 速度误差达到该值(m/s)时 p≈0
   adaboot_ema_alpha = 0.9       # 误差 EMA 平滑系数
   adaboot_ramp_iterations = 500 # linear 模式爬升迭代数
   adaboot_p_min = 0.0           # p 下限
   ```

### 3.4 TensorBoard

```bash
# 使用训练环境自带的 TensorBoard
bash scripts/training_python.sh -m tensorboard.main --logdir outputs/training/bpx_dreamwaq

# 或直接调用环境内可执行文件
~/miniconda3/envs/isaaclab/bin/tensorboard --logdir outputs/training/bpx_dreamwaq
```

浏览器打开 `http://localhost:6006`。全部曲线按迭代（约每 24×N 步）记录一次；
括号里的数值是本机 256 环境 × 250 迭代实测训练的参考形态（完整记录见
[验证记录](docs/VALIDATION.md)）。

**① 策略学习进度——`policy/*`、`Episode Reward/*`、`Episode Termination/*`**

| 曲线 | 含义 | 健康形态（实测） |
| --- | --- | --- |
| `policy/mean_reward` | 最近 100 个回合的平均总奖励，最核心的学习指标 | 持续上升（−1.24→−0.40） |
| `policy/episode_length` | 平均回合步数（满 20 s = 1000 步） | 先跌后回升（20→9→25） |
| `Episode Reward/track_*` | 速度跟踪奖励分项 | 随总奖励上升 |
| `Episode Reward/*_l2` | 力矩/加速度/动作变化率等罚项 | 负值、量级小；暴涨=动作抖动加剧 |
| `Episode Termination/*` | 终止原因占比 | `base_contact/base_tilt` 下降、`time_out` 上升 |

`Train/mean_reward` 与 `policy/mean_reward` 同源，看一个即可。回合长度长期
个位数 = 机器人秒倒；`time_out` 占比是最直接的“站稳了吗”指标。

**② CENet 状态估计质量——`cenet/*`（任务 1 的验收指标）**

| 曲线 | 含义 | 健康形态（实测） |
| --- | --- | --- |
| `cenet/velocity_mse` | 速度估计 MSE。**判断 CENet 成败的首要指标** | 单调下降后趋平（0.091→0.017） |
| `cenet/velocity_error_x/y/z` | 三轴平均绝对误差（m/s） | 0.18/0.17/0.33 → 0.09/0.08/0.11；`velocity_mae` < 0.1 即相当准 |
| `cenet/prediction_loss` | Decoder 预测下一时刻观测的 MSE | 缓慢下降（22.9→10.5） |
| `cenet/kl_loss` | VAE KL 散度 | 不发散即可（β=1e-3 下加权贡献很小） |
| `cenet/total_auxiliary_loss` | 三项辅助损失加权和 | 平稳有限 |
| `cenet/adaboot_p_boot` / `adaboot_est_fraction` | AdaBoot 概率 / 实际使用估计速度的占比 | 两者联动（reward_cv 约 0.5–0.65） |

警告信号：`velocity_mse` 居高不下（检查历史观测输入）、NaN（学习率过大）、
`kl_loss` 无界增长（调大 `beta_kl`）。

**③ 地形课程进展——`curriculum/terrain/*`（模块 A）**

- `mean_level`：平均地形难度（0–9），**步态稳定后阶梯式爬升**；早期恒 0 属预期。
- `level_XX_frac`：各难度环境比例，健康时整体向高等级迁移。
- `move_up_total` / `move_down_total`：累计升降级次数，早期 down 为主、中后期 up 反超。
- `level_XX_mean_reward/length`：各难度回合表现——爬级后先回落再恢复是课程起作用的正常表现。

**④ 指令课程进展——`curriculum/command/*`（模块 B）**

- `activated_cells`：激活指令单元数（初始 9=3×3），策略掌握当前区域后增加。
- `vx_range` / `yaw_range`：激活区域覆盖范围（初始 0.75）。
- `expansions`：累计激活数，单调不减。
- `Metrics/base_velocity/error_vel_xy、error_vel_yaw`：指令跟踪误差。

**⑤ PPO 内部状态——`Loss/*`、`Policy/mean_noise_std`**

- `Loss/surrogate`：clipped 代理损失，**不需要单调下降**。
- `Loss/value_function`：Critic 回归损失，下降趋稳。
- `Loss/entropy` / `Policy/mean_noise_std`：中后期缓慢下降；接近 0 会失去探索。
- `Loss/learning_rate`：自适应学习率，锯齿波动正常。

**⑥ 性能——`Perf/*`**：`total_fps`（实测 ≈2900 @256 env）、
`collection time` / `learning time` 占比。

**快速看板（按优先级）**：① `policy/mean_reward` ↑ 且 `time_out` 占比 ↑
→ 策略在学；② `cenet/velocity_mse` ↓ → 估计器在学；③ `curriculum/*` 阶梯
上升 → 课程在推进；④ 无 NaN、`value_function` 收敛、熵缓降 → 训练稳定。

### 3.5 导出部署策略

训练完成后导出不含特权信息的部署策略（输入 450 维历史观测 → 输出 12 维动作，
内部执行 CENet 确定性前向 + Actor）：

```bash
# 同时导出 ONNX 和 TorchScript
bash scripts/training_python.sh scripts/export_policy.py \
    --checkpoint outputs/training/bpx_dreamwaq/<run>/model_<iter>.pt --format both

# 只导出 ONNX，并指定输出前缀
bash scripts/training_python.sh scripts/export_policy.py \
    --checkpoint outputs/training/bpx_dreamwaq/<run>/model_<iter>.pt \
    --format onnx --out policies/policy_history_cv
```

默认输出 `policies/policy.onnx` 与 `policies/policy.pt`。导出模型不依赖
Critic、真实速度、高度扫描或 VAE Decoder。

**手工推理验证**（不用特权信息）：

```python
import torch

model = torch.jit.load("policies/policy.pt")
model.eval()
history = torch.zeros(1, 450)   # 部署端 ObservationHistory 的输出布局
action = model(history)         # (1, 12) 关节位置动作
print(action.shape)
```

也可从 checkpoint 直接构建：

```python
from scripts_path import build_policy_from_checkpoint  # scripts/export_policy.py
deployable, info = build_policy_from_checkpoint("outputs/training/.../model_100.pt")
action = deployable(history_450)
```

导出接口与部署运行时一致：

- 输入：展平历史观测 `(B, 450)`，term-major、项内从最旧到最新
  （`[角速度×10, 投影重力×10, 指令×10, 关节角×10, 关节速度×10, 上一步动作×10]`）；
- 内部：最新一帧 45 维 → CENet（速度估计 3 + 隐变量均值 16）→ 拼接为 64 维 Actor 输入；
- 输出：12 维关节位置动作，部署侧执行 `q = q_default + 0.5 × a`。

### 3.6 训练测试

```bash
# 离线单元测试（无需 Isaac Sim，136 项：维度/梯度/存储/课程/奖励）
bash scripts/training_python.sh -m unittest discover -s tests/training -v
```

已验证内容见 [验证记录](docs/VALIDATION.md)。短时训练冒烟命令见 3.2。

---

## 4. sim2sim（MuJoCo）

无需策略权重即可运行模型与 PD 框架检查：

```bash
# 默认 zero 后端：只验证 PD、起身状态机与控制流程
python scripts/sim2sim.py --duration 2 --no-realtime
python scripts/sim2sim.py --viewer --duration 20
```

放入导出权重后显式选择推理后端：

```bash
python scripts/sim2sim.py --backend onnx --policy policies/policy_20261023_iter2250_stable.onnx --viewer
python scripts/sim2sim.py --config configs/sim2sim/bpx_mixed.toml \
    --backend onnx --gamepad --viewer \
    --velocity-plot --torque-plot --log outputs/sim2sim/run.csv

# 在第 2 档（8 cm）楼梯前的平地趴卧出生
python scripts/sim2sim.py --config configs/sim2sim/bpx_mixed.toml \
    --backend onnx --gamepad --viewer --stair-level 2
```

默认启用起身状态机：手柄模式以趴姿等待 RB 起身；无手柄时自动 Init。
起身采用足端五次轨迹 + PD，确认站稳后平滑接入策略。左摇杆控制前后/横移，
右摇杆水平控制转向，松杆后减速站立。`--direct-policy` 可跳过 Init 做接口对拍。
MuJoCo 不因高度、倾角、躯干触地或 Init 超时自动退出；策略接管后持续执行，
运行到设定时长、关闭窗口或 Ctrl+C 时结束。

`--terrain flat|three_level|stairs|mixed` 可覆盖地形；综合场地包含 ±2/±4 cm
崎岖路面和 3/8/10/15 cm 四档上下楼梯。楼梯采用 box 碰撞体，踏面深 20 cm，
每侧 5 级高差，含顶部平台。详见 [sim2sim 操作与验证](docs/SIM2SIM.md)。

| 配置 | 用途 | 策略输入 |
| --- | --- | --- |
| `bpx_history.toml` | 默认平地，保留历史观测接口 | 450 |
| `bpx_flat.toml` | 旧单帧接口参考，包含机体线速度 | 48 |
| `bpx_terrain_history.toml` | 固定三段测试地形，趴姿起步 | 450 |
| `bpx_mixed.toml` | 崎岖 + 分级上下楼梯，趴姿起步，300 秒手柄测试 | 450 |
| `bpx_terrain_standing.toml` | 旧单帧接口参考，固定测试地形、站姿起步 | 48 |

常用参数：`--viewer`（MuJoCo 窗口）、`--velocity-plot` / `--torque-plot`
（速度与力矩曲线）、`--gamepad`（Linux 手柄）、`--no-realtime`（尽快跑完）、
`--duration`、`--log`、`--supervisor` / `--direct-policy`、`--auto-init`。

---

## 5. sim2real（官方 SDK）

复用 `BPX_test` 的官方 SDK 部署流程，当前默认加载与 `bpx_mixed.toml`
一致的 DreamWaQ 模型。可直接使用本机已有的 `bpx-real` 环境：

```bash
conda activate bpx-real
python scripts/sim2real.py check
python scripts/sim2real.py run
python scripts/monitor.py --demo
```

以上不连接机器人。预检校验模型、试推理、SDK 依赖和关节映射；
`ready: true` 表示软件资源可用。监控窗口需要 Tkinter 和图形桌面。

### 5.1 实机控制

完成现场网络、支撑和坐标核对后，执行：

```bash
python scripts/sim2real.py run --live

# 多手柄时指定设备；可设置运行时长和日志路径
python scripts/sim2real.py run --live --gamepad-index 0 --duration 20 \
    --output outputs/sim2real/live-01.jsonl

# 可临时覆盖模型：--policy 放在子命令之前
python scripts/sim2real.py --policy policies/other.onnx run --live
```

1. 启动时读取完整关节与 IMU 反馈。趴卧时保持当前关节姿态，等待 RB/R1。
   已处于 SDK `Motion` 状态时从实测关节姿态平滑接管。
2. 按 RB/R1 后执行 4 秒足端五次轨迹与解析 IK：前 1 秒建立支撑，随后抬升。
3. 起身结束进入 Stand，平滑切换增益和策略目标；左摇杆控制 vx/vy，
   右摇杆水平控制 wz，持续推杆经防抖进入 Walk。
4. 松杆后指令逐渐归零，经 STOPPING 回到 Stand。同一模型与历史始终连续。
5. Ctrl+C、指定时长结束或运行异常时退出，尝试发送三次 kp=0、kd=1 阻尼
   并断开 SDK。阻尼发送属于尽力处理，不能保证网络故障时送达。

策略以 50 Hz 更新、目标以 100 Hz 发送，自动保存日志，可离线分析与回放。
权重文件不纳入 Git，其他机器需复制模型或重新导出。

### 5.2 只读可视化启动检查

启动前可使用从 `BPX_test` 迁入的只读可视化界面，核对模型、12 关节反馈、
IMU 姿态、手柄输入与未执行的策略目标：

```bash
python scripts/visualize.py          # 先检查模型，再手动连接只读反馈
python scripts/visualize.py --demo   # 模拟反馈，不连接机器人
python scripts/visualize.py --demo --no-policy   # 只看反馈与姿态
python scripts/visualize.py --recording outputs/sim2real/preflight-01.jsonl
python scripts/visualize.py --list-gamepads
```

界面包含三页：

| 页面 | 内容 |
| --- | --- |
| 启动前检查 | 模型文件/SHA256、输入输出尺寸、试推理与预热、SDK 依赖、观测契约、关节映射、反馈新鲜度、手柄、策略预览状态 |
| 实时反馈与姿态 | 12 关节表、SDK/官方/模型三套角度、action 与未发送目标、四元数与 RPY、三维骨架、10 秒曲线 |
| 官方定义 / 当前配置 | 官方示例参数与当前 kp/kd、关节范围、标定通道逐项对照 |

VS Code 可直接选择 **BPX: 只读启动前检查** 调试配置启动，使用 `bpx-real`。
详见 [只读窗口说明](docs/SIM2REAL_WINDOW.md)。

### 5.3 配置与日志

控制与监控入口均优先读取 `configs/sim2real/local.runtime.toml`，不存在时读取
`configs/sim2real/runtime.toml`。需要修改网络或标定时创建本机配置：

```bash
cp -n configs/sim2real/runtime.toml configs/sim2real/local.runtime.toml
cp -n configs/sim2real/joint_calibration.template.toml configs/sim2real/local.calibration.toml
```

TOML 中的模型和标定路径相对于配置文件；命令行 `--policy` 路径相对于调用目录。
本机配置已被 `.gitignore` 排除。

日志分析与回放（均不连接 SDK）：

```bash
python scripts/sim2real.py inspect --output outputs/sim2real/state-01.jsonl --duration 10
python scripts/sim2real.py analyze outputs/sim2real/live-01.jsonl
python scripts/sim2real.py replay outputs/sim2real/live-01.jsonl \
    --output outputs/sim2real/replay-01.jsonl
python scripts/monitor.py --recording outputs/sim2real/live-01.jsonl
```

完整接口、标定与安全边界见 [实机部署说明](docs/SIM2REAL.md)。

---

## 6. 工程逻辑与设计

### 6.1 观测、动作与频率契约

| 项目 | 训练 | 部署 |
| --- | --- | --- |
| 策略频率 | 50 Hz（200 Hz 物理 × decimation 4） | 50 Hz 策略 / 100 Hz 发送 |
| 单帧观测 | 45 维：角速度 3、投影重力 3、指令 3、相对关节角 12、关节速度 12、上一步动作 12 | 同左 |
| 历史观测 | H=10，term-major、项内旧→新，共 450 维 | 同左 |
| Critic 观测 | 71 维：真实线速度 3 + 45 维单帧 + 高度扫描 23 | 不使用 |
| 动作 | 12 维关节位置目标，`q = q_default + 0.5 × a` | 同左 |
| 关节顺序 | FL/FR/HL/HR roll → pitch → knee | 同左；SDK 通道 `[0,3,6,9,1,4,7,10,2,5,8,11]` |
| 默认角 | 每腿 `[0, 0.7, -1.4]` rad | 同左 |
| 增益 | 策略 kp=40、kd=1；起身 kp=80、kd=2 | 同左 |

观测布局由 `tasks/obs_contract.py` 唯一定义，训练端 Isaac Lab 的
`history_length=10, flatten_history_dim=True` 与部署端
`runtime/history.py` 的输出逐项一致。

### 6.2 CENet 与 PPO 训练逻辑

- **CENet 输入**：450 维历史观测；共享编码器 `[256, 128]`。
- **速度头**：`[128] → 3`，输出 `vel_hat`，回归仿真真实线速度。
- **VAE 头**：`[128] → μ/σ (16)`；训练时采样 `z` 送入 Decoder，Actor 使用 μ。
- **Decoder**：`[128, 128] → 45`，预测下一时刻单帧观测。
- **Actor**：`[45 + 3 + 16 = 64] → [512, 256, 128] → 12`。
- **Critic**：71 维特权观测 → `[512, 256, 128] → 1`。
- **PPO**：lr 1e-3（自适应 KL 0.01）、γ 0.99、λ 0.95、clip 0.2、熵系数 0.01；
  每迭代 24 步 × 5 epoch × 4 minibatch；`max_iterations=1500`，
  `save_interval=50`。
- **辅助损失**：`λ_vel=1.0`、`λ_pred=1.0`、`β_KL=1e-3`；done 掩码屏蔽
  终止转移。
- **部署导出**：只保留 `cenet.*` 与 `actor.*`；确定性 μ 前向，不依赖
  Critic、真实速度、高度扫描或 Decoder。

**关键超参数**（`tasks/agents/rsl_rl_ppo_cfg.py`，均可配置）：

| 类别 | 参数 | 默认值 |
| --- | --- | --- |
| 网络 | Actor/Critic MLP | `[512, 256, 128]`，ELU |
| 网络 | CENet 编码器 / 速度头 / VAE 头 / Decoder | `[256,128]` / `[128]` / `[128]` / `[128,128]` |
| 网络 | 隐变量 / 速度 / 历史长度 | 16 / 3 / 10 |
| PPO | 学习率 / 自适应 KL | 1e-3 / 0.01 |
| PPO | γ / λ / clip / 熵系数 / 值损失系数 | 0.99 / 0.95 / 0.2 / 0.01 / 1.0 |
| PPO | 每迭代步数 × epoch × minibatch | 24 × 5 × 4 |
| PPO | 最大迭代 / 保存间隔 | 1500 / 50 |
| PPO | 梯度裁剪 / 噪声初始化 | 1.0 / 1.0（scalar） |
| 辅助损失 | λ_vel / λ_pred / β_KL | 1.0 / 1.0 / 1e-3 |
| 特征策略 | CENet 梯度 / Actor 隐变量 | detach / 确定性 μ |

### 6.3 环境设计

**场景**（`BPXSceneCfg`）

| 组件 | 配置 |
| --- | --- |
| 地形 | 生成器网格：`8 m × 8 m` 地块，`num_rows=num_terrain_levels`（默认 10），`num_cols=10`，边界 20 m |
| 机器人 | BPX USD（缺失时首次训练由 URDF 自动转换） |
| 高度扫描 | `GridPatternCfg(resolution=0.1, size=[1.6, 1.0])`，机体上方 20 m 向下，yaw 对齐 |
| 接触传感器 | `.*` 刚体，`history_length=3`，记录滞空时间 |
| 光照 | DomeLight 强度 750 |
| 物理 | `dt=0.005`（200 Hz），`decimation=4`（50 Hz），回合 20 s |
| 并行环境 | 配置默认 1024；示例命令用 `--num_envs 512` 覆盖，按显存调整 |

**地形类型**（按比例随机组合）

| 子地形 | 比例 | 参数 | 作用 |
| --- | --- | --- | --- |
| `random_rough` | 20% | 噪声 0.01–0.06 m，步长 0.01 | 连续崎岖路面 |
| `boxes` | 20% | 网格 0.45 m，高度 0.03–0.12 m，平台 2.0 m | 离散方块 |
| `pyramid_stairs` | 20% | 级高 0.03–0.15 m，踏面 0.10 m，平台 3.0 m | 上台阶 |
| `pyramid_stairs_inv` | 20% | 同上 | 下台阶 |
| `hf_pyramid_slope` | 10% | 坡度 0–0.4 | 上坡 |
| `hf_pyramid_slope_inv` | 10% | 坡度 0–0.4 | 下坡 |

`horizontal_scale=0.1`、`vertical_scale=0.005`、`slope_threshold=0.75`；
`use_cache=False`，每次启动重新生成。

**指令**（`CommandsCfg` / `GridAdaptiveVelocityCommandCfg`）

| 项目 | 值 |
| --- | --- |
| 任务量程 | vx ∈ [−1.0, 1.0] m/s，vy ∈ [−0.5, 0.5] m/s，wz ∈ [−1.0, 1.0] rad/s |
| 重采样 | 每 10 s（episode 内） |
| 站立比例 | 每次重采样 10% 精确全零指令 |
| 航向 | `heading_command=False`，直接采样角速度 |
| 课程关闭时 | `TerrainAwareUniformVelocityCommand` 均匀采样（仍记录地形统计） |

**域随机化**（`EventCfg`）

| 事件 | 模式 | 参数 |
| --- | --- | --- |
| 物理材质 | startup | 静摩擦 0.8、动摩擦 0.6、恢复系数 0、64 桶随机 |
| 机身质量 | startup | torso 加 −1.0～+3.0 kg |
| 初始位姿 | reset | x/y ∈ ±0.5 m，yaw ∈ ±π，初速度为零 |
| 关节复位 | reset | 位置比例 1.0，速度 0 |
| 外力推挤 | interval 10–15 s | x/y 速度 ±0.5 m/s |

**终止条件**（`TerminationsCfg`）

| 终止 | 条件 |
| --- | --- |
| `time_out` | 20 s 回合超时（正常结束） |
| `base_contact` | torso 接触力 > 1 N |
| `base_tilt` | 机身倾角 > 1.0 rad |

### 6.4 奖励设计

所有奖励由 Isaac Lab `RewardManager` 统一乘权重和策略时间步 `dt=0.02 s`；
`BPXManagerBasedRLEnv` 在求和后再对每个环境的**单步总奖励**施加下限
`minimum_step_reward = −10.0`（只抬升负值，不缩放正奖励，NaN/Inf 仍交给
训练器报错）。

**任务奖励**

| 奖励项 | 权重 | 实现 | 作用 |
| --- | --- | --- | --- |
| `track_lin_vel_xy_exp` | +1.5 | Isaac Lab 内置，`std=0.5` | 平面线速度跟踪 |
| `track_ang_vel_z_exp` | +0.75 | 内置，`std=0.5` | 偏航角速度跟踪 |

**姿态与能耗正则**

| 奖励项 | 权重 | 实现 | 作用 |
| --- | --- | --- | --- |
| `lin_vel_z_l2` | −2.0 | 内置 | 抑制机身竖直速度 |
| `ang_vel_xy_l2` | −0.05 | 内置 | 抑制 roll/pitch 角速度抖动 |
| `base_height_l2` | −1.0 | 本工程 | `(机身高度 − 扫描地面均值 − 0.42)²`；无有效射线时跳过该环境 |
| `flat_orientation_l2` | −0.3 | 内置 | 惩罚 roll/pitch 偏离水平（投影重力） |
| `joint_pos_limits` | −0.2 | 内置 | 关节位置越界 |
| `dof_torques_l2` | −1.0e-4 | 内置 | 关节力矩平方和 |
| `dof_acc_l2` | −2.5e-7 | 本工程 | 关节加速度平方和；单环境上限 1e8，异常加速度按上限计 |
| `action_rate_l2` | −0.01 | 内置 | 一阶动作变化率 |
| `action_smoothness_l2` | −0.01 | 本工程 | 二阶动作差分 `sum((a_t − 2a_{t−1} + a_{t−2})²)` |

**步态与接触**

| 奖励项 | 权重 | 实现 | 作用 |
| --- | --- | --- | --- |
| `feet_air_time` | +0.1 | 本工程 | 触地瞬间，滞空时间超过 0.4 s 的部分给予奖励；平移或 yaw 指令超过 0.1 时启用 |
| `feet_air_time_excess` | −1.0 | 本工程 | 单脚连续悬空超过 0.6 s 后线性惩罚，每脚上限 1 |
| `feet_contact_balance` | −0.5 | 本工程 | 四脚触地占空比偏差平方，1 s 指数滑动平均；允许交替摆腿 |
| `feet_clearance` | −0.01 | 本工程 | `toe_link` 距局部地面 0.10 m 的误差平方 × 足端水平速度 |
| `feet_lateral_distance` | −0.3 | 本工程 | 左右足横向间距不足 0.26 m 的归一化缺口平方 |
| `feet_longitudinal_distance` | −0.3 | 本工程 | 同侧前后足纵向间距不足 0.30 m 的归一化缺口平方 |
| `undesired_contacts` | −1.0 | 内置 | torso 接触力 > 1 N |

所有权重都在 `tasks/rough_env_cfg.py` 的 `RewardsCfg` 中调整；修改奖励后需要
重新训练，已导出的策略不会随之改变。

**设计说明**

- 高度项使用高度扫描的**平均地面**作参考，避免在坡面/台阶上按世界系高度误罚；
  扫描均值是局部支撑面的近似，跨台阶时仍有地形估计误差。
- 净空项的 `target_height=0.10 m` 指 `toe_link` 原点高度，不是鞋底净空；
  扫描分辨率 0.1 m，台阶边缘只能近似。
- 足距项只消除机身 yaw，不使用 roll/pitch，因此坡面上的高度差不会改变足距；
  达到下限后代价为零，不鼓励无限张腿。
- `feet_air_time` 只在落脚瞬间给奖励；`feet_air_time_excess` 持续惩罚超时悬空，
  防止三足支撑策略通过始终不落脚逃避着地项。
- 逐脚指标可在 TensorBoard 查看：`gait/{fl,fr,hl,hr}/contact_fraction`、
  `air_time_max_s`、`long_air_fraction`。左后腿长期抬起问题的复现与修正见
  [步态诊断](docs/GAIT_DIAGNOSIS.md)。

### 6.5 课程学习（模块 A / 模块 B）

| 开关 | 位置 | 默认 | 说明 |
| --- | --- | --- | --- |
| `enable_terrain_curriculum` | `BPXRoughEnvCfg` / `--disable_terrain_curriculum` | 开 | 整回合表现与空间覆盖地形课程 |
| `num_terrain_levels` | 同上 / `--num_terrain_levels` | 10 | 地形难度等级数 |
| `terrain_curriculum_mode` | 同上 | `standard` | `standard` 升降级 / `upgrade_only` 只升不降 |
| `terrain_curriculum_debug` | 同上 | 关 | 额外课程调试日志 |
| `enable_grid_adaptive_command_curriculum` | 同上 / `--enable_command_curriculum` / `--disable_command_curriculum` | 开 | Grid-Adaptive 指令课程 |

**模块 A：地形课程**（`tasks/mdp/curriculums.py`）。每个 episode 结束时按
环境独立判定，整回合数据由指令模块逐步累计，不在 10 s 指令重采样时清零：

- 升级：正常超时完成回合、无失败终止，累计平移指令路程 ≥ 1 m、移动时长
  ≥ 2 s，且满足实际空间覆盖条件。不检查速度跟踪达标比例或误差容限。
- 降级：未满足升级条件，本回合至少 50% 的时间有明确平移指令（平面指令速度
  > 0.1 m/s），并发生失败终止，或在满足 1 m / 2 s 评价门槛后至少 50% 的
  移动采样跟踪不良。明确移动任务中提前摔倒可直接降级；等级 0 不再下降。
- 保持：零速、原地旋转、评价数据不足，或正常低速/转弯且跟踪良好但覆盖不足。

降级统计中的跟踪良好定义为同一步平面速度误差 ≤ 0.1 m/s 且偏航误差
≤ 0.5 rad/s。误差容限仅用于降级和日志；满足升级条件时优先升级。
空间覆盖采用整回合的轴向最大距离：相对真实出生点、相对地形中心均须达到
地形边长 × 0.25（8 m 地形为 2 m），避免往返被净位移误判，也避免只在中央
方形平台绕小圈或沿对角线移动就升级。这是空间覆盖代理指标，并非精确障碍
接触判定。`upgrade_only` 保留上述升级条件并禁用降级。

终止步在机器人 reset 前补采，回合 reset 后重新记录出生点，各环境独立清零；
指令课程开关关闭时，均匀指令采样也使用同一套地形统计。升降级互斥，达到
最高级后随机分配等级。逐等级回合奖励/长度、实际升降级次数、保持次数写入
TensorBoard（`curriculum/terrain/*`、`Curriculum/terrain_levels/*`）；
开启 debug 可查看指令路程、移动时间、空间覆盖和跟踪达标占比。关闭课程时
等级随机初始化且不再变化。

**模块 B：Grid-Adaptive 指令课程**（`tasks/mdp/command_grid.py` + `commands.py`）。
在 (vx, wz) 联合网格（默认 0.5 m/s × 0.5 rad/s 分辨率，中心在零速）上维护
每个指令组合的采样概率与跟踪统计；初始激活 [−0.5, 0.5]²，最大扩展到任务
指令量程（[−1, 1] × [−1, 1]）。单元在「区间平均 |Δvx| < 0.25 m/s 且
|Δwz| < 0.5 rad/s」的 EMA 成功率达标且采样 ≥ `min_samples` 后激活 4 邻域。

非站立样本中，默认 70% 指令使用原分布（90% 激活单元均匀 + 10% 全合法单元
均匀探索），另外 30% 从已激活格与零速附近区域的交集采样；低速区尚未激活时
退回原分布。低速分布按格与低速区域的交叠面积加权，不绕过课程激活门控。
vx 与 wz 先联合选同一单元，再在格内连续采样；两端格截断到任务量程。
常规分支 vy 均匀采样于任务量程，低速分支同步收窄到 ±0.1 m/s。每次重采样
独立以 10% 概率发送精确全零站立指令（`command_rel_standing_envs`），不参与
课程成功统计。因此默认最终采样比例约为 10% 精确零速、27% 近零连续运动、
63% 常规网格课程与探索；比例指重采样次数，实际步数占比受回合提前结束影响。

统计按指令执行区间（episode 内 10 s 重采样周期）结算，不混入整个 episode。
关闭时恢复 `UniformVelocityCommand` 均匀采样；两套课程同时关闭即为原始
PPO 基线。网格数据（概率/采样数/成功率）随 checkpoint 导出
`curriculum_<iter>.npz`，可用任意工具绘制热力图。

### 6.6 设计决策与论文差异

依据 DreamWaQ 论文（arXiv:2301.10602, ICRA 2023）核对后的实现差异与决策：

1. **历史长度**：论文 H=5；本工程取 10 帧，与部署端 `ObservationHistory`
   既有契约（450 维）一致。CENet 结构（共享编码器 + 速度头 + β-VAE 头 +
   Decoder 预测下一时刻观测）与论文一致。
2. **隐变量送入 Actor**：论文未说明用 μ 还是采样 z；本实现默认确定性 μ
   （保证 PPO 概率比一致性），可配置 `"sample"`。
3. **梯度连接**：论文未说明 PPO 梯度是否回传 CENet；默认 detach，可配置。
4. **损失权重**：论文未给出 λ_vel/λ_pred/β 数值；默认 1.0/1.0/1e-3。
5. **观测归一化**：论文 VAE 先验依赖零均值单位方差输入；本工程部署契约无
   观测归一化器，故 `empirical_normalization=False`（记录为差异）。
6. **AdaBoot**：论文按回合奖励变异系数 p = 1 − tanh(CV(R))；本实现默认采用
   该公式，并提供按速度估计误差过渡的 `error` 模式与 `linear` 线性爬升模式。
7. **Critic 特权信息**：论文含随机扰动力 d_t；本实现 Critic 为真实线速度 +
   单帧观测 + 高度扫描（扰动只有 push 事件，未把扰动力显式入观测）。
8. **地形课程**：DreamWaQ 使用游戏式课程规则（走半块升级/不足预期半程降级），
   与 Isaac Lab `terrain_levels_vel` 一致，本实现复用该语义并增加低速保护、
   只升不降模式与逐等级统计。地形类型为 DreamWaQ 风格的缩放组合，未逐字
   复现论文地形生成器。
9. **指令课程**：Grid-Adaptive Curriculum 出自 Lee et al. 2020
   (Rapid Locomotion)，论文未公开完整概率更新细节；本实现的近似规则
   （EMA 成功率 + min_samples 门控 + 4 邻域激活 + 探索下限）在
   `command_grid.py` 模块注释中逐条记录。
10. **最小步奖励**：`minimum_step_reward=−10` 是本工程的数值保护，用于限制
    异常单步负奖励的量级；论文没有该项，训练时可通过环境配置调整。

---

## 7. 测试与验证

```bash
# 一键运行当前解释器可执行的套件
python scripts/setup_env.py --tests

# 分套件运行
PYTHONPATH=src python -m unittest discover -s tests/project -v
PYTHONPATH=src python -m unittest discover -s tests/sim2sim -v
PYTHONPATH=src python -m unittest discover -s tests/sim2real -v
bash scripts/training_python.sh -m unittest discover -s tests/training -v
```

| 套件 | 数量 | 需要 |
| --- | --- | --- |
| `tests/project` | 9 | NumPy；MuJoCo 测试按可用性跳过 |
| `tests/sim2sim` | 42 | `mujoco` |
| `tests/sim2real` | 80 | NumPy；ONNX/Tk 测试按依赖和桌面可用性跳过 |
| `tests/training` | 136 | 训练环境（torch + rsl_rl） |

测试使用模拟 SDK 和临时文件，不接入实机。已验证内容、实机前的离线验证
与未覆盖范围见 [验证记录](docs/VALIDATION.md)。

---

## 8. 文档与阅读顺序

1. [工程架构](docs/ARCHITECTURE.md)——模块边界、数据流与扩展位置
2. [训练环境](docs/TRAINING_ENV.md)——版本、Warp 兼容与检查结果
3. [sim2sim 操作](docs/SIM2SIM.md)——场地、控制与曲线
4. [实机部署](docs/SIM2REAL.md)——配置、标定、运行与安全边界
5. [只读窗口说明](docs/SIM2REAL_WINDOW.md)——启动检查、三页界面与问题分析
6. [验证记录](docs/VALIDATION.md)——已执行测试与实测数据
7. [迁移记录](docs/MIGRATION.md)——从参考工程迁移的范围与取舍
8. [策略文件说明](policies/README.md)——导出契约与模型放置

模型许可证见 [assets/robots/bpx/LICENSE](assets/robots/bpx/LICENSE)。
