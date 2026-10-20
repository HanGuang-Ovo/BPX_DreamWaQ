# BPX_DreamWaQ

BPX 四足机器人强化学习工程：基于 Isaac Lab 2.2.1 + RSL-RL 2.3.3 的
DreamWaQ 训练框架（CENet 状态估计 + PPO + 双课程学习），以及配套的
MuJoCo sim2sim 与官方 SDK sim2real 部署运行时。

## 目录

```text
BPX_DreamWaQ/
├── assets/robots/bpx/          # URDF、MJCF、网格；USD 首次训练时自动转换
├── configs/
│   ├── sim2sim/                # 平地、历史观测、固定测试地形
│   └── sim2real/               # SDK 网络、控制参数和标定模板
├── src/bpx_dreamwaq/
│   ├── assets.py              # 机器人资源入口
│   ├── paths.py               # 工作区路径
│   ├── runtime/               # 共用关节、观测、历史、推理与控制
│   ├── sim2sim/               # MuJoCo 配置、适配、运行与曲线
│   ├── sim2real/              # SDK、标定、控制、监控、录制与回放
│   ├── tasks/
│   │   ├── obs_contract.py    # 训练/部署共用观测契约（45 维单帧、450 维历史）
│   │   ├── rough_env_cfg.py   # Manager-Based 崎岖地形速度跟踪环境 + 课程开关
│   │   ├── mdp/
│   │   │   ├── rewards.py         # 复用 Isaac Lab 内置奖励
│   │   │   ├── terminations.py    # 过倾/过低终止
│   │   │   ├── curriculums.py     # 模块 A：游戏式地形难度课程
│   │   │   ├── command_grid.py    # 模块 B：Grid-Adaptive 指令课程核心（纯 torch）
│   │   │   └── commands.py        # GridAdaptiveVelocityCommand（CommandTerm）
│   │   └── agents/
│   │       ├── cenet.py               # CENet（共享编码器 + 速度头 + VAE + Decoder）
│   │       ├── dreamwaq_actor_critic.py # CENet + 非对称 Actor-Critic + 部署导出模块
│   │       ├── dreamwaq_ppo.py          # 扩展 Rollout 存储、辅助损失 PPO、Runner
│   │       └── rsl_rl_ppo_cfg.py        # 全部训练超参数与课程/AdaBoot 开关
├── scripts/
│   ├── train.py               # 训练入口（Isaac Sim headless）
│   ├── export_policy.py       # checkpoint → ONNX/TorchScript 部署策略
│   └── （sim2sim / sim2real / 检查脚本见下文）
├── tests/
│   ├── training/              # CENet、PPO、课程的离线单元测试（46 项）
│   ├── sim2sim/  sim2real/  project/
├── policies/                  # 策略文件放置说明；不包含权重
├── outputs/                   # 仿真、录制、诊断与训练日志
└── docs/                      # 架构、实机使用和迁移记录
```

## 安装与离线检查

需要 Python 3.11+。在工程根目录执行：

```bash
cd ~/BPX_DreamWaQ
python -m pip install -e '.[deploy]'
python scripts/check_project.py
```

依赖分组：基础安装仅需 NumPy；`[sim]` 增加 MuJoCo，`[inference]` 增加 ONNX Runtime，
`[deploy]` 同时安装二者，`[torchscript]` 为可选 PyTorch 后端。
不自动安装 Isaac Lab 或机器人 SDK，不修改已有 Conda 环境配置。
当前按源码工作区使用，模型与配置保留在项目目录，采用可编辑安装。

本机可直接借用现有解释器验证，无需重新安装：

```bash
~/miniconda3/envs/bpx-sim2sim/bin/python scripts/check_project.py
```

## 训练环境

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

完整版本、验证结果与兼容问题见 [训练环境说明](docs/TRAINING_ENV.md)。
本机 16 GB RAM / 8 GB VRAM：建议从 16 环境冒烟、512–1024 环境正式训练。

## DreamWaQ 训练（任务 1 + 任务 2）

### 算法结构

数据流（50 Hz 策略，200 Hz 物理，与部署一致）：

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

关键设计（依据 arXiv:2301.10602 论文结构核验 + 任务要求，详见
[设计决策与差异](#设计决策与论文差异)）：

- **Actor 不读取真实机体线速度**；真实速度仅作为 CENet 训练标签与 Critic 特权输入。
- **PPO 概率比一致性**：rollout 时实际送入 Actor 的 CENet 特征（含 AdaBoot
  混合速度）被保存到扩展 Rollout 存储，更新阶段原样复用；Actor 使用确定性
  隐变量均值 μ，不做二次采样。
- **梯度连接**：默认 `detach_cenet_from_actor=True`——Actor 的 PPO 梯度不回传
  CENet，CENet 仅由辅助损失更新（论文未明确该连接，此处采用任务伪代码的
  detach 实现，可配置 `ppo_grad_to_cenet=True`）。
- **单一优化器**：CENet 是策略的子模块，与 Actor/Critic 共用一个 Adam 优化器，
  参数必然注册；每个 minibatch 的总损失 = PPO 损失 + CENet 辅助损失。
- **done 屏蔽**：episode 终止转移的"下一观测"是 reset 后的新 episode 观测，
  预测损失与 KL 损失按 done 掩码屏蔽。
- **AdaBoot（默认开启，`reward_cv` 模式）**：训练期间按概率在真实速度与估计
  速度间切换 Actor 输入，缓解训练初期 CENet 估计不准的问题。支持 `error`
  （按速度估计误差过渡）、`reward_cv`（论文 p = 1 − tanh(CV(R))）、`linear`
  三种模式；实际送入 Actor 的速度被保存并在更新时原样复用（不破坏 PPO
  概率比）；部署始终使用估计速度。

### 训练命令

```bash
# 默认配置训练（当前默认：双课程开 + AdaBoot 开 reward_cv 模式）
bash scripts/training_python.sh scripts/train.py --headless --num_envs 512

# 短时冒烟测试（首次运行自动 URDF→USD 转换，需要几分钟）
bash scripts/training_python.sh scripts/train.py --headless --num_envs 16 --max_iterations 3

# 恢复训练
bash scripts/training_python.sh scripts/train.py --headless --num_envs 512 \
    --resume --checkpoint outputs/training/bpx_dreamwaq/<run>/model_100.pt
```

日志与 checkpoint 输出在 `outputs/training/bpx_dreamwaq/<时间戳>_<run_name>/`，
TensorBoard：`tensorboard --logdir outputs/training/bpx_dreamwaq`。

### 配置方式：命令行与配置文件

所有课程/AdaBoot 开关都有**命令行**（临时实验）和**配置文件**（改默认值）
两种用法；命令行优先级更高，在配置默认值基础上覆盖。开关在训练启动时
读取，不能中途切换；`--resume` 时换开关会从该点改变行为，对照实验建议
独立开跑。

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

# AdaBoot 切换为按估计误差过渡（任务伪代码风格）
bash scripts/training_python.sh scripts/train.py --headless --num_envs 512 --adaboot_mode error
```

**配置文件默认值**分两处：

1. 环境侧（课程开关与网格参数）——`tasks/rough_env_cfg.py` 的
   `BPXRoughEnvCfg` 类属性：

   ```python
   enable_terrain_curriculum = True        # 地形课程总开关
   num_terrain_levels = 10                 # 难度等级数（地形网格行数）
   terrain_curriculum_mode = "standard"    # "standard" 升降级 / "upgrade_only" 只升不降
   terrain_curriculum_debug = False        # 逐等级环境数、批次距离等调试日志

   enable_grid_adaptive_command_curriculum = True    # 指令课程总开关
   command_grid_resolution_vx = 0.5        # vx 网格分辨率 (m/s)
   command_grid_resolution_yaw = 0.5       # wz 网格分辨率 (rad/s)
   command_curriculum_initial_vx_range = (-0.5, 0.5)  # 初始激活范围
   command_curriculum_initial_yaw_range = (-0.5, 0.5)
   # 最大扩展范围从 command_lin_vel_x_range / command_ang_vel_z_range（±1.0）读取
   ```

   更细的门控参数（成功阈值、min_samples、EMA 系数、探索下限、指令抖动）
   在 `tasks/mdp/commands.py` 的 `GridAdaptiveVelocityCommandCfg`，
   由 `command_grid.py` 的 `GridAdaptiveCommandScheduler` 使用。

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

**AdaBoot 语义**：开启后 rollout 期间每个环境按概率 `p_boot` 决定 Actor 的
速度输入用 CENet 估计还是仿真真实值——`error` 模式 p 随 EMA 估计误差下降
从 0 升到 1；`reward_cv` 模式按论文 p = 1 − tanh(CV(R))（回合奖励变异系数，
需至少 2 个完整回合后生效，此前 p = p_min）；`linear` 模式按迭代数线性爬升。
无论哪种模式，部署导出的策略只含估计速度路径；TensorBoard 观察
`cenet/adaboot_p_boot` 与 `cenet/adaboot_est_fraction`。

### 课程学习（模块 A / 模块 B，可独立启停）

| 开关 | 位置 | 默认 | 说明 |
| --- | --- | --- | --- |
| `enable_terrain_curriculum` | `BPXRoughEnvCfg` / `--disable_terrain_curriculum` | 开 | 游戏式地形难度课程 |
| `num_terrain_levels` | 同上 / `--num_terrain_levels` | 10 | 地形难度等级数（地形网格行数） |
| `terrain_curriculum_mode` | 同上 | `standard` | `standard` 升降级 / `upgrade_only` 只升不降 |
| `terrain_curriculum_debug` | 同上 | 关 | 额外课程调试日志 |
| `enable_grid_adaptive_command_curriculum` | 同上 / `--enable_command_curriculum` / `--disable_command_curriculum` | 开 | Grid-Adaptive 指令课程 |

**模块 A：地形课程**（`tasks/mdp/curriculums.py`）。每个 episode 结束时按
环境独立判定：移动距离 > 地形块边长/2 → 升级；移动距离 < 指令预期距离/2
（且指令速度 > 0.1 m/s，静止指令不判定）→ 降级；二者互斥，等级裁剪在
[0, max]，达到最高级后随机分配等级（Isaac Lab `TerrainImporter` 内置）。
逐等级回合奖励/长度、升降级计数写入 TensorBoard（`curriculum/terrain/*`、
`Curriculum/terrain_levels/*`）。关闭课程时等级随机初始化且不再变化。

**模块 B：Grid-Adaptive 指令课程**（`tasks/mdp/command_grid.py` + `commands.py`）。
在 (vx, wz) 联合网格（默认 0.5 m/s × 0.5 rad/s 分辨率，中心在零速）上维护
每个指令组合的采样概率与跟踪统计；初始激活 [-0.5, 0.5]²，最大扩展到任务
指令量程（[-1, 1] × [-1, 1]，读取自环境配置）。单元在「区间平均 |Δvx| <
0.25 m/s 且 |Δwz| < 0.5 rad/s」的 EMA 成功率达标且采样 ≥ min_samples 后
激活 4 邻域；分布 = 90% 激活单元均匀 + 10% 全合法单元均匀（探索下限）。
vx 与 wz 从同一单元联合采样，不独立组合；vy 保持原均匀采样。统计按指令
执行区间（episode 内 10 s 重采样周期）结算，不混入整个 episode。
关闭时恢复 `UniformVelocityCommand` 均匀采样；两套课程同时关闭即为原始
PPO 基线。网格数据（概率/采样数/成功率）随 checkpoint 导出
`curriculum_<iter>.npz`，可用任意工具绘制热力图。

### TensorBoard 重点指标

- `cenet/velocity_mse`、`cenet/velocity_error_x/y/z`：速度估计精度（判断 CENet
  是否训练成功的首要指标，不能只看 PPO 奖励）。
- `cenet/prediction_loss`、`cenet/kl_loss`、`cenet/total_auxiliary_loss`：VAE 辅助损失。
- `cenet/adaboot_p_boot`、`cenet/adaboot_est_fraction`：AdaBoot 概率与估计速度占比。
- `policy/mean_reward`、`policy/episode_length`（与 `Train/*` 同源）。
- `curriculum/terrain/mean_level`、`level_XX_frac`、`level_XX_mean_reward/length`、
  `move_up/down_total`：地形课程进度与各难度表现。
- `curriculum/command/activated_cells`、`vx_range`、`yaw_range`、`expansions`、
  误差均值：指令课程扩张进度。

### 训练超参数（`tasks/agents/rsl_rl_ppo_cfg.py`）

网络：Actor/Critic MLP [512, 256, 128]；CENet 编码器 [256, 128]，速度头/VAE 头
[128]，Decoder [128, 128]；隐变量 16 维，速度 3 维，历史 10 帧——全部可配置。
PPO：lr 1e-3（自适应 KL 0.01）、γ 0.99、λ 0.95、clip 0.2、熵系数 0.01、
每迭代 24 步 × 5 epoch × 4 minibatch。辅助损失权重 λ_vel=1.0、λ_pred=1.0、
β_KL=1e-3（论文未给数值，工程默认）。梯度连接 `ppo_grad_to_cenet=False`、
隐变量模式 `latent_actor_mode="mean"`。AdaBoot 默认值见上文配置方式一节。

### 训练测试

```bash
# 离线单元测试（无需 Isaac Sim，46 项：维度/梯度/存储/课程逻辑）
bash scripts/training_python.sh -m unittest discover -s tests/training -v
```

已验证内容见 [验证记录](docs/VALIDATION.md)；短时训练冒烟命令见上文。

## 导出与部署

训练完成后导出不含特权信息的部署策略（输入 450 维历史观测 → 输出 12 维动作，
内部执行 CENet 确定性前向 + Actor）：

```bash
bash scripts/training_python.sh scripts/export_policy.py \
    --checkpoint outputs/training/bpx_dreamwaq/<run>/model_<iter>.pt --format both
```

输出 `policies/policy.onnx` 与 `policies/policy.pt`（TorchScript）。导出的
模型不依赖 Critic、真实速度、高度扫描或 VAE Decoder。**手工推理验证**
（不用特权信息）：

```python
import torch
from bpx_dreamwaq.tasks.agents.dreamwaq_actor_critic import DeployableDreamWaQPolicy
# 或直接加载 TorchScript：model = torch.jit.load("policies/policy.pt")
model = torch.jit.load("policies/policy.pt"); model.eval()
history = torch.zeros(1, 450)          # 部署端 ObservationHistory 的输出布局
action = model(history)                # (1, 12) 关节位置动作
```

也可从 checkpoint 直接构建并推理：

```python
from scripts_path import build_policy_from_checkpoint  # scripts/export_policy.py
deployable, info = build_policy_from_checkpoint("outputs/training/.../model_100.pt")
action = deployable(history_450)
```

## sim2sim

无需策略权重即可运行模型与 PD 框架检查：

```bash
python scripts/sim2sim.py --duration 2 --no-realtime
python scripts/sim2sim.py --viewer --duration 20
```

放入导出权重后显式选择推理后端：

```bash
python scripts/sim2sim.py --backend onnx --policy policies/policy.onnx --viewer
python scripts/sim2sim.py --config configs/sim2sim/bpx_terrain_history.toml \
    --backend onnx --supervisor --auto-init --viewer
python scripts/sim2sim.py --backend onnx --supervisor --gamepad --viewer \
    --velocity-plot --torque-plot --log outputs/sim2sim/run.csv
```

`--supervisor --gamepad` 时以趴姿等待 RB 起身；起身采用足端轨迹 + PD，
站立、行走和减速共用一个策略模型，站立时输入零速度指令。

| 配置 | 用途 | 策略输入 |
| --- | --- | --- |
| `bpx_history.toml` | 默认平地，保留历史观测接口 | 450 |
| `bpx_flat.toml` | 旧单帧接口参考，包含机体线速度 | 48 |
| `bpx_terrain_history.toml` | 固定三段测试地形，趴姿起步 | 450 |
| `bpx_terrain_standing.toml` | 旧单帧接口参考，固定测试地形、站姿起步 | 48 |

## sim2real

先离线检查模板和运行资源；以下命令不连接机器人：

```bash
python scripts/sim2real.py check
python scripts/sim2real.py run
python scripts/monitor.py --demo --no-policy
```

新工程不含权重，`check`/`run` 会列出缺少的统一策略模型 `policies/policy.onnx`。
监控窗口需要 Tkinter 和图形桌面。详见 [实机框架说明](docs/SIM2REAL.md)。

## 设计决策与论文差异

依据 DreamWaQ 论文（arXiv:2301.10602, ICRA 2023）核对后的实现差异与决策：

1. **历史长度**：论文 H=5；本工程取 10 帧，与部署端 `ObservationHistory`
   既有契约（450 维）一致。CENet 结构（共享编码器 + 速度头 + β-VAE 头 +
   Decoder 预测下一时刻观测）与论文一致。
2. **隐变量送入 Actor**：论文未说明用 μ 还是采样 z；本实现默认确定性 μ
   （保证 PPO 概率比一致性），可配置 `"sample"`。
3. **梯度连接**：论文未说明 PPO 梯度是否回传 CENet；默认 detach（任务伪代码
   一致），可配置。
4. **损失权重**：论文未给出 λ_vel/λ_pred/β 数值；默认 1.0/1.0/1e-3。
5. **观测归一化**：论文 VAE 先验依赖零均值单位方差输入；本工程部署契约无
   观测归一化器，故 `empirical_normalization=False`（记录为差异）。
6. **AdaBoot**：论文按回合奖励变异系数 p = 1 − tanh(CV(R))；本实现默认采用
   该论文公式（`reward_cv` 模式），并提供任务伪代码要求的按速度估计误差
   过渡的 `error` 模式与 `linear` 线性爬升模式，可配置切换。
7. **Critic 特权信息**：论文含随机扰动力 d_t；本实现 Critic 为真实线速度 +
   单帧观测 + 高度扫描（本工程扰动只有 push 事件，未把扰动力显式入观测）。
8. **地形课程**：DreamWaQ 使用游戏式课程规则（走半块升级/不足预期半程降级），
   与 Isaac Lab `terrain_levels_vel` 一致，本实现复用该语义并增加低速保护、
   只升不降模式与逐等级统计。地形类型为 DreamWaQ 风格的缩放组合（崎岖/
   方块/台阶/坡面），未逐字复现论文地形生成器。
9. **指令课程**：Grid-Adaptive Curriculum 出自 Lee et al. 2020
   (Rapid Locomotion)，论文未公开完整概率更新细节；本实现的近似规则
   （EMA 成功率 + min_samples 门控 + 4 邻域激活 + 探索下限）在
   `command_grid.py` 模块注释中逐条记录。

## 验证与阅读

```bash
PYTHONPATH=src python -m unittest discover -s tests/sim2sim -v
PYTHONPATH=src python -m unittest discover -s tests/sim2real -v
PYTHONPATH=src python -m unittest discover -s tests/project -v
bash scripts/training_python.sh -m unittest discover -s tests/training -v
```

测试使用模拟 SDK 和临时文件，不接入实机；MuJoCo 测试需要 `[sim]` 依赖；
training 测试需要训练环境（torch + rsl_rl）。
查看 [工程架构](docs/ARCHITECTURE.md)、[迁移记录](docs/MIGRATION.md)、
[验证记录](docs/VALIDATION.md)、[训练环境](docs/TRAINING_ENV.md)、
[实机框架](docs/SIM2REAL.md) 和 [策略文件说明](policies/README.md)。

模型许可证见 [assets/robots/bpx/LICENSE](assets/robots/bpx/LICENSE)。
