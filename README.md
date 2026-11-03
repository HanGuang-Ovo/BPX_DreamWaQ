# BPX_DreamWaQ

BPX 四足机器人的 DreamWaQ 工程：在 Isaac Lab 中用 CENet + PPO 训练速度跟踪策略，
导出融合 CENet 与 Actor 的模型，在 MuJoCo 中验证，再通过官方 SDK 部署到机器人。
站立、行走和减速共用一份策略；趴卧起身由足端轨迹、逆运动学和 PD 控制完成。

本文先解释项目思路和工程结构，再按实际使用顺序说明训练与部署流程。
默认值按 **2026-11-03 工作区代码**核对；
已有 checkpoint 的实际配置以其元数据及运行目录 `params/` 为准，修改源码不会改变已导出的模型。

阅读导航：[项目思路](#project-design) · [工程结构](#project-structure) ·
[完整流程](#workflow) · [环境准备](#environment) · [训练](#training) ·
[验收与导出](#export) · [MuJoCo 验证](#sim2sim) · [实机部署](#sim2real) ·
[接口契约](#contract) · [开发与文档](#development)

<a id="project-design"></a>

## 1. 项目思路：从本体感知到可部署的运动策略

### 1.1 要解决的问题

项目面向 BPX 十二关节四足机器人，目标是根据机体坐标系速度指令
`[vx, vy, wz]`，实现站立、平移、转向以及崎岖地形运动。
策略部署时使用 IMU、关节位置、关节速度和历史动作，不要求输入相机、地形扫描或真实机体线速度。
仿真训练可以获取这些额外信息，用于价值评估和状态估计监督，帮助学习可在实机使用的策略。

工程围绕以下分工组织：

| 设计 | 解决的问题 | 本工程的实现 |
| --- | --- | --- |
| 历史本体观测 + CENet | 单帧无法充分表达速度、接触和运动上下文 | 从 10 帧历史估计速度并提取隐变量 |
| 非对称 Actor-Critic | 部署输入有限，但训练可以使用仿真真值 | Actor 使用本体信息与估计特征；Critic 额外读取真实速度和地形高度 |
| PPO + 辅助学习 | 运动控制与状态估计需要不同的学习信号 | PPO 优化控制与价值网络，辅助损失训练 CENet |
| 地形课程 + 指令课程 | 地形难度和速度组合共同影响学习难度 | 分别调整地形等级与 `(vx, wz)` 采样区域，可独立开关 |
| 统一观测与动作契约 | 训练成功后仍可能因部署数据排列错误而失效 | 共用关节顺序、历史布局、动作定义，缩放随模型导出 |
| 学习策略 + 确定性起身控制 | 当前运动策略从站姿训练，趴卧起身需要单独处理 | 足端轨迹和逆运动学完成起身，再交给同一运动策略 |

这里的“隐式地形感知”通过机器人与地面的交互历史形成潜在特征；
网络不输出显式地形地图，也不把训练时的高度扫描作为部署输入。

### 1.2 CENet、Actor 与 Critic 如何协作

默认网络的数据流如下，维度均指当前训练配置：

```text
450 维本体历史 ── 网络内缩放 ── CENet 共享编码器
                                  ├── 速度估计头 ── v_hat：3 维 ─────┐
                                  └── VAE 头 ───── mu：16 维 ───────┤
                                             └── z：16 维          │
                                                  │                │
                                      Decoder 预测下一帧 45 维     │
                                      （仅辅助训练使用）           │
                                                                   ▼
当前本体帧 45 维 ── 网络内缩放 ────────────────────────────► 拼接 64 维
                                                                   │
                                                            Actor MLP
                                                                   │
                                                            12 维动作
                                                                   │
                                                  q_default + 0.5 × action
                                                                   │
                                                        PD / SDK 关节控制

真实速度 3 + 本体帧 45 + 高度扫描 187 ──► Critic ──► 状态价值 V(s)
                                                      （仅训练使用）
```

Actor 和 Critic 的隐藏层均为 `[512, 256, 128]`，默认激活函数为 ELU。
CENet 编码器隐藏层配置为 `[256, 128]`，速度头和 VAE 头各使用 `[128]`，
Decoder 隐藏层为 `[128, 128]`。
默认 Actor 使用隐变量均值 `mu`，Decoder 使用重参数化采样得到的 `z`。
**当前 Decoder 默认只输入 16 维 `z`**；可通过 `cenet_decoder_use_velocity`
改为拼接估计速度的 19 维输入，已有 checkpoint 必须保持原结构。

CENet 的辅助目标为：

```text
L_CENet = lambda_vel  × MSE(v_hat, v_true)
        + lambda_pred × MSE(predicted_next_obs, next_obs)
        + beta_kl     × KL(q(z | history) || N(0, I))
```

默认权重分别是 `1.0`、`1.0`、`0.001`。速度监督帮助估计机体运动，下一帧预测
促使隐变量保留运动上下文，KL 项约束潜在分布。损失按网络采用的缩放尺度计算；
终止转移不参与下一帧预测损失，避免把 reset 后的新状态误当作连续运动的下一帧。

PPO 与 CENet 使用独立 Adam 优化器。默认 PPO 梯度不回传 CENet；
rollout 保存实际送入 Actor 的估计特征，PPO 更新时复用这些特征，
避免更新过程中 CENet 的变化干扰新旧策略概率比的计算。
导出后的前向推理只使用缩放、CENet 编码与估计头、Actor，不需要 Critic 或 Decoder。

### 1.3 AdaBoot：训练引导与部署验收分开

训练初期速度估计可能不准确，AdaBoot 允许每个环境按概率选择估计速度或真实速度，
作为 Actor 的速度特征。当前默认开启，模式为 `error`：根据速度估计误差的滑动平均
调整使用估计速度的概率，误差下降时逐步提高该概率。代码也支持 `reward_cv` 和 `linear`。

因此，训练时“本体观测不含真实线速度”并不意味着 Actor 的特征始终不含真值：
AdaBoot 是训练阶段的显式替换分支。导出与独立验收始终使用 CENet 估计速度和确定性动作，
需要单独确认脱离真值引导后的表现。关闭 AdaBoot 和两套课程仍保留 CENet + PPO，
不是去掉估计器后的普通 Actor-Critic。

### 1.4 环境、奖励和课程为何这样组织

[`rough_env_cfg.py`](src/bpx_dreamwaq/tasks/rough_env_cfg.py) 是环境设计的主入口，
采用 Isaac Lab Manager-Based 结构，把场景、观测、动作、指令、奖励、终止、随机化和课程分别配置。
默认每回合 20 秒，以 200 Hz 执行物理仿真、50 Hz 更新策略。

地形由 8 m × 8 m 的地块组成，默认 10 个难度等级、10 列地块。
地形类型包括随机粗糙面、离散方块、上台阶、下台阶、正向坡面和反向坡面，
比例分别为 `20% / 20% / 20% / 20% / 10% / 10%`。
台阶高度范围为 `0.03–0.15 m`，踏面深度为 `0.15 m`。

奖励把速度跟踪与姿态、步态和控制代价结合起来：

| 奖励方向 | 代表项 | 设计目的 |
| --- | --- | --- |
| 指令跟踪 | 平面速度、偏航角速度跟踪 | 使机器人响应给定速度指令 |
| 零速站定 | 窄核零速跟踪、足端防滑、四足接触、默认关节姿态 | 减少零指令漂移、悬腿与站姿偏离 |
| 机体稳定 | 竖直速度、roll/pitch 角速度、离地高度、姿态水平 | 限制机体抖动和姿态偏差 |
| 步态与足端几何 | 净空、腾空时间、过长腾空、触地均衡、横纵足距 | 约束碎步、持续悬腿、偏载和不合理足距 |
| 控制正则 | 功率、功率分布、力矩、关节加速度、一阶/二阶动作差分 | 降低控制代价并平滑动作 |
| 接触与限位 | 机身接触、关节位置越界 | 抑制非预期接触及越界动作 |

各站立奖励的触发门槛不完全相同：默认姿态项要求精确零指令，部分足端项作用于
三轴指令均小于阈值的区域。修改低速采样或奖励门槛时，应一起检查慢走与站定之间的过渡。
奖励按项加权并乘策略步长后求和；环境最后将有限的单步总奖励下限设为 `-10`，不限制正奖励。
终止条件包括超时、机身接触、过倾和数值异常；当前过倾阈值以配置实值 `0.5 rad` 为准。

两套课程分别控制“在哪里走”和“按什么指令走”：

- **地形课程**：默认从等级 0 开始，按完整回合的移动任务与空间覆盖升级；
  未升级时，明确移动任务中的失败或持续跟踪不良可导致降级。零速、原地旋转和覆盖不足的正常回合保持等级。
  升级不以跟踪误差作为门槛，误差参与降级判定和日志。支持 `standard` 与 `upgrade_only`。
- **指令课程**：维护 `(vx, wz)` 联合网格，根据指令区间的跟踪统计激活相邻单元；
  在格内连续采样，同时保留探索和低速分支。`vy` 单独按配置范围采样。
  当前任务量程为 `vx ∈ [-1, 1] m/s`、`vy ∈ [-0.5, 0.5] m/s`、`wz ∈ [-1, 1] rad/s`。
- **站立与低速覆盖**：抽取指令时以 `20%` 概率选择精确零指令，网格课程下保持到回合结束；
  非站立采样中 `30%` 使用额外低速分支，`vx` 幅值为 `0.05–0.25 m/s` 并正负对称采样。
  这两类数据分别用于学习持续站定与低速响应。

观测噪声、启动时材质与 PD 增益随机化、复位姿态扰动，以及每隔 10–15 秒施加的平面推扰，
用于覆盖部分传感与动力学变化。数值保护会标记并重置异常环境，把无效转移从训练更新中隔离；
具体边界见 [数值异常隔离](docs/NUMERICAL_QUARANTINE.md)。

### 1.5 运动策略与起身流程的边界

训练从默认站姿开始；部署流程额外处理等待起身和趴卧起身。
起身使用足端轨迹、逆运动学与 PD，不依赖另一份起身网络。
起身之后，站立、行走和减速共享同一模型及连续历史，通过指令过滤与行为切换组织控制。

```text
等待起身 ── 起身请求 ──► 足端轨迹 / IK ──► 站立 ── 运动指令 ──► 行走
                                           ▲                    │
                                           └──── 减速 ◄── 松杆 ─┘
```

MuJoCo 由 `runtime/supervisor.py` 组织行为，实机由 `sim2real/behavior.py`
结合 `runtime/routing.py` 控制。两端共用观测和策略契约，但起身完成与状态切换条件不同：
仿真可以读取高度、速度与接触信息；当前实机起身按轨迹时间完成，不要求相同的高度/接地验收。
因此应分别验证起身过程、交接到策略的瞬间，以及行走后的减速站定。

### 1.6 与论文的差异和工程定位

本仓库围绕 DreamWaQ 的历史估计、潜在表征和非对称强化学习思路进行 BPX 适配，
不把工程默认参数视为论文的逐项复现。当前实现选择包括：16 维隐变量、只用 `z` 的 Decoder、
Actor 默认使用隐变量均值、独立优化器与默认梯度隔离、`error` 模式 AdaBoot，
以及 BPX 专用奖励、两套课程、固定缩放、显式 PD 和数值隔离。
这些选择均需结合实验评估；仓库具备完整训练部署流程，不等于任意 checkpoint 已经通过实机验收。

<a id="project-structure"></a>

## 2. 工程结构与模块职责

项目采用 `src/` 包布局。资源、运行配置、代码、权重和实验输出分开存放：

```text
BPX_DreamWaQ/
├── README.md                         # 项目总览与运行入口
├── pyproject.toml                    # Python 包、可选依赖、命令行入口
├── assets/robots/bpx/                # 机器人资源与许可证
│   ├── bpx/urdf/ + bpx/meshes/       # URDF 与配套网格
│   ├── mjcf/                        # MuJoCo MJCF 与配套网格
│   └── usd/                         # Isaac Lab USD 缓存与转换配置
├── configs/
│   ├── training/versions.toml        # 训练栈版本记录；不是训练超参数
│   ├── sim2sim/                     # 平地、历史接口与混合地形 TOML
│   └── sim2real/                    # 实机网络/控制配置与标定模板
├── scripts/                         # 用户执行的脚本与训练环境包装
├── src/bpx_dreamwaq/
│   ├── paths.py / assets.py         # 统一工作区路径与模型资源入口
│   ├── validation.py                # 资源、配置与工作区检查
│   ├── tasks/                       # Isaac Lab 训练任务
│   │   ├── __init__.py              # Gymnasium 任务注册
│   │   ├── rough_env_cfg.py         # 场景、MDP 与环境默认值
│   │   ├── rough_env.py             # ManagerBasedRLEnv 扩展与奖励下限
│   │   ├── obs_contract.py          # 观测维度、历史布局、最新帧切片
│   │   ├── evaluation.py            # 确定性验收指标
│   │   ├── actuators.py             # 可选执行器实现；默认使用 IdealPD
│   │   ├── mdp/
│   │   │   ├── rewards.py / terminations.py  # 自定义奖励与终止函数
│   │   │   ├── commands.py          # 指令生成与回合统计接线
│   │   │   ├── command_grid.py      # 指令网格调度核心
│   │   │   ├── curriculums.py       # 地形课程与回合评价
│   │   │   └── numerical_guard.py   # 异常状态检查与奖励隔离
│   │   └── agents/
│   │       ├── rsl_rl_ppo_cfg.py     # 网络、PPO、CENet 与 Runner 参数
│   │       ├── cenet.py              # 历史编码、速度估计、VAE 与 Decoder
│   │       ├── dreamwaq_actor_critic.py  # 网络连接及部署模型封装
│   │       ├── dreamwaq_ppo.py       # RolloutStorage、PPO、AdaBoot、Runner
│   │       ├── normalization.py     # 固定缩放与经验归一化
│   │       └── checkpoint.py        # 验收/导出共用的权重加载器
│   ├── runtime/                     # 部署侧共用的观测、推理和控制基础模块
│   ├── sim2sim/                     # MuJoCo 适配、地形、控制循环与曲线
│   │   ├── terrain.py               # 崎岖面、楼梯、模型加载与场地标签
│   │   └── complex_terrain.py       # 坡道、横坡、波浪、方块、横障碍与踏石
│   ├── sim2real/                    # SDK 适配、标定、控制、监控与录制回放
│   └── algorithms/                  # 算法实验预留；正式算法在 tasks/agents/
├── tests/                           # project / training / sim2sim / sim2real
├── docs/                            # 架构、训练、部署与诊断专题文档
├── policies/                        # 本地导出模型与模型说明，权重不入 Git
├── outputs/                         # 本地训练、验收、仿真和诊断产物
├── THIRD_PARTY_NOTICES.md            # 第三方来源与许可说明
└── task.md                          # 初始实现需求记录
```

### 2.1 模块之间如何连接

| 模块 | 关键文件与职责 | 依赖边界 |
| --- | --- | --- |
| 训练环境 | `tasks/rough_env_cfg.py` 定义任务；`rough_env.py` 扩展执行行为；`mdp/` 提供具体函数 | 环境需 Isaac Lab/Isaac Sim；部分课程核心可仅用 Torch 测试 |
| 学习算法 | `tasks/agents/` 管理估计器、策略、采样缓存、优化、保存与加载 | PyTorch、RSL-RL；验收/导出共用 checkpoint 加载逻辑 |
| 共用运行时 | `joints.py` 处理关节顺序；`observation.py` 构造单帧；`history.py` 管理历史；`policy.py` 提供 Zero/ONNX/TorchScript 后端 | 主要依赖 NumPy，推理后端按需加载 |
| 控制基础 | `init_controller.py` 提供起身轨迹与运动学；`transition.py` 提供平滑；`supervisor.py`、`routing.py` 组织行为 | 不直接持有机器人 SDK 连接 |
| MuJoCo | `config.py` 解析参数；`robot.py` 适配模型；`terrain.py` 构建地形；`runner.py` 执行循环；曲线模块展示速度/力矩 | `runtime/` + MuJoCo |
| 实机控制 | `sdk_robot.py` 对接 SDK；`calibration.py` 转换标定；`controller.py` 构造策略目标；`behavior.py`、`runner.py` 组织起身与发送 | 共享策略接口，单独处理设备反馈与通信 |
| 监控与诊断 | `preflight.py` 预检；`monitor_*` 提供只读窗口与后台检查；`diagnostics.py`、`recording.py` 支持诊断和录制 | 窗口与运动控制入口分离 |

`scripts/train.py` 负责启动 Isaac App、读取注册配置并应用命令行覆盖，再交给
`DreamWaQRunner`。仿真和实机脚本主要进入包内 CLI；部署基础模块不依赖 Isaac Sim 启动。
默认训练任务 ID 为 `BPX-Velocity-Rough-DreamWaQ-v0`，另注册
`BPX-Velocity-Rough-DreamWaQ-Play-v0` 供小规模可视化使用。

### 2.2 配置与产物的归属

训练配置来自 Python，部署配置来自 TOML，运行记录保存在输出目录；三者需要明确区分。

| 内容 | 配置/产物位置 | 作用范围 |
| --- | --- | --- |
| 环境、奖励、随机化、课程 | `tasks/rough_env_cfg.py` | 新建训练环境 |
| 网络、损失、学习率、AdaBoot | `tasks/agents/rsl_rl_ppo_cfg.py` | 新训练或当前恢复/微调过程 |
| MuJoCo 模型、地形与控制 | `configs/sim2sim/*.toml` | sim2sim，不改变训练配置 |
| 网络接口、标定、模型与控制频率 | `configs/sim2real/*.toml` | 实机控制与只读监控 |
| 实际运行配置与权重 | `outputs/training/.../params/`、`model_*.pt` | 追踪某次实验实际使用的设置 |
| 部署模型 | `policies/*.onnx`、导出的 TorchScript `.pt` | 推理，不恢复优化器或继续训练 |

路径也有不同基准：sim2sim TOML 的 `[paths]` 相对工作区根目录；sim2real TOML 中的模型与
标定路径相对该 TOML 所在目录；CLI 显式传入的相对路径以调用者当前目录为基准。
工程按源码工作区或可编辑安装使用，外部资源没有打包成独立 wheel 数据。

<a id="workflow"></a>

## 3. 整个工程如何运行

```text
机器人 URDF ──首次训练转换/缓存 USD──► Isaac Lab 并行环境
                                           │ 本体历史、特权观测、奖励
                                           ▼
                                     CENet + PPO 训练
                                           │
                           checkpoint + params + TensorBoard
                                           │
                          evaluate_policy.py 平地确定性验收
                                           │
                          export_policy.py 导出 ONNX / TorchScript
                                           │
机器人 MJCF + sim2sim TOML ────────────────► MuJoCo 验证
                                           │ 起身、站立、速度跟踪、地形
                                           ▼
                         实机配置/标定 + ONNX + 官方 bpx_sdk
                                           │
                           离线预检 → 只读反馈 → run --live
```

训练环境提供真实速度作为监督标签和 Critic 输入；部署模型从历史观测估计速度，
不需要真实线速度或地形扫描。默认训练启用 AdaBoot，采样时可能用真实速度替换
Actor 的估计速度，因此训练奖励不能代替纯 CENet 策略的独立验收。

| 阶段 | 入口 | 输入 | 产物/用途 |
| --- | --- | --- | --- |
| 环境检查 | `scripts/setup_env.py` | 当前 Python、资源、配置 | 依赖与工作区检查报告 |
| 训练 | `scripts/train.py` | Python 环境/算法配置 | 训练 checkpoint `.pt`、日志 |
| 验收 | `scripts/evaluate_policy.py` | 训练 checkpoint | 逐环境速度误差及通过状态 JSON |
| 导出 | `scripts/export_policy.py` | 训练 checkpoint | ONNX 或 TorchScript 部署模型 |
| sim2sim | `scripts/sim2sim.py` | MJCF、TOML、部署模型 | MuJoCo 仿真、CSV、曲线 |
| sim2real | `scripts/sim2real.py` | ONNX、runtime、标定、SDK | 预检、录制、回放或实机控制 |
| 只读窗口 | `scripts/monitor.py` / `visualize.py` | 模拟/SDK/录制反馈 | 启动检查、姿态与策略目标预览 |

<a id="environment"></a>

## 4. 环境准备

在项目根目录执行下面的命令。部署脚本可从外部目录调用；本文相对路径命令统一以项目根目录为基准。

| 环境 | 用途 | 依赖 |
| --- | --- | --- |
| 训练环境（本机 `isaaclab`） | 训练、验收、导出 | Isaac Lab 2.2.1、Isaac Sim 5.0、RSL-RL 2.3.3、PyTorch |
| 仿真环境（本机 `bpx-sim2sim`） | MuJoCo 与 ONNX 推理 | Python 3.11+、NumPy、MuJoCo、ONNX Runtime |
| 实机环境（本机 `bpx-real`） | SDK 控制和只读窗口 | Python 3.11+、NumPy、ONNX Runtime、官方 SDK；窗口需 Tkinter |

环境名是本机约定，不是强制名称。无需把三套依赖都装在同一个解释器中。

```bash
# 只检查，不安装
python scripts/setup_env.py
python scripts/setup_env.py --all-envs

# 安装到当前解释器：可编辑安装 + deploy extras
python scripts/setup_env.py --install
# 按需选择依赖，也可用 python -m pip install -e '.[deploy]'
python scripts/setup_env.py --install --extras sim,inference

# 资源/配置检查与当前解释器可执行的离线测试
python scripts/check_project.py
python scripts/setup_env.py --tests
```

`--install` 会调用当前解释器的 pip，可能安装或调整该环境的依赖；不加此参数仅做检查。
它不安装 Isaac Lab、Isaac Sim 或机器人 SDK。权重不随 Git 分发，缺少权重时需先训练导出或复制匹配模型。

训练复用已有 Isaac 环境：

```bash
source scripts/activate_training.sh
python scripts/check_training_env.py --require-gpu

# 无需激活当前终端的等价包装入口
bash scripts/training_python.sh scripts/check_training_env.py --smoke-test \
    --report outputs/training/environment_report.json
```

训练栈版本记录在 `configs/training/versions.toml`。其他机器可用
`BPX_CONDA_ROOT`、`BPX_TRAIN_ENV`、`BPX_ISAACLAB_ROOT`、`BPX_ISAACSIM_ROOT`
覆盖路径；Warp 兼容补丁和安装细节见 [训练环境](docs/TRAINING_ENV.md)。

<a id="training"></a>

## 5. 训练：从配置到 checkpoint

训练参数来自 Python 配置，**不读取 sim2sim TOML**：

| 要修改的内容 | 文件 |
| --- | --- |
| 地形、奖励、终止、随机化、指令、环境数量 | `src/bpx_dreamwaq/tasks/rough_env_cfg.py` |
| 网络、PPO、CENet、AdaBoot、缩放、保存间隔 | `src/bpx_dreamwaq/tasks/agents/rsl_rl_ppo_cfg.py` |
| 训练启动、命令行覆盖、恢复与微调 | `scripts/train.py` |

当前默认：1024 个环境、1500 次迭代、每次采样 24 步；两套课程开启；
AdaBoot 开启且为 `error` 模式；固定观测缩放开启、经验归一化关闭；
Decoder 只用 16 维隐变量。训练从约 0.42 m 默认站姿复位，
MuJoCo 历史配置从约 0.13 m 趴卧姿态起身，两者初始状态不同。

```bash
# 先确认仿真、奖励和反向更新链路
bash scripts/training_python.sh scripts/train.py --headless --num_envs 16 --max_iterations 3

# 正式训练示例；环境数按显存调整
bash scripts/training_python.sh scripts/train.py --headless --num_envs 512 \
    --max_iterations 3000 --seed 42 --run_name bpx_rough

# 消融：关闭课程与 AdaBoot，仍然保留 CENet + PPO
bash scripts/training_python.sh scripts/train.py --headless --num_envs 512 \
    --disable_terrain_curriculum --disable_command_curriculum --disable_adaboot
```

首次训练若缺少 USD，会从工程内 URDF 转换并缓存。需要重转时设置
`BPX_FORCE_USD_CONVERSION=1`。显存不足可减少 `--num_envs` 或增加
`--num_mini_batches`；后者同时改变每轮优化器更新次数。

默认输出目录：

```text
outputs/training/bpx_dreamwaq/<时间戳>[_<run_name>]/
├── params/env.yaml             # 本次环境配置
├── params/agent.yaml           # 算法与控制契约、加载模式、源 checkpoint
├── model_<iter>.pt             # 网络、优化器、迭代与配置；默认每 50 轮保存
├── curriculum_<iter>.npz       # 指令网格快照（启用相应课程时）
└── events.out.tfevents.*        # TensorBoard 日志
```

`BPX_LOG_DIR` 可覆盖日志根目录。监控训练：

```bash
bash scripts/training_python.sh -m tensorboard.main --logdir outputs/training/bpx_dreamwaq
```

重点观察回合奖励/长度、终止原因、CENet 速度误差、逐脚触地率、站立漂移、
课程进展和数值隔离比例。详细奖励权重、课程规则与日志指标见 [训练设计](docs/TRAINING.md)。

恢复同一训练与开始微调的区别：

```bash
# 替换为实际训练 checkpoint；下文验收和导出沿用此变量
CHECKPOINT=outputs/training/bpx_dreamwaq/你的运行目录/model_100.pt

# 恢复网络、优化器及轮次；max_iterations 是本次继续训练的轮数
bash scripts/training_python.sh scripts/train.py --headless --num_envs 512 \
    --resume --checkpoint "$CHECKPOINT" --max_iterations 500

# 加载权重和缩放/归一化状态，重建优化器，从第 0 轮开始
bash scripts/training_python.sh scripts/train.py --headless --num_envs 512 \
    --finetune --checkpoint "$CHECKPOINT" --max_iterations 500 --run_name finetune
```

微调默认 PPO 学习率 `1e-4`、关闭 AdaBoot，可用 `--learning_rate` / `--adaboot` 覆盖。
`--resume` 与 `--finetune` 互斥，`--checkpoint` 必须配合其中之一。
恢复/微调仍使用当前环境与算法配置；网络结构、Decoder 输入和缩放模式必须与源权重匹配，
不应把历史策略直接当作已经适配当前奖励和显式 PD。导出加载器则按 checkpoint 恢复模型结构。

<a id="export"></a>

## 6. 独立验收与导出

先选定实际 checkpoint（若刚完成微调，更新上面的 `CHECKPOINT`）：

```bash
bash scripts/training_python.sh scripts/evaluate_policy.py \
    --checkpoint "$CHECKPOINT" --num_envs 16 --duration 20 \
    --output outputs/evaluation/acceptance.json
```

验收从默认站姿出发，在平地测试全零指令和 `vx=0.3 m/s`，关闭噪声、材质/增益随机化与推扰。
使用确定性 CENet + Actor，不采样动作，也不使用 AdaBoot 真值替换。
每个环境都必须完成案例且没有终止；前 1 秒只排除速度评分，失败仍计入。
默认平面速度/偏航角速度 RMSE 门槛：站立 `0.05 m/s / 0.10 rad/s`，
运动 `0.15 m/s / 0.20 rad/s`。全部通过返回 `0`，未达标返回 `2`。
可重复 `--command VX VY WZ` 增加案例，并用 `--linear_tolerance`、`--yaw_tolerance` 覆盖门槛。
这一步不覆盖起身和复杂地形，之后还需做 MuJoCo 验证。

```bash
# --out 是文件前缀；生成 policies/policy.onnx 和 policies/policy.pt
bash scripts/training_python.sh scripts/export_policy.py \
    --checkpoint "$CHECKPOINT" --format both --out policies/policy
```

导出不启动 Isaac Sim。训练 checkpoint `.pt` 与 TorchScript `.pt` 用途不同：
训练 checkpoint 交给验收/导出脚本，部署只接收导出的 ONNX/TorchScript。
导出模型内部包含缩放或归一化，部署端继续输入原始观测。详见 [策略文件](policies/README.md)。

<a id="sim2sim"></a>

## 7. MuJoCo sim2sim 验证

在安装了 MuJoCo 与 ONNX Runtime 的解释器下运行。显式指定刚导出的模型，确保测试对象一致：

```bash
# 平地、无手柄：自动起身后执行固定指令
python scripts/sim2sim.py --config configs/sim2sim/bpx_history.toml \
    --backend onnx --policy policies/policy.onnx --vx 0.2 --duration 30 --viewer \
    --log outputs/sim2sim/flat.csv

# 综合场地、手柄：按 RB 起身，摇杆控制运动，松杆减速站立
python scripts/sim2sim.py --list-gamepads
python scripts/sim2sim.py --config configs/sim2sim/bpx_mixed.toml \
    --backend onnx --policy policies/policy.onnx --gamepad --viewer \
    --velocity-plot --torque-plot --log outputs/sim2sim/mixed.csv

# 直接出生在第 3 条楼梯前：级高 10 cm、踏面深 15 cm
python scripts/sim2sim.py --config configs/sim2sim/bpx_mixed.toml \
    --backend onnx --policy policies/policy.onnx --gamepad --viewer --stair-level 3

# 新增复杂路线：在随机方块前方起身，手柄沿 +X 行进
python scripts/sim2sim.py --config configs/sim2sim/bpx_mixed.toml \
    --backend onnx --policy policies/policy.onnx --gamepad --viewer --terrain-section blocks
```

综合场地还包含 `ramps`（坡道）、`cross_slope`（横坡）、`waves`（波浪）、
`blocks`（随机方块）、`hurdles`（横向障碍）、`stepping_stones`（交错踏石）六条路线。
使用 `--terrain-section 名称` 选择平地出生区，`--complex-scale 0.5` 可将新增地形高度降为一半；
TOML 的 `complex_sections` 控制启用路线与排列。尺寸和场地布局见 [sim2sim](docs/SIM2SIM.md)。

不指定 `--backend` 时使用 `zero`，只验证 PD/流程，不加载模型。
CLI 默认配置是 `bpx_history.toml`；`bpx_mixed.toml` 当前配置路径为
`policies/policy_20261103_iter6300_stable.onnx`，不等于通用导出文件 `policies/policy.onnx`。
因此换模型后应明确使用 `--policy` 或修改 TOML 的 `[paths].policy`。

先检查零指令站定，再检查前进、后退、侧移、转向和松杆，最后测试崎岖地形与楼梯。
MuJoCo 当前不会因跌倒、低高度或 Init 超时自动退出；持续运行不等于策略通过验收。
场地、手柄、状态机与日志说明见 [sim2sim](docs/SIM2SIM.md)。

<a id="sim2real"></a>

## 8. 实机部署

实机使用 ONNX 与官方 `bpx_sdk`。控制和监控优先读取
`configs/sim2real/local.runtime.toml`，不存在时使用 `runtime.toml`。
先复制并核对本机网络和关节/IMU 标定：

```bash
cp -n configs/sim2real/runtime.toml configs/sim2real/local.runtime.toml
cp -n configs/sim2real/joint_calibration.template.toml configs/sim2real/local.calibration.toml
```

将本机 runtime 的 `calibration_file` 指向 `local.calibration.toml`。
实机模板仍引用 `policy_20261023_iter2250_stable.onnx`，与当前综合仿真配置不同；
使用已验收模型时通过 `--policy` 明确选择，避免误用旧权重。

```bash
# 离线预检，不创建 SDK 连接；run 不加 --live 同样仅预检
python scripts/sim2real.py --policy policies/policy.onnx check

# 模拟反馈预览；随后只读连接实际反馈，核对通道、方向和 IMU
python scripts/monitor.py --policy policies/policy.onnx --demo
python scripts/monitor.py --policy policies/policy.onnx

# 完成现场核对并关闭占用端口的只读窗口后，显式启动实机控制
python scripts/sim2real.py --policy policies/policy.onnx run --live
```

`ready: true` 只说明软件预检通过。实机策略 50 Hz，目标发送 100 Hz；
RB 起身，左摇杆控制 vx/vy，右摇杆控制 wz，松杆后经 STOPPING 返回站立。
窗口只预览目标，不发送运动指令。实机配置、退出阻尼、录制与回放见
[实机部署](docs/SIM2REAL.md) 和 [只读窗口](docs/SIM2REAL_WINDOW.md)。

实机通信、标定、4 秒 RB 起身与退出阻尼按已验证的 `BPX_test/sim2real` 流程核对。
DreamWaQ 在等待与起身时也按 50 Hz 累积历史，接管时保留最近十帧反馈和已发送动作；
100 Hz 发送帧不重复推进历史。站立、行走和减速持续使用同一模型与历史。

<a id="contract"></a>

## 9. 训练与部署共用的契约

| 项目 | 当前约定 |
| --- | --- |
| 单帧 | 45 维：角速度 3、投影重力 3、指令 3、相对关节角 12、关节速度 12、上次动作 12 |
| 历史 | 10 帧共 450 维；term-major，每项内部旧 → 新；首帧填满窗口 |
| Actor | 最新帧 45 + 估计速度 3 + 隐变量均值 16 → 12 维动作 |
| Critic | 真实速度 3 + 本体观测 45 + 高度扫描 187 = 235 维，仅训练使用 |
| 关节顺序 | roll、pitch、knee 各组内均为 FL、FR、HL、HR |
| 动作 | `q_target = q_default + 0.5 × action`；默认 roll=0、pitch=0.7、knee=−1.4 rad |
| 训练/MuJoCo | 物理 200 Hz、策略 50 Hz，显式力矩 PD：Kp=40、Kd=1、±30 Nm |
| 速度参数 | 20 rad/s 为名义参数；训练与 MuJoCo 不施加此硬速度制动 |
| 实机 | 发送关节目标与 kp/kd，由 SDK/固件执行；不能视为与仿真动力学完全等价 |

当前默认固定缩放在网络内部执行，经验归一化关闭，二者互斥。
不要把十个完整单帧直接拼起来，也不要在部署端重复缩放。
完整布局和模块数据流见 [架构](docs/ARCHITECTURE.md)、[固定缩放](docs/FIXED_SCALING.md)。

<a id="development"></a>

## 10. 代码阅读、扩展与文档索引

### 10.1 推荐阅读顺序

1. 先读 `tasks/obs_contract.py` 与 `runtime/joints.py`，理解观测排列、历史布局和关节顺序。
2. 读 `tasks/rough_env_cfg.py`，沿场景 → 观测 → 动作 → 指令 → 奖励 → 终止 → 课程理解任务定义。
3. 读 `tasks/agents/rsl_rl_ppo_cfg.py`、`cenet.py`、`dreamwaq_actor_critic.py`，把配置与网络数据流对应起来。
4. 读 `scripts/train.py` 和 `tasks/agents/dreamwaq_ppo.py`，理解采样、辅助学习、PPO 更新及 checkpoint 保存。
5. 读 `scripts/evaluate_policy.py`、`scripts/export_policy.py` 与 `tasks/agents/checkpoint.py`，理解部署模型如何从训练权重恢复。
6. 读 `runtime/observation.py`、`history.py`、`policy.py`，再读 `sim2sim/runner.py`，跟踪一帧状态如何变成关节控制。
7. 最后读 `sim2real/calibration.py`、`controller.py`、`behavior.py`、`runner.py` 与 `sdk_robot.py`，理解实机坐标、时序和数据转换。

### 10.2 修改功能时从哪里入手

| 要做的修改 | 主要位置 | 需要联动核对 |
| --- | --- | --- |
| 新增/调整奖励 | `tasks/mdp/rewards.py`、`rough_env_cfg.py::RewardsCfg` | 权重、触发门槛、日志和奖励测试 |
| 调整地形与课程 | `rough_env_cfg.py`、`mdp/curriculums.py` | 指令回合统计、升级/降级条件和对应课程测试 |
| 调整速度采样 | `mdp/commands.py`、`command_grid.py` | 任务量程、站立概率、低速分支和部署指令范围 |
| 修改 CENet/PPO | `tasks/agents/` | 网络配置、采样特征复用、损失、checkpoint 与导出兼容 |
| 修改观测或历史长度 | `obs_contract.py`、训练观测配置、`runtime/observation.py`、`history.py` | 网络维度、布局、缩放、TOML 与模型重新训练/导出 |
| 修改动作或执行器 | `rough_env_cfg.py`、`runtime/joints.py`、两端控制配置 | 关节顺序、默认角、动作尺度、PD 与标定 |
| 增加验证地形 | `sim2sim/terrain.py`、`configs/sim2sim/` | 模型出生位置与地形测试；不会自动改变训练地形 |
| 调整实机或监控 | `sim2real/`、`configs/sim2real/` | SDK 映射、只读预览一致性与录制回放 |

现有测试按四个子目录组织：`project` 检查资源和路径，`training` 覆盖网络、奖励、课程、
数值保护与验收逻辑，`sim2sim` 覆盖历史布局、起身、地形、CLI 和曲线，
`sim2real` 覆盖标定、控制、预检、诊断和监控。离线测试验证接口和计算逻辑；
完整学习效果仍通过训练、独立验收及仿真/实机实验检查。

### 10.3 专题文档与验证命令

| 文档 | 内容 |
| --- | --- |
| [架构](docs/ARCHITECTURE.md) | 模块职责、路径规则、数据流 |
| [训练环境](docs/TRAINING_ENV.md) | 本机环境参考、版本和 Warp 兼容 |
| [训练设计](docs/TRAINING.md) | CENet/PPO、默认奖励、地形、课程、日志 |
| [固定缩放](docs/FIXED_SCALING.md) | 输入尺度与旧 checkpoint 兼容 |
| [数值异常隔离](docs/NUMERICAL_QUARANTINE.md) | 异常环境重置及样本排除 |
| [sim2sim](docs/SIM2SIM.md) | MuJoCo 场地、起身、手柄和曲线 |
| [sim2real](docs/SIM2REAL.md) / [窗口](docs/SIM2REAL_WINDOW.md) | 实机配置、控制、监控与回放 |
| [策略文件](policies/README.md) | checkpoint 与部署模型、导出接口 |
| [验证记录](docs/VALIDATION.md) | 按日期保留的测试结果和覆盖边界 |
| [步态诊断](docs/GAIT_DIAGNOSIS.md) / [迁移记录](docs/MIGRATION.md) | 历史问题与工程演进 |

离线测试按当前环境运行，缺少依赖时查看跳过原因；数量以实际输出为准：

```bash
python scripts/setup_env.py --tests
PYTHONPATH=src python -m unittest discover -s tests/project -v
PYTHONPATH=src python -m unittest discover -s tests/sim2sim -v
PYTHONPATH=src python -m unittest discover -s tests/sim2real -v
bash scripts/training_python.sh -m unittest discover -s tests/training -v
```

`task.md` 保留初始实现需求，不作为当前默认参数说明。模型资源许可证见
[assets/robots/bpx/LICENSE](assets/robots/bpx/LICENSE)。
