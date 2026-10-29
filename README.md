# BPX_DreamWaQ

BPX 四足机器人的 DreamWaQ 工程：在 Isaac Lab 中用 CENet + PPO 训练速度跟踪策略，
导出融合 CENet 与 Actor 的模型，在 MuJoCo 中验证，再通过官方 SDK 部署到机器人。
站立、行走和减速共用一份策略；趴卧起身由足端轨迹、逆运动学和 PD 控制完成。

本文按实际使用顺序说明工程流程。默认值按 **2026-10-29 工作区代码**核对；
已有 checkpoint 的实际配置以其元数据及运行目录 `params/` 为准，修改源码不会改变已导出的模型。

## 1. 整个工程如何运行

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

## 2. 环境准备

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

## 3. 训练：从配置到 checkpoint

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

## 4. 独立验收与导出

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

## 5. MuJoCo sim2sim 验证

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
```

不指定 `--backend` 时使用 `zero`，只验证 PD/流程，不加载模型。
CLI 默认配置是 `bpx_history.toml`；`bpx_mixed.toml` 当前配置路径为
`policies/policy_20261029_iter5349.onnx`，不等于通用导出文件 `policies/policy.onnx`。
因此换模型后应明确使用 `--policy` 或修改 TOML 的 `[paths].policy`。

先检查零指令站定，再检查前进、后退、侧移、转向和松杆，最后测试崎岖地形与楼梯。
MuJoCo 当前不会因跌倒、低高度或 Init 超时自动退出；持续运行不等于策略通过验收。
场地、手柄、状态机与日志说明见 [sim2sim](docs/SIM2SIM.md)。

## 6. 实机部署

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

## 7. 训练与部署共用的契约

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

## 8. 目录与文档索引

```text
assets/robots/bpx/   URDF、MJCF、网格、USD 缓存及资源许可证
configs/            sim2sim/sim2real TOML 与训练栈版本
scripts/            环境检查、训练、验收、导出、仿真与实机入口
src/bpx_dreamwaq/
  tasks/            Isaac Lab 环境、MDP、CENet、PPO、checkpoint 加载与验收指标
  runtime/          共用关节/观测/历史、推理、手柄、路由和起身状态机
  sim2sim/          MuJoCo 配置、地形、控制循环、CSV 和曲线
  sim2real/         SDK、标定、控制、预检、只读窗口与录制回放
  validation.py     工作区资源和配置检查
policies/           本地导出模型（权重不入 Git）
tests/              project、training、sim2sim、sim2real 测试
outputs/            本地训练、仿真、验收和诊断产物
```

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
