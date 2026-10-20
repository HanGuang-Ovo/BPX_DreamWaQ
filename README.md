# BPX_DreamWaQ

基于本机 `BPX_test` 整理的 BPX 强化学习工程基础。本阶段只迁移机器人模型、
共享运行时、MuJoCo sim2sim 和官方 SDK sim2real 框架。
奖励、课程和 DreamWaQ 算法均未实现；当前不提供训练入口。

## 目录

```text
BPX_DreamWaQ/
├── assets/robots/bpx/          # URDF、MJCF、网格和 USD 转换配置
├── configs/
│   ├── sim2sim/                # 平地、历史观测、固定测试地形
│   └── sim2real/               # SDK 网络、控制参数和标定模板
├── src/bpx_dreamwaq/
│   ├── assets.py              # 机器人资源入口
│   ├── paths.py               # 工作区路径
│   ├── runtime/               # 共用关节、观测、历史、推理与控制
│   ├── sim2sim/               # MuJoCo 配置、适配、运行与曲线
│   ├── sim2real/              # SDK、标定、控制、监控、录制与回放
│   ├── tasks/mdp/
│   │   ├── rewards.py         # 空：仅说明，无奖励实现
│   │   └── curriculums.py     # 空：仅说明，无课程实现
│   └── algorithms/            # 空：DreamWaQ 算法预留
├── scripts/                   # 四个直接运行入口
├── tests/                     # 迁移后的离线回归与模型验证
├── policies/                  # 策略文件放置说明；不包含权重
├── outputs/                   # 新的仿真、录制和诊断输出
├── docs/                      # 架构、实机使用和迁移记录
└── pyproject.toml
```

所有包都使用 `bpx_dreamwaq.*` 命名，与 BPX_test 的已安装包隔离。
新工程运行不读取 BPX_test 的源码、训练记录或权重。

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

检查通过表示本阶段必需资源和配置完整。报告会同时列出尚未实现的训练算法、
尚未提供的权重以及源工程缺失的 USD，不会将这些项目标记为已完成。

## sim2sim

无需策略权重即可运行模型与 PD 框架检查：

```bash
python scripts/sim2sim.py --duration 2 --no-realtime
python scripts/sim2sim.py --viewer --duration 20
```

默认 `zero` 策略只返回零动作，不代表 DreamWaQ 策略或行走效果。
默认配置使用平地、45 维单帧和 10 帧历史，完整输入为 450 维。

放入匹配的导出权重后，显式选择推理后端：

```bash
python scripts/sim2sim.py --backend onnx --policy policies/walk/policy.onnx --viewer
python scripts/sim2sim.py --config configs/sim2sim/bpx_terrain_history.toml \
    --backend onnx --supervisor --auto-init --viewer
```

可保留手柄控制和实时诊断功能：

```bash
python scripts/sim2sim.py --list-gamepads
python scripts/sim2sim.py --backend onnx --supervisor --gamepad --viewer \
    --velocity-plot --torque-plot --log outputs/sim2sim/run.csv
```

`--supervisor --gamepad` 时以趴姿等待 RB 起身。起身采用足端轨迹 + PD。
站立策略可选；未配置时沿用行走策略的零速度指令回退。

| 配置 | 用途 | 策略输入 |
| --- | --- | --- |
| `bpx_history.toml` | 默认平地，保留历史观测接口 | 450 |
| `bpx_flat.toml` | 旧单帧接口参考，包含机体线速度 | 48 |
| `bpx_terrain_history.toml` | 固定三段测试地形，趴姿起步 | 450 |
| `bpx_terrain_standing.toml` | 旧单帧接口参考，固定测试地形、站姿起步 | 48 |

测试地形只改变仿真场景，不包含课程调度。
当前接口沿用源工程，最终 DreamWaQ 网络输入和隐变量契约需在算法阶段另行确定。

## sim2real

先离线检查模板和运行资源；以下命令不连接机器人：

```bash
python scripts/sim2real.py check
python scripts/sim2real.py run
python scripts/monitor.py --demo --no-policy
```

新工程不含权重，`check`/`run` 会列出缺少的 Stand/Walk 模型。
监控窗口需要 Tkinter 和图形桌面；`--demo --no-policy` 不加载模型或 SDK。
框架保留只读录制、离线回放和 `run --live` 显式控制入口，详见 [实机框架说明](docs/SIM2REAL.md)。

## 验证与阅读

```bash
PYTHONPATH=src python -m unittest discover -s tests/sim2sim -v
PYTHONPATH=src python -m unittest discover -s tests/sim2real -v
PYTHONPATH=src python -m unittest discover -s tests/project -v
```

测试使用模拟 SDK 和临时文件，不接入实机。MuJoCo 测试需要 `[sim]` 依赖。
查看 [工程架构](docs/ARCHITECTURE.md)、[迁移记录](docs/MIGRATION.md)、
[验证记录](docs/VALIDATION.md) 和 [策略文件说明](policies/README.md)。

源工程未提供 `bpx.usd`，仅保留其转换配置。URDF 和 MJCF 可用；
接入 Isaac Lab 训练前需重新生成 USD。模型许可证见 [assets/robots/bpx/LICENSE](assets/robots/bpx/LICENSE)。
