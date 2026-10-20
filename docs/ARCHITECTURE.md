# 工程架构

`src/bpx_dreamwaq` 是唯一代码包。资源、参数、导出权重和运行产物分别存放，
命令入口只负责转入包内 CLI。底层模块不修改 `sys.path`，也不借用 BPX_test 的 Python 包。

```text
scripts/ → sim2sim.cli → sim2sim.runner → sim2sim.robot → MuJoCo
                     ↘ runtime/（观测、历史、推理、关节、手柄、控制）
scripts/ → sim2real.cli → sim2real.runner → sim2real.sdk_robot → 官方 SDK
                       ↘ sim2real.monitor → monitor_window / monitor_worker
                                                    ↘ diagnostics / monitor_input

tasks/、algorithms/    后续训练扩展；当前不参与部署运行
```

| 层 | 职责 | 依赖边界 |
| --- | --- | --- |
| `paths.py`、`assets.py` | 工作区路径、机器人模型入口 | Python 标准库 |
| `runtime/` | 关节契约、观测、历史、推理、起身轨迹、行为切换 | NumPy；推理后端按需导入 |
| `sim2sim/` | 配置、模型适配、物理循环、CSV、曲线与测试地形 | runtime、MuJoCo |
| `sim2real/` | SDK 适配、标定、控制、录制回放、只读监控 | runtime；SDK 仅在创建连接对象时导入 |
| `tasks/mdp/` | 奖励、课程预留模块 | 当前仅模块说明 |
| `algorithms/` | DreamWaQ 算法预留位置 | 当前仅模块说明 |

## 代码阅读顺序

1. `runtime/joints.py`：策略按关节类型排列，SDK 按腿排列；二者不能直接混用。
2. `runtime/observation.py`、`history.py`：单帧与历史张量如何构造。
3. `runtime/policy.py`：Zero、ONNX、TorchScript 推理接口。
4. `sim2sim/config.py`、`robot.py`、`runner.py`：参数到 MuJoCo 控制循环。
5. `sim2real/config.py`、`calibration.py`、`controller.py`、`runner.py`：反馈到策略目标。
6. `sim2real/sdk_robot.py`：唯一直接持有 SDK 对象的适配层。
7. `sim2real/diagnostics.py`、`monitor_worker.py`：离线诊断和只读采样线程。
8. `sim2real/monitor.py`、`monitor_window.py`、`monitor_geometry.py`：窗口入口、布局刷新和骨架几何。

## 当前部署契约

- 仅一个策略模型：默认 `policies/policy.onnx`，站立、行走和减速共用模型及连续历史。
- 起身由足端轨迹 + PD 执行，等待起身时保持初始姿态，不使用独立策略权重。
- 12 维动作；`q_target = q_default + 0.5 × action`。
- 默认角按策略顺序为 roll=0、pitch=0.7、knee=-1.4。
- 默认物理步长 0.005 s；策略每 4 步更新，即 50 Hz。
- `bpx_history.toml` 和实机模板：单帧 45 维，不输入机体线速度；10 帧历史共 450 维。
- 单帧顺序：角速度 3、投影重力 3、速度指令 3、相对关节角 12、关节速度 12、上次动作 12。
- 历史按观测项分组，每项内部从最旧到最新排列；首帧填满窗口。不能直接将完整帧逐个拼接。
- `bpx_flat.toml` 和 `bpx_terrain_standing.toml` 作为旧 48 维单帧接口参考，额外输入机体线速度。
- 三段测试地形是固定 MuJoCo 场景，不包含训练课程或难度更新。

## 路径规则

- 模型资源内的相对路径沿用源文件布局。
- sim2sim TOML 的 `[paths]` 相对于工作区根目录。
- sim2sim 仅通过 `[paths].policy` 或 `--policy` 选择策略；ONNX/TorchScript 共用此入口。
- sim2real TOML 的模型、标定路径相对于 TOML 所在目录，便于复制实机配置。
- sim2real 的 `[policy].model` 为唯一策略路径；旧本机配置需迁移至此键。
- 命令行传入的配置、日志和策略覆盖路径相对于调用者当前目录。
- 使用源码工作区或 `pip install -e .`；外部资源目前不制作独立 wheel 安装包。

## 后续扩展位置

训练任务在 `tasks/` 添加，算法在 `algorithms/` 添加。奖励和课程当前没有实现，
因此不提供训练入口或 Gym 环境注册。接入 Isaac Lab 时先生成 USD，
再明确 DreamWaQ 观测、估计器、隐变量及导出契约，最后添加训练配置。
