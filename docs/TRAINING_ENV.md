# 训练环境配置与检查

环境实测快照日期：2026-10-20；流程说明按 2026-10-29 代码更新。
工作区复用本机已有安装，训练依赖不复制进源码。BPX 训练、验收与导出接线已实现。
下表是当时本机环境，不代表其他机器或当前依赖探测结果；请运行检查脚本确认。

## 本机版本

| 组件 | 当前配置 |
| --- | --- |
| 系统 | Ubuntu 22.04.5，x86_64 |
| Python | `/home/hanguang/miniconda3/envs/isaaclab/bin/python`，3.11.16 |
| Isaac Lab | `/home/hanguang/IsaacLab-2.2.1`，仓库版本 2.2.1 |
| Isaac Sim | `/home/hanguang/Isaac_Sim`，VERSION 为 `5.0.0-rc.45+release.23960.184afb15.gl` |
| RSL-RL | `rsl-rl-lib==2.3.3`，导入名 `rsl_rl` |
| PyTorch / torchvision | Isaac Sim 自带的 `2.7.0+cu128` / `0.22.0+cu128` |
| NumPy / Gymnasium | 运行时选中 `1.26.0` / `1.2.0` |
| Warp | 工作区 `.training-deps/` 中的 `warp-lang==1.8.1` |
| 辅助依赖 | Hydra、TensorBoard、ONNX 已安装并可定位 |
| GPU / 驱动 | RTX 4060 Laptop，8 GB 显存；580.178.04 |
| 内存 / 磁盘 | 约 16 GB RAM；检查时工作区磁盘剩余约 243 GB |

Lab 仓库的 2.2.1 与 Python 包 `isaaclab` 的 0.45.9 是不同层次的版本号。
Isaac Sim 的完整 VERSION 包含候选构建标记，检查报告保留该原始信息。
这套配对按 [Isaac Lab v2.2.1 安装说明](https://github.com/isaac-sim/IsaacLab/blob/v2.2.1/docs/source/setup/installation/index.rst)
及该版本 [RSL-RL 依赖声明](https://github.com/isaac-sim/IsaacLab/blob/v2.2.1/source/isaaclab_rl/setup.py) 配置。

## 工作区入口

```bash
cd ~/BPX_DreamWaQ
source scripts/activate_training.sh
python scripts/check_training_env.py --require-gpu

# 无需预先激活终端的等价入口
bash scripts/training_python.sh scripts/check_training_env.py --require-gpu

# 运行有界的仿真检查；不使用 BPX、不连接 SDK、不训练策略
timeout --kill-after=10s 120s bash scripts/training_python.sh -u scripts/check_training_env.py \
    --smoke-test --report outputs/training/environment_report.json
```

激活脚本清除继承的 Python 路径，加入本工程 `src`、工作区依赖和 Isaac Sim 的包路径，
并优先选择 Conda 的 C++ 运行库。它不会安装或升级已有 Conda 包。
其他机器可通过 `BPX_CONDA_ROOT`、`BPX_TRAIN_ENV`、`BPX_ISAACLAB_ROOT`、`BPX_ISAACSIM_ROOT` 覆盖路径。

`.vscode/settings.json` 配置了默认 Python、Pylance 搜索路径和训练终端配置。
新建终端时会加载激活脚本；当前已有终端需手动 `source`。
如果 VS Code 之前保存过其他解释器选择，需要选择上表中的 Python 3.11 路径。
`.vscode/training.env` 提供运行/调试使用的本机环境变量，受 Git 忽略。
本工程源码通过 `PYTHONPATH` 加载，已有 Conda 中的旧 BPX_test 包不参与本工程入口。

## Warp 兼容修复

Isaac Sim 自带 Warp 1.7.1，在当前 R580 驱动上出现 `cuDeviceGetUuid` 初始化错误。
Conda 中已有的 Warp 1.16.0 虽可使用 CUDA，却移除了 Isaac Sim 5.0 使用的 `warp.types.array`。
工作区因此固定使用同时保留旧 API、包含驱动修复的 1.8.1；
相关修复见 [NVIDIA Warp 官方问题记录](https://github.com/NVIDIA/warp/issues/940)。

补丁安装方式如下；当前工作区已完成安装：

```bash
~/miniconda3/envs/isaaclab/bin/python -m pip install --no-deps \
    --target .training-deps -r requirements/training-fixes.txt
```

`scripts/training_bootstrap/sitecustomize.py` 在 Kit 修改搜索路径之前预载此版本。
此配置只由训练入口启用，不影响原有 MuJoCo/实机环境。
删除 `.training-deps/` 后，需先重新安装补丁再使用训练环境。

系统 `nvcc` 为 CUDA Toolkit 13.2；PyTorch 使用其随包提供的 CUDA 12.8 运行库。
当前检查不编译 CUDA 扩展，不需要把系统 Toolkit 降级或重新安装 PyTorch。

## 环境检查范围与训练接线

GPU 矩阵运算、`rsl_rl.runners.OnPolicyRunner` 与 `isaaclab.app.AppLauncher` 导入已通过。
修复 Warp 后，Isaac Sim 无界面启动、立方体十步 CUDA 物理仿真，以及
`isaaclab_rl.rsl_rl.RslRlVecEnvWrapper` 导入均已通过；该环境检查本身不运行 BPX 训练。后续训练记录见 [验证记录](VALIDATION.md)。
GPU 在受限沙箱内不可见时应在主机终端复查，不应仅凭该沙箱的 `nvidia-smi` 失败重装驱动。
仿真检查结果记录在 `outputs/training/environment_report.json` 的 `simulation` 字段。
`tested=false` 或 `result=running` 表示未完成，不能当作物理仿真通过。

通用 `pip check` 仍会列出 Isaac Sim 捆绑包的元数据问题，例如缺少 `usd-core`、
`docstring-parser`、`tenacity`、`trio` 等，以及 lxml/scipy/cryptography/s3transfer 版本约束。
这些包涉及 USD 导出、绘图、浏览器或云服务等额外功能，不能据此批量升级共享环境。
Kit 自带 `pxr`，不要为消除元数据提示直接覆盖它的 USD 库；对应功能需单独验证。

启动本工程 BPX 强化学习训练的原缺项（2026-10-20 已全部实现）：

1. BPX USD：`tasks/rough_env_cfg.py` 在首次训练启动时通过 UrdfConverter 从 URDF
   自动转换（转换参数由 `_ensure_bpx_usd()` 定义，`usd/config.yaml` 为生成记录），结果缓存复用；
   可用环境变量 `BPX_FORCE_USD_CONVERSION=1` 强制重转。
2. Isaac Lab 向量环境、观测（45 维单帧 + 450 维历史 + 235 维特权）、动作、指令、
   复位、终止与 Gymnasium 注册：`tasks/rough_env_cfg.py`、`tasks/__init__.py`。
3. 奖励（复用 Isaac Lab 内置 + 任务层 feet_air_time）与 PPO 超参数：
   `tasks/mdp/rewards.py`、`tasks/agents/rsl_rl_ppo_cfg.py`。
4. DreamWaQ CENet、速度估计、VAE 隐变量与辅助损失：`tasks/agents/` 三个模块，
   含扩展 Rollout 存储、AdaBoot 与部署导出（450 维 → 12 维）。
5. 训练与导出入口：`scripts/train.py`、`scripts/export_policy.py`。

课程学习（地形 + Grid-Adaptive 指令）已实现且可独立启停；
当前启用材质、PD 增益、复位位姿/关节、观测噪声与推扰；不启用质量/质心随机化。
训练不需要已训练权重，也不需要机器人 SDK、实机网络或 ROS。
具体模型的收敛和部署结论应查看对应运行与验收报告，不能由环境检查推断；见 [VALIDATION.md](VALIDATION.md)。

本机 16 GB RAM / 8 GB VRAM 低于该版本文档推荐的 32 GB / 16 GB。
初次任务调试建议无界面、无相机，从 16 个环境开始测量占用，再逐步增加，
不要直接沿用旧工程数千个并行环境的配置；该并行数量尚未在 BPX 任务上验证。
