# 验证记录

日期：2026-10-20。所有测试在本机离线执行，未连接机器人。

## sim2real 部署验证（2026-10-23）

- 当前实机默认模型与 `configs/sim2sim/bpx_mixed.toml` 一致：
  `policy_20261023_iter2250_stable.onnx`；SHA256 为
  `fd3ee9395d99653fddd3891c81f447c3473dd3c04cc6e7eb0e4b9b23150407c4`。
- `bpx-real` 环境中的 `check` 和默认 `run` 离线预检均返回 `ready: true`，
  确认 float32 `[1,450] → [1,12]`、名义站姿有限输出及 SDK / ONNX 依赖。
  从 `/tmp` 调用和 `--policy` 替换模型也通过验证。
- sim2real 80 项、sim2sim 42 项、project 9 项全部通过，共 131 项。
  当前实机模型的试推理和完整控制流程测试均实际执行，无模型相关跳过。
- 使用当前 ONNX 模型、模拟 SDK 反馈与时钟，完成 9 秒 / 900 次发送的
  `waiting_init → init → stand → walk → stopping → stand` 流程；
  不提供真实线速度或里程计高度，全部关节目标保持有限。
- 新增回归覆盖模型接口 / dtype / 非有限输出、连接前预热、预热历史隔离、
  发送耗时波动时的策略调度、实机日志分析与回放、手柄输入保存和配置覆盖。
- 本轮未创建真实 SDK 连接，未执行实机运动；上述结论仅验证软件部署链路。

复现命令：

```bash
~/miniconda3/envs/bpx-real/bin/python scripts/sim2real.py check
PYTHONPATH=src ~/miniconda3/envs/bpx-real/bin/python -m unittest discover -s tests/sim2real -v
PYTHONPATH=src ~/miniconda3/envs/bpx-sim2sim/bin/python -m unittest discover -s tests/sim2sim -v
PYTHONPATH=src ~/miniconda3/envs/bpx-sim2sim/bin/python -m unittest discover -s tests/project -v
```

## 只读可视化启动检查（2026-10-23）

- 保留原界面的关节表、骨架、曲线和参数对照，新增 `scripts/visualize.py`
  对应入口、VS Code 启动配置，以及独立“启动前检查”页。
- 在 `bpx-real` 桌面环境中，sim2real 的 80 项测试（含 2 项真实 Tk 窗口测试）
  和 project 的 9 项测试全部通过；无桌面环境中 2 项 Tk 测试按预期跳过。
- 实际创建 Tk 窗口验证三页、12 行反馈、ONNX 预览、禁用模型模式、无回调异常、
  线程退出，并截取三页 DEMO 图片逐页检查。全程禁止创建 SDK 连接对象。
- 新增状态回归覆盖 DEMO 来源、过期 / 结束会话、关节与 IMU 独立新鲜度、
  缺失数据、有效零手柄输入、模型失败和后台错误回传。
- 图形界面测试需可用桌面：

```bash
PYTHONPATH=src ~/miniconda3/envs/bpx-real/bin/python -m unittest discover \
    -s tests/sim2real -p test_monitor_gui.py -v
```

窗口截图与使用流程见 [只读窗口说明](SIM2REAL_WINDOW.md)。

## 一键环境检查（2026-10-23）

- 新增 `scripts/setup_env.py`：当前解释器与资源检查、`--install` 可编辑安装、
  `--tests` 离线测试、`--train-check` 训练栈检查、`--all-envs` 环境探测和 `--json`。
- 在 `bpx-sim2sim`、`bpx-real` 与 `isaaclab` 下实际执行默认检查均通过；
  `--tests` 运行 project 9 项、sim2sim 42 项、sim2real 80 项（training 按环境
  跳过或以 `training_python.sh` 另行运行）；`--train-check` 返回
  `bpx_training_ready: true`。
- `tests/project` 新增入口回归：从工程外调用 `setup_env.py` 返回 0 且报告通过。

## 资源与项目边界

- 29 个模型、网格、许可证和说明文件与 BPX_test 原文件的 SHA-256 一致。
- URDF 和 MJCF 的全部网格引用存在，12 个受控关节与策略契约一致。
- 四份 MuJoCo TOML 和实机 TOML 均可解析；默认仿真/实机的观测、历史、关节顺序、
  默认角、动作缩放和策略周期一致。
- 源码无 BPX_test 或旧运行时包的导入，也没有旧工程绝对路径和训练日志路径。
- 奖励与课程模块只有说明字符串，没有函数、配置、权重或调度逻辑。

## 回归测试

| 测试组 | 运行解释器 | 结果 |
| --- | --- | --- |
| `tests/sim2sim` | 已有 `bpx-sim2sim`，Python 3.11.15 | 31 通过 |
| `tests/sim2real` | 已有 `bpx-real`，Python 3.11.15 | 55 通过，1 跳过 |
| `tests/project` | 已有 `bpx-sim2sim`，Python 3.11.15 | 7 通过 |
| `tests/training` | 训练环境（`scripts/training_python.sh`，torch+rsl_rl） | 46 通过 |

共运行 140 项，139 项通过，1 项因不附带策略权重而跳过。
实机相关测试注入模拟 SDK、反馈和策略，覆盖标定、起身流程、发送周期、退出处理、
只读监控、录制及离线回放，没有创建真实硬件连接。
新增单策略回归覆盖 ONNX/TorchScript 共用路径入口、模型仅加载一次、
站立/行走/减速共享连续历史，以及起身切换后的实际动作历史。

复现命令（在项目根目录运行）：

```bash
PYTHONPATH=src ~/miniconda3/envs/bpx-sim2sim/bin/python -m unittest discover -s tests/sim2sim -v
PYTHONPATH=src ~/miniconda3/envs/bpx-real/bin/python -m unittest discover -s tests/sim2real -v
PYTHONPATH=src ~/miniconda3/envs/bpx-sim2sim/bin/python -m unittest discover -s tests/project -v
bash scripts/training_python.sh -m unittest discover -s tests/training -v
```

## 训练算法离线测试（tests/training，2026-10-20 新增）

不启动 Isaac Sim，直接验证算法模块：

- CENet：输出维度（速度 3 / 隐变量可配置）、确定性前向、Decoder 维度、
  辅助损失有限、所有子模块梯度有效、done 掩码屏蔽无效未来观测、
  logvar 裁剪。
- DreamWaQActorCritic：Actor 输入维度 45+3+16=64、动作输出 12 维、
  Critic 输入 71 维、推理确定性、特征 detach、保存的特征复现、
  checkpoint 含全部 CENet 参数、部署模块（450→12）与最新帧提取有效性。
- DreamWaQPPO：扩展 Rollout 存储字段、CENet 参数注册进优化器、
  完整 rollout+update 流程（含末步全终止的 done 屏蔽）、Actor/CENet/Critic
  参数均被更新、cenet/* 指标齐全有限、AdaBoot 三种模式与关闭状态。
- 地形课程：升/降级判定、互斥、静止指令保护、只升不降模式、
  伪环境接线的等级更新与逐等级统计。
- 指令课程：初始网格与概率归一化、联合采样边界、探索混合、
  min_samples/双轴成功门控、边界单元邻域扩展、激活单调性、EMA 更新。

## 运行与安装

- Zero 后端无权重 MuJoCo 仿真完成；另执行了 6 秒自动起身与状态机检查，
  日志经历 INIT → STAND。Zero 后端只验证控制链路，不用于评价站立或行走策略。
- 在 `/tmp` 临时虚拟环境中完成无依赖、无构建隔离的可编辑安装。
- 已从 `/tmp` 工作目录运行四个安装入口的检查、短仿真、运行前检查和监控帮助。
- 直接脚本在没有项目 PYTHONPATH、且当前目录位于临时目录时运行成功。

## 训练与导出实测（2026-10-20，Isaac Sim 5.0 headless，RTX 4060）

按任务要求逐项实测，全部通过：

1. **短时 headless 训练（默认配置）**：16 环境 × 3 迭代正常完成。
   首次启动自动从 URDF 转换 USD（`assets/robots/bpx/usd/bpx.usd`，已缓存），
   checkpoint（model_0.pt / model_2.pt）与 TensorBoard 事件文件正常保存；
   63 个标量指标含全部 `cenet/*`、`curriculum/*`、`policy/*` 项。
2. **地形课程开**：地形平均等级从 0 起步并保持（首迭代误升级伪影已用
   episode_length==0 保护修复），随机策略下 `move_down_total` 正常累计（5→11）。
3. **Grid-Adaptive 指令课程开**：初始 3×3=9 单元激活、有效 vx 范围 0.75 m/s，
   与配置一致；随机策略未达成功阈值，网格不扩张（符合门控设计）。
4. **地形课程关**：等级随机初始化（均值 ≈ 4.94/10）且全程不变，恢复基线行为。
5. **恢复训练**：从 model_2.pt 恢复后继续训练并保存 model_3.pt。
6. **导出**：`scripts/export_policy.py` 输出 ONNX 与 TorchScript
   （固定形状 (1, 450) → (1, 12)，静态 batch 满足部署诊断契约）。
7. **部署链路**：导出的 `policies/policy.onnx` 被本仓库 sim2sim 运行时
   （bpx-sim2sim 环境 + ONNX Runtime）正常加载：450 维历史输入、
   50 Hz 策略、12 维动作、200 Hz PD 全链路运行。
8. **sim2real 离线诊断**：实际 ONNX 模型通过维度契约检查（56 项测试通过）。

## 250 迭代实质训练实测（2026-10-20，256 环境，RTX 4060，约 35 分钟）

完整跑通 250 个 PPO 迭代（fps ≈ 2900，checkpoint 每 50 迭代保存），关键趋势：

| 指标 | 起始 | 结束 | 说明 |
| --- | --- | --- | --- |
| `policy/mean_reward` | −1.24 | −0.40 | 策略奖励持续提升 |
| `policy/episode_length` | 20 | 24.9（中途最低 ≈9） | 探索期回落后恢复并超过初始 |
| `cenet/velocity_mse` | 0.091 | 0.017 | 速度估计精度提升约 5 倍 |
| `cenet/velocity_error_x/y/z` | 0.18/0.17/0.33 | 0.09/0.08/0.11 m/s | 三轴误差同步下降 |
| `cenet/prediction_loss` | 22.9 | 10.5 | 下一观测预测持续学习 |
| `curriculum/terrain/mean_level` | 0 | 0 | 步态未稳前不升级，符合课程设计 |

观察：`cenet/kl_loss` 初期峰值约 140 后回落至 55（β=1e-3 时加权贡献约 0.05，
远小于 PPO 损失；如需更强正则可增大 `beta_kl`）。地形课程在 250 迭代内
未触发升级——episode 尚短，属预期行为，收敛训练（1000+ 迭代）中才会爬级。

## 尚未验证的部分

- 未进行千迭代量级的收敛训练与步态质量/地形爬级评估（需数小时 GPU 时间）。
- 没有启动图形窗口做视觉检查；窗口逻辑与采样线程通过离线测试验证。
- 未安装新 SDK、连接机器人、发送运动命令或验证实机效果。

## 训练环境检查

同日另行配置工作区训练入口，复用 Python 3.11 / Isaac Lab 2.2.1 / Isaac Sim 5.0 /
RSL-RL 2.3.3 / PyTorch 2.7.0+cu128，并在工作区内安装 Warp 1.8.1 兼容补丁。
CUDA 矩阵运算、RSL-RL Runner 导入、Isaac Sim 无界面启动及十步 GPU 物理仿真通过。
本机 VS Code 终端激活、从工作区外调用包装脚本、错误 Python 环境诊断均已验证；
`tests/project` 七项通过，项目资源检查无错误。
详细结果与限制见 [TRAINING_ENV.md](TRAINING_ENV.md)，本工程仍未实现 BPX 训练任务。
