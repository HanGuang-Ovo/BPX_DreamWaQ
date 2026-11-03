# DreamWaQ sim2real 部署

沿用 `BPX_test/sim2real` 的官方 SDK 通信、坐标转换、RB 起身、手柄输入、
Stand/Walk/STOPPING 路由和退出阻尼。在本工程中，站立、行走和减速共用
**一份融合 CENet + Actor 的 DreamWaQ ONNX 模型和连续观测历史**。

## 与已验证 BPX_test 流程的对应关系

按 2026-11-03 的 `BPX_test/sim2real` 实现及其本机配置核对：

| 环节 | 本工程采用的方式 |
| --- | --- |
| 官方 SDK 通信 | 只读使用 `RequestRobotState`，控制使用 `JointLevelControl`；等待真实 TCP 连接后读取完整反馈 |
| 高频反馈 | 以 `getJointStateSeqHighRate` 前后序号校验关节与 IMU 快照，读取中更新则重试 |
| 坐标变换 | SDK 按腿排列、策略按关节类型排列；方向、偏移、四元数方向与 IMU 安装旋转由标定转换 |
| 等待与起身 | 保持实测关节姿态，RB/R1 触发足端五次轨迹和解析 IK，按 4 秒轨迹时间完成 |
| 接管 | 首次目标保持实测姿态，随后按接管时间平滑过渡；起身/策略增益独立混合 |
| 行为切换 | Stand → Walk → STOPPING → Stand；保留参考工程的指令限幅、斜率限制、迟滞及驻留时间 |
| 退出 | 首次尝试发送后，退出时尽力发送三次 `kp=0、kd=2` 阻尼，再断开；清理失败保留原始故障 |

BPX_test 分别使用站立、行走模型。本工程使用同一份 DreamWaQ 模型，并让历史在等待、
起身、站立、行走、减速之间连续保留。等待和起身时只按 50 Hz 构造零指令观测，不运行 Actor；
100 Hz 的额外发送帧不推进历史。起身完成后的首次推理读取最近十帧真实反馈和已发送目标对应的动作，
接管插值仍从当时的实测姿态开始，不用实测关节角覆盖“上一动作”。
正常发送成功后才提交目标作为下一帧动作，发送失败进入退出清理。

这部分逻辑分别位于 `sdk_robot.py`（SDK）、`calibration.py`（坐标）、
`behavior.py`（实机起身与增益）、`controller.py`（观测与策略）、`runner.py`（双频率发送与录制），
代码在 `src/bpx_dreamwaq/sim2real/` 下，不要求运行时导入参考工程。

## 运行环境

本机已有 `bpx-real` 环境，可直接复用其中的 NumPy、ONNX Runtime、官方
`bpx_sdk` 和 Tkinter，不需要安装 Isaac Lab、PyTorch 或 MuJoCo：

```bash
cd ~/BPX_DreamWaQ
conda activate bpx-real
python scripts/sim2real.py check
```

也可直接使用 `~/miniconda3/envs/bpx-real/bin/python`。脚本从任意当前目录
均可运行，不要求先安装本工程；可编辑安装后还可使用 `bpx-dreamwaq-real`
和 `bpx-dreamwaq-monitor`。

在其他机器上准备 Python 3.11+，执行 `python -m pip install -e '.[inference]'`，
再安装匹配当前 Python、系统架构和机器人固件的官方 SDK wheel。
参考工程的离线 wheel 位于 `../BPX_test/bpx_sdk_open-master/wheelhouse/`，
本工程运行时不导入 `BPX_test`，也不复制 SDK 二进制。

## 配置和策略

控制入口和监控入口均优先读取 `configs/sim2real/local.runtime.toml`，
不存在时读取 `configs/sim2real/runtime.toml`。`--config` 可显式指定。
需要修改网络或标定时，创建本机配置：

```bash
cp -n configs/sim2real/runtime.toml configs/sim2real/local.runtime.toml
cp -n configs/sim2real/joint_calibration.template.toml configs/sim2real/local.calibration.toml
```

将本机 runtime 的 `calibration_file` 改为 `"local.calibration.toml"`，核对
`[network]` 的机器人地址和端口。默认沿用参考工程的 `10.21.20.1`、状态
UDP 9873、关节 UDP 7895。本机配置已被 `.gitignore` 排除。

实机模板默认模型为 `policies/policy_20261023_iter2250_stable.onnx`；
当前 `configs/sim2sim/bpx_mixed.toml` 引用 `policy_20261103_iter6300_stable.onnx`，两者不同。
部署前应通过 `--policy` 或本机配置选择实际已验收的同一模型。模型文件不纳入 Git；
其他机器需复制该文件，或使用本工程的 `scripts/export_policy.py` 导出模型。
选用其他权重时，可修改 `[policy].model`，或在命令行中覆盖：

```bash
python scripts/sim2real.py --policy policies/other.onnx check

# scripts/monitor.py：打开监控窗口，显示关节反馈、姿态和策略目标等。
# --policy policies/other.onnx：指定加载的 ONNX 模型。other.onnx 是示例名称，需要换成真实存在的模型文件。
# --demo：使用模拟反馈，不连接机器人，也不发送控制命令。它用于检查界面和模型推理，不能验证机器人实际行走效果。

python scripts/monitor.py --policy policies/other.onnx --demo
```

TOML 中的模型和标定路径相对于配置文件；命令行 `--policy` 路径相对于
调用目录。全局 `--config`、`--policy` 参数放在 `check`、`run` 等子命令之前。

| 项目 | 实机契约 |
| --- | --- |
| 模型接口 | float32，`[1,450] → [1,12]` |
| 单帧观测 | 角速度 3、投影重力 3、指令 3、相对关节角 12、关节速度 12、上一帧实际动作 12 |
| 历史 | H=10，按观测项分组，每项从旧到新，首帧填满历史 |
| 策略 / 发送频率 | 50 Hz / 100 Hz；发送频率必须是策略频率的正整数倍 |
| 策略关节顺序 | FL/FR/HL/HR roll，然后四腿 pitch，然后四腿 knee |
| 对应 SDK 通道 | `[0,3,6,9,1,4,7,10,2,5,8,11]` |
| 动作目标 | `q_default + 0.5 × action`；每腿默认 `[0,0.7,-1.4]` rad |
| 策略增益 | 实机调试 kp=60、kd=1.5（待实机验证；训练基准 kp=40、kd=1）；起身增益 kp=80、kd=2 |
| 手柄最大指令 | vx=±0.7 m/s、vy=±0.4 m/s、wz=±0.7 rad/s |

Actor 不读取真实机体线速度、里程计高度或地形扫描，速度估计在 CENet
内部完成；模型内部执行训练时保存的缩放/归一化，反馈不应在外部重复缩放。IMU 默认使用 SDK 的 xyzw 四元数、body-to-world 方向、单位
安装旋转和零陀螺仪偏置；标定可覆盖通道、符号、偏移和 IMU 变换。

## 离线检查与只读预览

原 `BPX_test` 只读界面提供对应入口 `python scripts/visualize.py`（与
`scripts/monitor.py` 相同）。打开默认显示“启动前检查”页，后台验证模型，
连接后显示反馈、IMU、手柄和预览状态。界面与各列含义见
[只读窗口说明](SIM2REAL_WINDOW.md)。

```bash
# 两者均不创建 SDK 对象、不连接机器人
python scripts/sim2real.py check
# 这条命令用于执行实机部署前的软件检查。虽然叫 run，但默认不会连接机器人，也不会发送控制命令。
# 它会检查配置、ONNX 模型、依赖和输入输出尺寸等，并在终端输出检查结果，不打开监控窗口。
python scripts/sim2real.py run

# 模拟反馈 + 当前 ONNX 模型；不读取真实手柄
python scripts/monitor.py --demo

# 模拟反馈，不加载模型
python scripts/monitor.py --demo --no-policy

# 枚举电脑 USB / 蓝牙手柄
python scripts/monitor.py --list-gamepads
```

预检报告包含实际配置、模型路径与 SHA256、输入输出形状、名义站姿试推理
结果、预热耗时、SDK / 推理依赖、关节映射和频率。`ready: true` 表示当前
软件资源预检通过；机器人通信和现场坐标对应仍需通过只读反馈确认。
损坏模型、错误接口和非有限输出会列入 `deploy_blockers` / `replay_blockers`。
实际运行也会在创建 SDK 连接之前完成模型校验与预热，预热不污染控制历史。

```bash
# 只读反馈窗口；仅创建 RequestRobotState
python scripts/monitor.py

# 只读录制
python scripts/sim2real.py inspect --output outputs/sim2real/state-01.jsonl --duration 10
```

监控窗口展示反馈、姿态和未发送的策略目标。模拟反馈没有物理闭环，
不会跟随策略目标行走。SDK 连接本身可能执行时间同步。

## 实机运行

完成设备支撑、上电、网络和坐标核对后，关闭占用相同端口的只读窗口，执行：

```bash
python scripts/sim2real.py run --live

# 多手柄时指定设备；可设置运行时长和日志路径
python scripts/sim2real.py run --live --gamepad-index 0 --duration 20 \
    --output outputs/sim2real/live-01.jsonl

# 临时选择与 sim2sim 验证一致的模型
python scripts/sim2real.py --policy policies/other.onnx run --live
```

1. 启动时读取完整关节与 IMU 反馈。趴卧时保持当前关节姿态，等待 RB/R1。
   已处于 SDK `Motion` 状态时从实测关节姿态平滑接管。
2. 按 RB/R1 后执行 4 秒足端五次轨迹与解析 IK：前 1 秒建立支撑，随后抬升。
3. 起身结束进入 Stand，平滑切换增益和策略目标；左摇杆控制 vx/vy，
   右摇杆水平控制 wz，持续推杆经防抖进入 Walk。
4. 松杆后指令逐渐归零，经 STOPPING 回到 Stand。同一模型与历史始终连续。
5. Ctrl+C、指定时长结束或运行异常时退出，尝试发送三次 kp=0、kd=2 阻尼
   并断开 SDK。阻尼发送属于尽力处理，不能保证网络故障时送达。

默认持续运行，日志自动保存至 `outputs/sim2real/live-<上海时间>.jsonl`。
`--gamepad-deadman` 可要求按住 LB 才接受摇杆速度，RB 起身仍独立生效。
`--command-source constant --vx ...` 用于已站立状态的固定指令接管，
不提供 RB 起身请求。手柄断开或事件丢失时结束当前控制会话。
SDK 报告 TCP 断开或必需反馈缺失时，同样结束本次控制，不自动恢复运动。
运行循环中的反馈检查失败会记录一条 `fault`，包含控制阶段、连接状态和缺失字段；
随后 `end` 记录保留原始错误。SDK 自身提示“automatic reconnect enabled”不表示应用会恢复控制。

沿用 `BPX_test` 的实机处理范围：检查有限数和四元数模长 `[0.5,1.5]`，
不增加高度、接地、倾角、跟踪误差、身份或 verified 启动门槛。
标定中的 minimum/maximum 仅用于模型范围对照，不限制发送目标。

## 日志分析与回放

实机日志保存配置、标定、模型哈希、原始反馈、手柄指令、状态、动作、目标、
增益和策略调度延迟。`inspect` 和 `run --live` 的日志均可离线分析与回放：

`policy_due` 表示 50 Hz 策略时刻；`policy.history_updated` 表示本帧推进了历史，
`policy.inference_ran` 表示本帧实际运行了 Actor。等待和起身期间前者可为真，后者为假；
非策略发送帧保留最近一次观测与目标，并将这两个标志设为假。

```bash
python scripts/sim2real.py analyze outputs/sim2real/live-01.jsonl
python scripts/sim2real.py replay outputs/sim2real/live-01.jsonl \
    --output outputs/sim2real/replay-01.jsonl
python scripts/monitor.py --recording outputs/sim2real/live-01.jsonl
```

CLI `replay` 使用指定的固定 `--vx/--vy/--wz`（默认全零）重新推理；
窗口回放从日志读取手柄输入。两者都使用当前配置和模型，不重演关节控制，
不连接 SDK。输出采用排他创建，不覆盖已有文件。

## 验证

```bash
PYTHONPATH=src python -m unittest discover -s tests/sim2real -v
PYTHONPATH=src python -m unittest discover -s tests/project -v
```

测试使用模拟 SDK、时钟与手柄；本地模型存在且安装 ONNX Runtime 时，
还以当前 DreamWaQ 模型验证等待 → 起身 → Stand → Walk → STOPPING → Stand
完整流程。这是软件链路验证，实机运动效果需在设备上另行验收。

### 2026-11-03 本机核对结果

- 使用 `bpx-real` 环境中的官方 SDK `1.0.9` 与 ONNX Runtime：软件预检 `ready: true`，
  模型接口为 `[1,450] → [1,12]`，模型预热通过；检查过程未创建 SDK 对象。
- 与参考工程的本机配置逐项核对网络、控制、起身、监控与策略接口参数；关节映射及 IMU 转换一致。
  已创建本工程的 `local.runtime.toml` 和 `local.calibration.toml`，使用参考工程的网络/标定，
  模型仍由本工程的 `[policy].model` 选择。这两个文件被 Git 忽略；其他机器按前文准备本机配置。
- 401 个起身轨迹采样点与 BPX_test 相同；使用相同注入常量策略、跟随已发送目标的模拟反馈，
  对照 900 个控制步骤，关节目标、增益和模式一致，最大数值差为零。
  对照报告及参考文件哈希保存在本地 `outputs/sim2real/reference_alignment.json`。
- sim2real 共 83 项测试：81 通过，2 项因没有图形桌面跳过；工程测试 9 项全部通过。
  新增回归覆盖起身历史连续性、接管时反馈与已发送动作不一致，以及 SDK 清理失败时保留原始异常。

以上结果是离线对照、模拟控制与模型试推理；本次未连接机器人，也未发送实机运动指令。
