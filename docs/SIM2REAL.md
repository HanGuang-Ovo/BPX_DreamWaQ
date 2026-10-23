# DreamWaQ sim2real 部署

沿用 `BPX_test/sim2real` 的官方 SDK 通信、坐标转换、RB 起身、手柄输入、
Stand/Walk/STOPPING 路由和退出阻尼。在本工程中，站立、行走和减速共用
**一份融合 CENet + Actor 的 DreamWaQ ONNX 模型和连续观测历史**。

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

默认模型与当前 `configs/sim2sim/bpx_mixed.toml` 一致：
`policies/policy_20261023_iter2250_stable.onnx`。模型文件不纳入 Git；
其他机器需复制该文件，或使用本工程的 `scripts/export_policy.py` 导出模型。
选用其他权重时，可修改 `[policy].model`，或在命令行中覆盖：

```bash
python scripts/sim2real.py --policy policies/other.onnx check
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
| 策略增益 | kp=40、kd=1；起身增益 kp=80、kd=2 |
| 手柄最大指令 | vx=±0.7 m/s、vy=±0.4 m/s、wz=±0.7 rad/s |

Actor 不读取真实机体线速度、里程计高度或地形扫描，速度估计在 CENet
内部完成。IMU 默认使用 SDK 的 xyzw 四元数、body-to-world 方向、单位
安装旋转和零陀螺仪偏置；标定可覆盖通道、符号、偏移和 IMU 变换。

## 离线检查与只读预览

原 `BPX_test` 只读界面提供对应入口 `python scripts/visualize.py`（与
`scripts/monitor.py` 相同）。打开默认显示“启动前检查”页，后台验证模型，
连接后显示反馈、IMU、手柄和预览状态。界面与各列含义见
[只读窗口说明](SIM2REAL_WINDOW.md)。

```bash
# 两者均不创建 SDK 对象、不连接机器人
python scripts/sim2real.py check
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
5. Ctrl+C、指定时长结束或运行异常时退出，尝试发送三次 kp=0、kd=1 阻尼
   并断开 SDK。阻尼发送属于尽力处理，不能保证网络故障时送达。

默认持续运行，日志自动保存至 `outputs/sim2real/live-<上海时间>.jsonl`。
`--gamepad-deadman` 可要求按住 LB 才接受摇杆速度，RB 起身仍独立生效。
`--command-source constant --vx ...` 用于已站立状态的固定指令接管，
不提供 RB 起身请求。手柄断开或事件丢失时结束当前控制会话。

沿用 `BPX_test` 的实机处理范围：检查有限数和四元数模长 `[0.5,1.5]`，
不增加高度、接地、倾角、跟踪误差、身份或 verified 启动门槛。
标定中的 minimum/maximum 仅用于模型范围对照，不限制发送目标。

## 日志分析与回放

实机日志保存配置、标定、模型哈希、原始反馈、手柄指令、状态、动作、目标、
增益和策略调度延迟。`inspect` 和 `run --live` 的日志均可离线分析与回放：

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
