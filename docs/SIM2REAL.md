# sim2real 框架

框架保留 BPX_test 的 SDK 读取、标定、策略控制、录制回放和只读窗口。
本次没有运行硬件连接或运动命令。

## 配置

在项目根目录准备本机配置：

```bash
cp -n configs/sim2real/runtime.toml configs/sim2real/local.runtime.toml
cp -n configs/sim2real/joint_calibration.template.toml configs/sim2real/local.calibration.toml
```

在 `local.runtime.toml` 中将 `calibration_file` 设为 `local.calibration.toml`，
核对 `[network]` 的机器人 IP、端口，以及 `[policy]` 的 Stand/Walk 模型路径。
模板不含任何机器人的实测标定值，保留源工程的默认 SDK 通道、sign=1、offset=0。
本机配置受 `.gitignore` 排除。

模型路径和标定路径相对于运行配置所在目录。
默认历史接口为 45 维单帧、10 帧历史、450 输入和 12 输出，按 50 Hz 更新。
SDK 仅在创建连接对象时导入；离线检查、分析和回放不创建 SDK 对象。

## 各入口的作用

```bash
# 离线资源检查和运行前检查
python scripts/sim2real.py --config configs/sim2real/local.runtime.toml check
python scripts/sim2real.py --config configs/sim2real/local.runtime.toml run

# 模拟反馈窗口，不连接机器人、不读取真实手柄、不加载策略
python scripts/monitor.py --demo --no-policy

# 分析已有记录；不连接机器人
python scripts/sim2real.py analyze outputs/sim2real/state.jsonl

# 离线策略回放；需要与录制接口匹配的 ONNX 模型
python scripts/sim2real.py --config configs/sim2real/local.runtime.toml \
    replay outputs/sim2real/state.jsonl --output outputs/sim2real/replay.jsonl
```

以下为后续接入硬件时使用的入口，本次未执行：

```bash
# 只读反馈窗口：默认优先使用 local.runtime.toml
python scripts/monitor.py

# 只读录制，不创建关节运动控制对象
python scripts/sim2real.py --config configs/sim2real/local.runtime.toml \
    inspect --output outputs/sim2real/state.jsonl --duration 10

# 显式进入关节控制；手柄 RB 起身，摇杆提供速度
python scripts/sim2real.py --config configs/sim2real/local.runtime.toml \
    run --live --command-source gamepad --duration 20
```

实际接入需要安装与当前 Python、系统架构和机器人固件匹配的官方 `bpx_sdk` wheel。
新工程不复制 SDK 源码或平台二进制，也不自动安装 SDK。

只读窗口没有运动入口，但官方 SDK 的连接过程可能执行时间同步。
实机控制沿用原框架：退出尝试发送阻尼；关节检查与配置中的 minimum/maximum
用途以现有代码为准，后者只用于窗口范围对照，不作为实机目标限幅。
具体标定、支撑和硬件操作仍按当前设备流程执行。
