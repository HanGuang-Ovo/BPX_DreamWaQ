# MuJoCo sim2sim

按 2026-10-29 代码核对；文末有日期的结果保留为历史记录。

## 运行

在项目根目录执行，Python 环境需要 `mujoco` 和 `onnxruntime`。
检查手柄、运行综合场地：

```bash
python scripts/sim2sim.py --list-gamepads
python scripts/sim2sim.py --config configs/sim2sim/bpx_mixed.toml \
    --backend onnx --policy policies/policy.onnx --gamepad --viewer \
    --log outputs/sim2sim/run.csv
```

多个手柄时增加 `--gamepad-index N`，也可显式指定
`--gamepad-device /dev/input/eventN`。默认综合配置运行 300 秒，关闭窗口或 Ctrl+C 退出。

1. 机器人以机身高度约 0.13 m 的趴卧姿态出生，保持关节位置；此时摇杆不驱动策略。
2. 按 RB（Linux 按钮码 311）启动 Init：约 1 秒收脚至低位支撑，再用 3 秒抬升。
3. 四足接触、轨迹完成且姿态/高度连续满足 0.5 秒后，平滑切换到策略。
4. 左摇杆前后为 vx、左右为 vy；右摇杆水平方向为 wz。松开摇杆后减速并站立。

`--gamepad-deadman` 启用 LB 按住才接受移动指令；RB 起身独立于 LB。
默认速度上限为 1.0 m/s、0.5 m/s、1.0 rad/s，可通过 `--gamepad-max-vx/vy/wz` 调整。
非标准手柄可用 `--gamepad-*-axis` 和 `--gamepad-*-sign` 调整映射。
输入断开或丢失会结束本次仿真，并在终端报告错误。

无手柄时自动起身，使用固定速度指令：

```bash
python scripts/sim2sim.py --config configs/sim2sim/bpx_mixed.toml \
    --backend onnx --vx 0.2 --duration 30 --viewer
```

`--auto-init` 可让手柄模式也自动起身。`--direct-policy` 跳过状态机，供站姿接口对拍使用。
默认 `zero` 后端只检查 PD 和流程，不产生策略步态。
CLI 默认读取 `bpx_history.toml`（平地，450 维）；综合场地需显式传 `--config`。
`bpx_mixed.toml` 当前引用 `policy_20261029_iter5349.onnx`，
而导出脚本默认生成 `policy.onnx`，使用新导出时请显式传 `--policy`。

MuJoCo 中已取消高度过低、倾角过大、躯干触地和 Init 超时导致的自动退出。
策略接管后持续响应手柄；起身未完成时继续执行 Init 控制，直到满足接管条件。
仿真在设定时长结束、关闭窗口或 Ctrl+C 时退出；输入/数值异常仍会报错。
旧的 `--terminate-on-fall`/`--no-terminate` 参数仅保留命令兼容，不再改变行为。

## 场地和难度

`--terrain` 覆盖 TOML 的 `terrain.kind`：

| 值 | 场地 |
| --- | --- |
| `flat` | 平地 |
| `three_level` | 平地 → ±2 cm 崎岖 → ±4 cm 崎岖，含平地回程区域 |
| `stairs` | 平地和多档上下楼梯 |
| `mixed` | 同时包含崎岖路线和楼梯路线 |

综合场地出生点为 `(0, 0)`，面向 +X。崎岖路线位于 `x=2..10 m`、`y=-2..2 m`，
边缘平滑过渡。`bpx_mixed.toml` 中楼梯起点均为 `x=2 m`，终点为 `x=4.4 m`，路线如下：

| 等级 | 单级高度 | 顶部高度 | 中心线 y | 颜色 |
| --- | --- | --- | --- | --- |
| 1 | 3 cm | 15 cm | -4.5 m | 绿 |
| 2 | 8 cm | 40 cm | -7.5 m | 蓝 |
| 3 | 10 cm | 50 cm | -10.5 m | 橙 |
| 4 | 15 cm | 75 cm | -13.5 m | 绿 |

每条楼梯宽 2 m，踏面深 0.15 m，上升 5 个高差，顶部平台长 1.2 m，再下降 5 个高差。
各楼梯为独立 box 碰撞体，保留竖直踢面；崎岖路面采用连续高度场，保留负高度谷地。
地形标签显示在 viewer 内，CSV 记录 `terrain_region`、`ground_z`、`base_clearance`。
高度判定使用机身正下方地面，跨台阶边缘时不是四足支撑面的平均高度。

直接选择楼梯前方的平地出生区，无需先横移到对应路线：

```bash
python scripts/sim2sim.py --config configs/sim2sim/bpx_mixed.toml \
    --backend onnx --gamepad --viewer --stair-level 1
# 自定义高度；出生等级必须在高度列表范围内
python scripts/sim2sim.py --terrain stairs --step-heights 0.03 0.06 0.10 \
    --stair-level 2 --backend onnx --gamepad --viewer --duration 300
```

TOML `[terrain]` 可设置 `step_heights`、`step_count`、`step_width`、`stair_width`、
`platform_width`；`--terrain-seed` 控制崎岖路面的可复现随机种子。
Init 轨迹面向水平趴卧状态，出生区安排在平地；侧翻/仰翻或 Init 超时不再自动结束仿真。

楼梯已统一并入综合场地，使用同一个配置即可测试崎岖地面和四档楼梯。
例如，选择单级高 10 cm、踏面深 15 cm 的路线：

```bash
python scripts/sim2sim.py --config configs/sim2sim/bpx_mixed.toml \
    --backend onnx --gamepad --viewer --stair-level 3
```

改用 `--stair-level 4` 可选择单级高 15 cm 的路线；不指定该参数时，
仍在崎岖路线前的平地出生。`step_width=0.15` 指沿行走方向的单级踏面深度，
`stair_width=2.0` 指整条楼梯的横向宽度。

## 关节与控制契约

沿用 `/home/hanguang/BPX_test` 的映射与参数，实现均位于本工程，不依赖参考工程导入。
策略动作/关节观测按类型优先：roll 的 FL/FR/HL/HR，pitch 的 FL/FR/HL/HR，
最后 knee 的 FL/FR/HL/HR。MuJoCo 的 qpos、qvel、actuator 均按名称查询，
加载 TOML 时拒绝不符合该策略顺序的关节列表。

| 参数 | 策略阶段 | Init 阶段 |
| --- | --- | --- |
| Kp / Kd | 40 / 1 | 80 / 2 |
| 力矩上限 | 30 Nm | 30 Nm |
| 动作零点 | roll=0、pitch=0.7、knee=-1.4 rad | 同一动作零点，目标来自足端轨迹 IK |
| 动作换算 | q_target = q_default + 0.5 × action | 相同 |
| 更新频率 | 策略 50 Hz，PD/物理 200 Hz | 轨迹/PD/物理 200 Hz |

默认部署输入为 10 帧 × 45 维 = 450 维，term-major、每项由旧到新。
输入不含真实线速度和地形扫描。Init 和等待阶段仍累计实际观测/实际施加动作历史；
切换时动作与 PD 增益在 0.3 秒内过渡。旧的 48 维配置只适用于相应的旧策略。

## 配置选择与曲线

| 配置 | 接口 | 场地 |
| --- | --- | --- |
| `bpx_history.toml` | 450 维历史 | 平地，CLI 默认 |
| `bpx_terrain_history.toml` | 450 维历史 | 三段固定测试地形 |
| `bpx_mixed.toml` | 450 维历史 | 崎岖 + 四档上下楼梯 |
| `bpx_flat.toml` | 48 维单帧，含真实线速度 | 旧接口平地参考 |
| `bpx_terrain_standing.toml` | 48 维单帧，含真实线速度 | 旧接口测试地形参考 |

当前 DreamWaQ 导出模型应配合 450 维配置，不能直接使用旧 48 维配置。
`--velocity-plot` 对照指令与实际速度，`--torque-plot` 查看 12 关节输出力矩；
`--velocity-window` / `--torque-window` 设置 1–30 秒窗口，默认 10 秒。
`--log` 保存策略频率 CSV；`--no-realtime` 取消实时限速，便于离线仿真。
训练和 MuJoCo 使用相同名义 PD，但物理引擎、接触与初始状态不同，不能只凭参数一致判断迁移成功。

## 验证记录（2026-10-21）

```bash
PYTHONPATH=src python -m unittest discover -s tests/sim2sim -v
PYTHONPATH=src python -m unittest discover -s tests/project -v
python scripts/check_project.py
```

sim2sim 39 项、工程 7 项测试通过。新增覆盖：各等级上下楼梯射线高度、竖直踢面、
四足/躯干接触、12 维 one-hot 关节目标/力矩/观测映射、CLI 地形与 RB 接线，
以及真实 MuJoCo 动力学下等待触发 → Init → 策略 → 行走 → 松杆站立。
另通过 sim2real 56 项回归测试。

已从 `outputs/training/bpx_dreamwaq/20261021_004207/model_3100.pt` 导出本地
`policies/policy.onnx` 和 `policies/policy.pt`，输入 450、输出 12。重新导出：

```bash
bash scripts/training_python.sh scripts/export_policy.py \
    --checkpoint outputs/training/bpx_dreamwaq/20261021_004207/model_3100.pt --format both
```

使用该 ONNX 策略、自动起身、固定 vx，在综合场地完成以下实际推理冒烟。
以下为当时 30 cm 踏面、3/8/15 cm 三档楼梯的历史记录，未验证当前 15 cm 踏面场地：

| 路线 | 指令 | 结果 |
| --- | --- | --- |
| 崎岖 | 0.4 m/s，32 s | 到达两档崎岖区域，未触发跌倒停止；存在横向漂移，后段进入回程区 |
| 3 cm 楼梯 | 0.3 m/s，32 s | 到达 15 cm 顶部并继续行走，下降末段偏离路线侧边 |
| 8 cm 楼梯 | 0.3 m/s | 首级附近倾倒，12.96 s 因 `supervisor_unsupported_init_pose` 停止 |
| 15 cm 楼梯 | 0.3 m/s，32 s | 首级附近受阻并侧移，未完成楼梯路线 |

日志在 `outputs/sim2sim/policy3100_*.csv`，场地离屏渲染在 `outputs/sim2sim/course.png`。
这些结果验证了加载、起身、策略接管和地形交互；该次测试的策略未稳定通过较高楼梯。
当时未检测到实体 Linux 手柄，RB/摇杆流程通过模拟输入和接线测试验证，仍需连接手柄实测。
