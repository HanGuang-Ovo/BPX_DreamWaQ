# 初始迁移记录

参考工作区：`/home/hanguang/BPX_test`。迁移日期：2026-10-20。以下描述当时的迁移范围，不代表当前功能缺项。
当前训练/部署流程见 [README](../README.md)，模块边界见 [架构](ARCHITECTURE.md)。

| 源位置 | 新位置 | 处理 |
| --- | --- | --- |
| `source/BPX_test/BPX_test/BPX_structure/` | `assets/robots/bpx/` | 复制现有模型、网格、转换配置和资源许可证 |
| `bpx_runtime/` | `src/bpx_dreamwaq/runtime/` | 保留通用实现 |
| `sim2sim/bpx_sim2sim/` | `src/bpx_dreamwaq/sim2sim/` | 保留仿真功能，删除仅用于旧导入路径的转发模块 |
| `sim2sim/run_mujoco.py` | `src/bpx_dreamwaq/sim2sim/cli.py` | 默认 Zero 后端；脚本入口移到 scripts |
| `sim2real/bpx_sim2real/` | `src/bpx_dreamwaq/sim2real/` | 保留 SDK 适配、只读监控、录制回放和显式控制 |
| 部署 TOML | `configs/sim2sim/`、`configs/sim2real/` | 清除旧日志时间戳和策略路径 |
| 部署离线测试 | `tests/sim2sim/`、`tests/sim2real/` | 调整包名、路径和无权重场景 |

实机监控进一步拆分为 `monitor.py`（入口）、`monitor_worker.py`（采样线程）、
`monitor_window.py`（Tk 窗口）和 `monitor_geometry.py`（骨架几何），便于独立阅读与修改。

没有复制旧训练任务、奖励、课程、PPO 配置、训练脚本、训练日志、运行记录、权重、缓存、Git 历史、个人本机配置或 SDK 源码/二进制包。
官方 SDK 是可选外部依赖；只在确实接入机器人时安装与 Python、架构及固件匹配的版本。

## 已知源资源缺项

源 `usd/` 目录只有 `config.yaml` 和缓存标记，没有 `bpx.usd` 及 `configuration/` 层。
因此新工程保留转换配置，不声明 USD 已可用，也不沿用失效缓存标记。
后续已在 `tasks/rough_env_cfg.py` 实现首次训练自动转换和缓存；
`BPX_FORCE_USD_CONVERSION=1` 可强制重转。资源 README 已更新，实际可用状态以项目检查报告为准。

## 行为边界

初始迁移阶段只整理部署框架，沿用旧控制参数和现有数值检查。未为实机新增控制规则，
也未执行联网、标零、起身或运动测试。只读监控与 `run --live` 仍为不同入口。
本次离线验证不能代替固件、标定和实际控制效果的核对。

已完成的验证与限制见 [VALIDATION.md](VALIDATION.md)。
