# 验证记录

日期：2026-10-20。所有测试在本机离线执行，未连接机器人。

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
| `tests/sim2sim` | 已有 `bpx-sim2sim`，Python 3.11.15 | 32 通过 |
| `tests/sim2real` | 已有 `bpx-real`，Python 3.11.15 | 53 通过，1 跳过 |
| `tests/project` | 已有 `bpx-sim2sim`，Python 3.11.15 | 7 通过 |

共运行 93 项，92 项通过，1 项因不附带策略权重而跳过。
实机相关测试注入模拟 SDK、反馈和策略，覆盖标定、起身流程、发送周期、退出处理、
只读监控、录制及离线回放，没有创建真实硬件连接。

复现命令（在项目根目录运行）：

```bash
PYTHONPATH=src ~/miniconda3/envs/bpx-sim2sim/bin/python -m unittest discover -s tests/sim2sim -v
PYTHONPATH=src ~/miniconda3/envs/bpx-real/bin/python -m unittest discover -s tests/sim2real -v
PYTHONPATH=src ~/miniconda3/envs/bpx-sim2sim/bin/python -m unittest discover -s tests/project -v
```

## 运行与安装

- Zero 后端无权重 MuJoCo 仿真完成；另执行了 6 秒自动起身与状态机检查，
  日志经历 INIT → STAND。Zero 后端只验证控制链路，不用于评价站立或行走策略。
- 在 `/tmp` 临时虚拟环境中完成无依赖、无构建隔离的可编辑安装。
- 已从 `/tmp` 工作目录运行四个安装入口的检查、短仿真、运行前检查和监控帮助。
- 直接脚本在没有项目 PYTHONPATH、且当前目录位于临时目录时运行成功。

## 尚未验证的部分

- 没有真实策略权重，因此未做实际 ONNX/TorchScript 策略效果或导出一致性验证。
- 没有启动图形窗口做视觉检查；窗口逻辑与采样线程通过离线测试验证。
- 源工程未提供 USD，未启动 Isaac Lab 或生成 USD。
- 未安装新 SDK、连接机器人、发送运动命令或验证实机效果。
- DreamWaQ 算法及训练任务尚未实现，当前不存在训练收敛结论。
