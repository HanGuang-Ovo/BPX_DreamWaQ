# 策略导出目录

本目录只保留一个策略模型入口，不包含训练权重。需要推理时，放入与配置匹配的导出文件：

- `policy.onnx`：仿真和实机共用的策略；默认历史接口为 450 输入、12 输出。
- 使用 TorchScript 导出时，可放入 `policy.pt`，通过同一个路径配置或 `--policy` 参数选择。

站立、行走和减速共用同一个模型实例与观测历史；站立时速度指令为零。
起身使用足端轨迹 + PD，等待起身时保持初始姿态，这两个阶段不调用神经网络策略。

sim2sim 在 TOML 的 `[paths].policy` 配置模型路径，也可用 `--policy` 临时覆盖：

```bash
python scripts/sim2sim.py --backend onnx --policy policies/policy.onnx --viewer
python scripts/sim2sim.py --backend torchscript --policy policies/policy.pt --viewer
```

MuJoCo 默认使用 `zero` 后端，不读取权重。实机和只读监控使用 ONNX，
在 `configs/sim2real/runtime.toml` 的 `[policy].model` 配置同一个模型路径。
不再提供独立的 stand、walk、init 策略路径和命令行参数；旧本机配置需改用上述统一键。

当前保留部署观测接口，尚未定义 DreamWaQ 的编码器、隐变量或最终导出形式。
新增算法后应同步更新观测构造、配置、导出接口和回归测试。
