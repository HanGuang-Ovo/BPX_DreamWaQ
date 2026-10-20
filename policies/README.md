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

## DreamWaQ 导出契约（已实现）

训练后从 checkpoint 导出融合策略（CENet 确定性前向 + Actor）：

```bash
bash scripts/training_python.sh scripts/export_policy.py \
    --checkpoint outputs/training/bpx_dreamwaq/<run>/model_<iter>.pt --format both
```

导出模型接口与部署运行时完全一致：

- 输入：展平历史观测 `(B, 450)`，term-major、项内从最旧到最新
  （`bpx_dreamwaq.runtime.history.ObservationHistory` 的输出布局，
  即 `[角速度×10, 投影重力×10, 指令×10, 关节角×10, 关节速度×10, 上一步动作×10]`）；
- 内部：提取最新一帧 45 维 → CENet 编码（速度估计 3 + 隐变量均值 16）→
  与单帧拼接为 64 维 Actor 输入；
- 输出：12 维关节位置动作，部署侧执行 `q = q_default + 0.5 × a`；
- 不依赖 Critic、真实机体线速度、高度扫描或 VAE Decoder——导出后策略
  可以完全脱离特权信息运行。
