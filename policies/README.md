# 策略导出目录

每个运行会话只使用一份统一策略；本目录可存放多个版本的本地导出模型，权重不纳入 Git。需要推理时，放入与配置匹配的导出文件：

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
当前实机模板引用 `policy_20261023_iter2250_stable.onnx`，
`configs/sim2sim/bpx_mixed.toml` 引用 `policy_20261029_iter5349.onnx`，
通用导出默认输出 `policy.onnx`。三者不会自动同步，需显式选择经过验收的模型。模型文件不纳入 Git，复制工程到新机器时
需同时提供该权重。实机控制和监控入口也可通过 `--policy policies/other.onnx`
临时覆盖；统一 CLI 将该参数放在 `check`、`run` 等子命令之前。
不再提供独立的 stand、walk、init 策略路径和命令行参数；旧本机配置需改用上述统一键。

训练 checkpoint `.pt` 包含训练状态，不能直接交给 TorchScript 后端；
`export_policy.py --format torchscript` 生成的 `.pt` 才是部署模型。
验收与导出共用 `tasks/agents/checkpoint.py`，按源模型恢复 Decoder 结构和缩放模式。

## DreamWaQ 导出契约

训练后从 checkpoint 导出融合策略（CENet 确定性前向 + Actor）：

```bash
bash scripts/training_python.sh scripts/export_policy.py \
    --checkpoint outputs/training/bpx_dreamwaq/<run>/model_<iter>.pt --format both
```

导出模型接口与部署运行时完全一致：

- 输入：ONNX 部署契约为 float32、固定 batch 的 `(1, 450)`，term-major、项内从最旧到最新
  （`bpx_dreamwaq.runtime.history.ObservationHistory` 的输出布局，
  即 `[角速度×10, 投影重力×10, 指令×10, 关节角×10, 关节速度×10, 上一步动作×10]`）；
- 内部：提取最新一帧 45 维 → CENet 编码（速度估计 3 + 隐变量均值 16）→
  与单帧拼接为 64 维 Actor 输入；
- 内部保留 checkpoint 的固定缩放或经验归一化，外部输入原始物理量；
- 输出：`(1, 12)` 关节位置动作，部署侧执行 `q = q_default + 0.5 × a`；
- 不依赖 Critic、真实机体线速度、高度扫描或 VAE Decoder——导出后策略
  可以完全脱离特权信息运行。
