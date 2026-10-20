# 策略导出目录

本次迁移不包含 BPX_test 的训练权重。需要推理时，主动放入与配置匹配的导出文件：

- `walk/policy.onnx`：行走策略；默认历史接口为 450 输入、12 输出。
- `walk/policy.pt`：同接口的 TorchScript 导出，可选。
- `stand/policy.onnx`：实机框架当前要求独立站立策略；默认同为 450 输入、12 输出。
- `init/`：保留给未来实验；当前起身使用足端轨迹 + PD，不需要权重。

MuJoCo 默认使用 `zero` 后端，不读取权重。ONNX 推理需要显式传入 `--backend onnx`。
sim2sim 的 `stand_policy` 默认留空，可在配置中设为 `policies/stand/policy.onnx`；
其 STAND 路由会在无独立站立策略时向行走策略发送零速度指令。

这些是从旧工程保留的部署接口，尚未定义 DreamWaQ 的编码器、隐变量或最终导出形式。
新增算法后应同步更新观测构造、配置、导出接口和回归测试。
