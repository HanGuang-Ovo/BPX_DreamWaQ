"""BPX 任务 MDP 扩展。

子模块：

- ``rewards``: 任务奖励（复用 Isaac Lab 内置实现，需 Isaac Sim 运行时）。
- ``terminations``: 补充终止条件（需 Isaac Sim 运行时）。
- ``curriculums``: 游戏式地形难度课程（仅依赖 torch，可离线测试）。
- ``command_grid``: Grid-Adaptive 指令课程核心调度器（仅依赖 torch）。
- ``commands``: Grid-Adaptive 指令的 CommandTerm 封装（需 Isaac Sim 运行时）。

注意：本 ``__init__`` 不导入依赖 Isaac Sim 的子模块，保证
``bpx_dreamwaq.tasks.mdp.command_grid`` 等纯逻辑模块可以在无仿真环境中
导入和单元测试。需要完整 mdp 命名空间时请在环境配置中显式导入子模块。
"""
