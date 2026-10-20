#!/usr/bin/env python3
"""Check the local training stack; optionally run ten headless GPU physics steps.

Usage: bash scripts/training_python.sh scripts/check_training_env.py --require-gpu
       bash scripts/training_python.sh scripts/check_training_env.py --smoke-test
"""

import argparse
import ast
import faulthandler
from datetime import datetime
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def package(name, module):
    spec = importlib.util.find_spec(module)
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        version = None
    return {"version": version, "path": spec.origin if spec else None}


def project_blockers():
    missing = []
    if not (ROOT / "assets/robots/bpx/usd/bpx.usd").is_file():
        missing.append("缺少 BPX USD：需将现有 URDF 转换为 Isaac Lab 可用的机器人资源")
    task_root = ROOT / "src/bpx_dreamwaq/tasks"
    if not any("gym.register" in p.read_text(encoding="utf-8") for p in task_root.rglob("*.py")):
        missing.append("缺少 BPX Isaac Lab 训练环境、观测/动作/复位配置及 Gymnasium 任务注册")
    for name, label in (("rewards", "奖励函数"),):
        tree = ast.parse((task_root / "mdp" / f"{name}.py").read_text(encoding="utf-8"))
        if not any(isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)) for node in tree.body):
            missing.append(f"尚未实现{label}")
    algorithm_root = ROOT / "src/bpx_dreamwaq/algorithms"
    if not any(p.name != "__init__.py" for p in algorithm_root.rglob("*.py")):
        missing.append("尚未接入 PPO/OnPolicyRunner 配置与 DreamWaQ 编码器、估计器和训练逻辑")
    if not (ROOT / "scripts/train.py").is_file():
        missing.append("缺少本工程 train/play/export 训练与导出入口")
    return missing


def check_environment(require_gpu=False):
    expected = tomllib.loads((ROOT / "configs/training/versions.toml").read_text(encoding="utf-8"))
    report = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "project_root": str(ROOT),
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "packages": {},
        "errors": [],
        "warnings": [],
        "gpu": {},
        "simulation": {"tested": False},
        "project_blockers": project_blockers(),
    }
    if f"{sys.version_info.major}.{sys.version_info.minor}" != expected["python"]:
        report["errors"].append(f"该栈需要 Python {expected['python']}；请通过 scripts/training_python.sh 运行")
    packages = (
        ("isaaclab", "isaaclab", "isaaclab"),
        ("isaaclab_rl", "isaaclab-rl", "isaaclab_rl"),
        ("isaaclab_tasks", "isaaclab-tasks", "isaaclab_tasks"),
        ("rsl_rl", "rsl-rl-lib", "rsl_rl"),
        ("torch", "torch", "torch"),
        ("torchvision", "torchvision", "torchvision"),
        ("numpy", "numpy", "numpy"),
        ("gymnasium", "gymnasium", "gymnasium"),
        ("hydra", "hydra-core", "hydra"),
        ("tensorboard", "tensorboard", "tensorboard"),
        ("onnx", "onnx", "onnx"),
        ("warp", "warp-lang", "warp"),
        ("bpx_dreamwaq", "bpx-dreamwaq", "bpx_dreamwaq"),
    )
    for key, distribution, module in packages:
        item = report["packages"][key] = package(distribution, module)
        if item["path"] is None:
            report["errors"].append(f"无法找到 {module}")
        if key in expected and key != "isaaclab" and item["version"] != expected[key]:
            report["errors"].append(f"{key} 版本应为 {expected[key]}，实际为 {item['version']}")

    for key, variable in (("isaaclab", "ISAACLAB_PATH"), ("isaacsim", "ISAACSIM_PATH")):
        root = Path(os.environ.get(variable, "/nonexistent"))
        version_file = root / "VERSION"
        version = version_file.read_text().strip() if version_file.is_file() else None
        report[key] = {"root": str(root), "version": version}
        if version is None or version.split("-")[0] != expected[key]:
            report["errors"].append(f"{variable} 未指向预期的 {key} {expected[key]} 安装目录")
        elif key == "isaacsim" and "-rc." in version:
            report["warnings"].append(f"本机 Isaac Sim VERSION 包含候选构建标记：{version}")

    try:
        import torch

        report["gpu"].update(cuda_available=torch.cuda.is_available(), torch_cuda=torch.version.cuda)
        if torch.cuda.is_available():
            tensor = torch.ones((32, 32), device="cuda:0")
            assert (tensor @ tensor)[0, 0].item() == 32.0
            props = torch.cuda.get_device_properties(0)
            report["gpu"].update(name=props.name, vram_gib=round(props.total_memory / 1024**3, 2), matmul="passed")
        elif require_gpu:
            report["errors"].append("当前进程无法访问 CUDA；在有 GPU 访问权限的主机终端复查")
        else:
            report["warnings"].append("当前进程无法访问 CUDA，未验证 GPU 运行；沙箱可能未暴露 GPU")
    except Exception as exc:
        report["errors"].append(f"PyTorch 检查失败：{type(exc).__name__}: {exc}")

    try:
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab.app import AppLauncher

        report["imports"] = {"OnPolicyRunner": OnPolicyRunner.__module__, "AppLauncher": AppLauncher.__module__}
    except Exception as exc:
        report["errors"].append(f"训练组件导入失败：{type(exc).__name__}: {exc}")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        report["gpu"]["nvidia_smi"] = (result.stdout or result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["gpu"]["nvidia_smi"] = str(exc)
    report["environment_ready"] = not report["errors"]
    report["bpx_training_ready"] = report["environment_ready"] and not report["project_blockers"]
    return report


def smoke_test(on_complete):
    # Simulation modules must be imported only after AppLauncher creates the app.
    from isaaclab.app import AppLauncher

    import warp

    warp.init()
    if not warp.is_cuda_available():
        raise RuntimeError(f"Warp {warp.__version__} 无法访问 CUDA")
    faulthandler.dump_traceback_later(45)
    launcher = AppLauncher(headless=True, device="cuda:0", kit_args="--/log/level=warning")
    app = launcher.app
    try:
        print("[CHECK] Isaac Sim app started", flush=True)
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        import isaaclab.sim as sim_utils

        sim = SimulationContext(SimulationCfg(dt=1.0 / 60, device="cuda:0"))
        cube = sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        cube.func("/World/CheckCube", cube, translation=(0.0, 0.0, 1.0))
        sim.reset()
        for _ in range(10):
            sim.step(render=False)
        result = {"tested": True, "device": "cuda:0", "physics_steps": 10, "result": "passed",
                  "rsl_rl_adapter": RslRlVecEnvWrapper.__module__, "warp": warp.__version__}
        on_complete(result)
        return result
    finally:
        faulthandler.cancel_dump_traceback_later()
        # Isaac Lab's standalone STOP callback renders until playback resumes.
        # Unsubscribe it before app.close() stops the timeline (as its context manager does).
        from isaaclab.sim import SimulationContext

        SimulationContext.clear_instance()
        app.close(wait_for_replicator=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true", help="无 CUDA 访问时返回失败")
    parser.add_argument("--smoke-test", action="store_true", help="启动无界面 Isaac Sim 并执行十步 GPU 物理仿真")
    parser.add_argument("--report", type=Path, help="另存 JSON 报告；Kit 启动日志仍输出到终端")
    args = parser.parse_args()
    report = check_environment(require_gpu=args.require_gpu or args.smoke_test)
    # Preserve preflight evidence even if native simulator startup crashes.
    def save():
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save()
    if args.smoke_test and report["environment_ready"]:
        report["simulation"] = {"tested": False, "result": "running"}
        save()

        def on_complete(result):
            report["simulation"] = result
            save()
            print("[CHECK] Ten GPU physics steps passed", flush=True)
            # Kit may terminate Python during fast shutdown, so print before close().
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

        try:
            report["simulation"] = smoke_test(on_complete)
        except Exception as exc:
            report["errors"].append(f"Isaac Sim 运行检查失败：{type(exc).__name__}: {exc}")
            report["simulation"] = {"tested": False, "result": "failed"}
            report["environment_ready"] = False
        save()
    if not report["simulation"].get("tested") or report["errors"]:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["environment_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
