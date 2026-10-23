#!/usr/bin/env python3
"""一键环境验证与安装：解释器、工作区资源、可选依赖和离线测试。

默认只做检查，不安装、不连接机器人、不启动 Isaac Sim：

    python scripts/setup_env.py                 # 当前解释器 + 工作区资源
    python scripts/setup_env.py --tests         # 额外运行当前解释器可跑的离线测试
    python scripts/setup_env.py --install       # 可编辑安装本工程（默认 extras=deploy）
    python scripts/setup_env.py --install --extras sim,inference
    python scripts/setup_env.py --train-check   # 另跑 Isaac Lab 训练栈检查
    python scripts/setup_env.py --all-envs      # 探测本机已知 Conda 环境
    python scripts/setup_env.py --json          # 机器可读输出

安装只处理本工程与 pip 可选依赖；不安装 Isaac Lab、Isaac Sim 或机器人 SDK，
也不修改已有 Conda 环境的包。训练栈检查通过 ``scripts/training_python.sh``
在现有 Isaac 环境中执行。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CONDA_ROOT = Path(os.environ.get("BPX_CONDA_ROOT", Path.home() / "miniconda3"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 已知环境名称与它们应具备的能力；只做探测，不自动切换或修改环境。
KNOWN_ENVS = {
    "bpx-sim2sim": ("MuJoCo 仿真", "ONNX / TorchScript 推理"),
    "bpx-real": ("官方 SDK 实机", "只读 Tk 界面"),
    "isaaclab": ("Isaac Lab 训练",),
}
CAPABILITY_MODULES = (
    "numpy", "mujoco", "onnxruntime", "tkinter", "bpx_sdk", "torch", "rsl_rl", "tensorboard",
)
DISTRIBUTIONS = {
    "numpy": "numpy",
    "mujoco": "mujoco",
    "onnxruntime": "onnxruntime",
    "tkinter": None,
    "bpx_sdk": "bpx-sdk",
    "torch": "torch",
    "rsl_rl": "rsl-rl-lib",
    "tensorboard": "tensorboard",
}
SUITES = {
    "project": (),
    "sim2sim": ("mujoco",),
    "sim2real": ("numpy",),
    "training": ("torch", "rsl_rl"),
}
# 在子进程中探测解释器时执行的代码；只读导入元数据，不导入重型模块。
PROBE = r"""
import importlib.metadata as md
import importlib.util as iu
import json, platform, sys

modules = {}
versions = {}
for name, distribution in json.loads(sys.argv[1]).items():
    try:
        modules[name] = iu.find_spec(name) is not None
    except (ImportError, ValueError):
        modules[name] = False
    if distribution:
        try:
            versions[name] = md.version(distribution)
        except md.PackageNotFoundError:
            versions[name] = None
print(json.dumps({
    "executable": sys.executable,
    "python": platform.python_version(),
    "modules": modules,
    "versions": versions,
}))
"""


def module_status(modules=CAPABILITY_MODULES, distributions=None):
    """返回当前解释器的模块可用性和版本；不导入模块本身。"""
    distributions = DISTRIBUTIONS if distributions is None else distributions
    available, versions = {}, {}
    for name in modules:
        try:
            available[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            available[name] = False
        distribution = distributions.get(name)
        if distribution and available[name]:
            try:
                versions[name] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
    return {"available": available, "versions": versions}


def workspace_report():
    """导入本工程的离线检查，避免依赖调用目录。"""
    try:
        from bpx_dreamwaq.validation import check_project
    except Exception as exc:  # 缺少 NumPy 等基础依赖时仍给出可读结论
        return {
            "project_root": str(ROOT),
            "errors": [f"无法导入工作区包：{type(exc).__name__}: {exc}"],
            "assets": {},
            "sim2sim_configs": [],
            "training_modules": [],
            "optional_dependencies": {},
            "pending": {},
        }
    try:
        return check_project()
    except Exception as exc:
        return {
            "project_root": str(ROOT),
            "errors": [f"工作区检查失败：{type(exc).__name__}: {exc}"],
            "assets": {},
            "sim2sim_configs": [],
            "training_modules": [],
            "optional_dependencies": {},
            "pending": {},
        }


def probe_interpreter(executable):
    """在指定解释器中探测能力；失败时返回错误说明。"""
    payload = json.dumps(DISTRIBUTIONS)
    try:
        result = subprocess.run(
            [str(executable), "-c", PROBE, payload],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"executable": str(executable), "error": str(exc)}
    if result.returncode != 0:
        return {"executable": str(executable), "error": (result.stderr or result.stdout).strip()}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {"executable": str(executable), "error": f"探测输出无法解析：{exc}"}


def known_interpreters():
    """列出已知 Conda 环境中实际存在的 Python 解释器。"""
    found = {}
    for name in KNOWN_ENVS:
        for relative in ("bin/python", "python.exe"):
            path = CONDA_ROOT / "envs" / name / relative
            if path.is_file():
                found[name] = path
                break
    return found


def validate_extras(extras):
    """只允许 pyproject 中声明的 extras，避免拼写错误导致静默安装基础包。"""
    with (ROOT / "pyproject.toml").open("rb") as stream:
        declared = tomllib.load(stream).get("project", {}).get("optional-dependencies", {})
    requested = [name for name in extras.split(",") if name]
    unknown = [name for name in requested if name not in declared]
    if unknown:
        raise ValueError(f"未知 extras：{', '.join(unknown)}；可选：{', '.join(sorted(declared))}")
    return requested


def install(extras):
    requested = validate_extras(extras)
    spec = str(ROOT) if not requested else f"{ROOT}[{','.join(requested)}]"
    command = [sys.executable, "-m", "pip", "install", "-e", spec]
    print(f"[install] {' '.join(command)}")
    return subprocess.run(command, cwd=ROOT).returncode


def run_suite(name, status):
    """运行离线测试套件；必需模块缺失时跳过并说明原因。"""
    missing = [module for module in SUITES[name] if not status["available"].get(module)]
    if missing:
        print(f"[tests] 跳过 tests/{name}：当前解释器缺少 {', '.join(missing)}")
        return None
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    # 关闭外部 pytest 插件自动加载不会影响 unittest；这里只清理可能污染的路径。
    env.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
    command = [sys.executable, "-m", "unittest", "discover", "-s", f"tests/{name}", "-v"]
    print(f"[tests] {name}: {' '.join(command)}")
    result = subprocess.run(command, cwd=ROOT, env=env)
    return result.returncode


def training_check():
    script = ROOT / "scripts" / "training_python.sh"
    check = ROOT / "scripts" / "check_training_env.py"
    if not script.is_file() or not check.is_file():
        print("[train] 未找到 scripts/training_python.sh 或 check_training_env.py，跳过")
        return None
    print(f"[train] bash {script.relative_to(ROOT)} {check.relative_to(ROOT)}")
    return subprocess.run(["bash", str(script), str(check)], cwd=ROOT).returncode


def print_interpreter(status, label="当前解释器"):
    modules = status["available"]
    versions = status["versions"]
    parts = []
    for name in CAPABILITY_MODULES:
        if modules.get(name):
            version = versions.get(name)
            parts.append(f"{name} {version}" if version else name)
        else:
            parts.append(f"{name} 缺失")
    print(f"[check] {label}: Python {status.get('python', platform.python_version())}")
    print(f"        {status.get('executable', sys.executable)}")
    print("        " + " | ".join(parts))


def print_report(report):
    print(f"[check] 工作区: {report.get('project_root', ROOT)}")
    assets = report.get("assets", {})
    configs = report.get("sim2sim_configs", [])
    print(f"[check] 机器人资源: {len(assets.get('missing_resources', []))} 个缺失；"
          f"关节契约: {assets.get('joint_contract', {})}")
    summaries = [f"{config['name']}({config['observation_dimension']}维)" for config in configs]
    print(f"[check] sim2sim 配置: {', '.join(summaries) or '无'}")
    print(f"[check] 训练模块: {', '.join(report.get('training_modules', [])) or '无'}")
    pending = report.get("pending", {})
    if pending.get("usd"):
        print(f"[check] 提示: {pending['usd']}")
    if pending.get("deploy_resources"):
        print(f"[check] 部署资源: {'；'.join(pending['deploy_resources'])}")
    if report.get("errors"):
        for error in report["errors"]:
            print(f"[错误] {error}")
    else:
        print("[check] 工作区资源与配置检查通过")


def print_envs(interpreters):
    print("[envs] 已知 Conda 环境：")
    if not interpreters:
        print("        未找到 BPX_CONDA_ROOT 下的 bpx-sim2sim / bpx-real / isaaclab")
        return
    for name, path in interpreters.items():
        status = probe_interpreter(path)
        if status.get("error"):
            print(f"        {name}: {status['error']}")
            continue
        modules = status["modules"]
        usable = [module for module in CAPABILITY_MODULES if modules.get(module)]
        print(f"        {name}: Python {status['python']} | {', '.join(usable) or '无可用模块'}")
    print("        说明：isaaclab 的 torch 由 Isaac Sim 的 setup_conda_env.sh 提供，"
          "需通过 scripts/training_python.sh 检查。")


def main(argv=None):
    parser = argparse.ArgumentParser(description="一键检查/安装 BPX DreamWaQ 运行与训练环境")
    parser.add_argument("--install", action="store_true", help="可编辑安装本工程到当前解释器")
    parser.add_argument("--extras", default="deploy", help="安装 extras，逗号分隔（默认 deploy）")
    parser.add_argument("--tests", action="store_true", help="运行当前解释器可执行的离线测试")
    parser.add_argument("--train-check", action="store_true", help="通过 training_python.sh 检查 Isaac Lab 训练栈")
    parser.add_argument("--all-envs", action="store_true", help="探测已知 Conda 环境的能力")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告")
    args = parser.parse_args(argv)
    # 子进程测试直接写终端；行缓冲保证检查输出与测试输出顺序一致。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    status = module_status()
    status.update(python=platform.python_version(), executable=sys.executable)
    report = workspace_report()

    failed = False
    if args.install:
        try:
            failed = install(args.extras) != 0 or failed
        except ValueError as exc:
            print(f"[install] {exc}", file=sys.stderr)
            failed = True

    if args.json:
        payload = {
            "interpreter": status,
            "workspace": report,
            "known_environments": {
                name: probe_interpreter(path) for name, path in known_interpreters().items()
            } if args.all_envs else {},
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print_interpreter(status)
        print_report(report)
        if args.all_envs:
            print_envs(known_interpreters())

    if args.tests:
        for name in SUITES:
            code = run_suite(name, status)
            if code not in (None, 0):
                failed = True

    if args.train_check:
        code = training_check()
        if code not in (None, 0):
            failed = True

    if report.get("errors"):
        failed = True
    if not status["available"].get("numpy"):
        print("[错误] 当前解释器缺少 NumPy；请先执行 pip install numpy 或使用已有环境", file=sys.stderr)
        failed = True

    if not args.json:
        print("[result] 检查%s" % ("失败" if failed else "通过"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
