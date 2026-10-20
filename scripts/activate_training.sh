#!/usr/bin/env bash
# Source this file to select the existing Isaac Lab / Isaac Sim training stack.
# Optional overrides: BPX_CONDA_ROOT, BPX_TRAIN_ENV, BPX_ISAACLAB_ROOT, BPX_ISAACSIM_ROOT.

_bpx_activate_training() {
    local project_root conda_root env_name lab_root sim_root
    project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || return
    conda_root="${BPX_CONDA_ROOT:-$HOME/miniconda3}"
    env_name="${BPX_TRAIN_ENV:-isaaclab}"
    lab_root="${BPX_ISAACLAB_ROOT:-$HOME/IsaacLab-2.2.1}"
    sim_root="${BPX_ISAACSIM_ROOT:-$HOME/Isaac_Sim}"
    if [[ ! -f "$conda_root/etc/profile.d/conda.sh" || ! -f "$lab_root/VERSION" ||
          ! -f "$sim_root/setup_conda_env.sh" ]]; then
        echo "缺少 Conda、Isaac Lab 或 Isaac Sim；请设置 BPX_CONDA_ROOT / BPX_ISAACLAB_ROOT / BPX_ISAACSIM_ROOT。" >&2
        return 1
    fi
    # Conda uses base Python internally; never feed it Kit's Python 3.11 packages.
    export PYTHONPATH=
    source "$conda_root/etc/profile.d/conda.sh" || return
    conda activate "$env_name" || return
    # Build a clean path for this workspace, excluding inherited ROS/other-project paths.
    export PYTHONPATH="$project_root/scripts/training_bootstrap:$project_root/.training-deps:$project_root/src"
    source "$sim_root/setup_conda_env.sh" || return
    export ISAACLAB_PATH="$lab_root"
    export ISAACSIM_PATH="$sim_root"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONNOUSERSITE=1
    export PYTHONDONTWRITEBYTECODE=1
    if [[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.11" ]]; then
        echo "当前 Isaac Sim 5.0 栈需要 Python 3.11；请检查 BPX_TRAIN_ENV。" >&2
        return 1
    fi
}

_bpx_activate_training
_bpx_training_status=$?
unset -f _bpx_activate_training
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    exit "$_bpx_training_status"
fi
return "$_bpx_training_status"
