#!/usr/bin/env bash
# Run Python with this workspace and Isaac Sim's bundled dependencies available.

source "$(dirname -- "${BASH_SOURCE[0]}")/activate_training.sh" || exit $?
exec "$CONDA_PREFIX/bin/python" "$@"
