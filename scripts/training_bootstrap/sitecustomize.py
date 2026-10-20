"""Keep the installed Warp selected before Kit prepends its older extension.

Isaac Sim 5.0 bundles Warp 1.7.1, whose CUDA entry-point lookup is incompatible
with R580 drivers. This workspace supplies compatible Warp 1.8.1 in .training-deps;
the Conda environment's newer 1.16.0 removed APIs used by Isaac Sim 5.0.
This file is only on PYTHONPATH in this workspace's training launcher.
"""

from importlib.util import find_spec

if find_spec("warp") is not None:
    import warp  # noqa: F401
