#!/usr/bin/env python3
"""从源码工作区启动实机配置检查、记录、回放或显式控制。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpx_dreamwaq.sim2real.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
