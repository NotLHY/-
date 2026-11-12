"""Root adapter: the competition expects cost.py at the package root."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
runpy.run_path(str(HERE / "code" / "cost.py"), run_name="__main__")
