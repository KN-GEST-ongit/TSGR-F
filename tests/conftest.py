from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
