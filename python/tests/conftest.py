"""Shared fixture access. The same JSON backs the TypeScript suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "fixtures.json").read_text(encoding="utf-8"))
PARAMS = FIXTURE["parameters"]
CANONICAL = FIXTURE["canonical"]
ZERO_MAD = FIXTURE["zero_mad"]


def params(**overrides) -> dict:
    return {**PARAMS, **overrides}
