"""Fintech Hampel Bad-Tick Filter — robust rolling-window outlier detection.

    score = |value − median(window)| / (scale × MAD(window))

A robust z-score in a rolling window: median instead of mean, MAD instead of
standard deviation. That substitution is the point — one bad tick pulls the mean
toward itself and inflates the standard deviation, so a classical z-score *hides*
the outlier it should catch (**masking**). Median and MAD have a 50% breakdown point.

Two properties decide whether it is safe in production. **Mode is a look-ahead
decision:** `causal` sees only the past, `centered` reads `k` observations ahead —
better detection, and silent look-ahead bias if used for backtest features. Every
point reports `lookahead_used`, and the `audit` module makes the cost explicit.
**Detection is the default:** `repair="none"` leaves `output` as the original value
and merely *suggests* the window median.

Companion article (canonical): https://thefintechbuilder.com/market-data-engineering/cleaning-and-validation/hampel-bad-tick-filter/
Catalog topic id: D01-F02-A02  (Domain D01 — Market Data Engineering / Family D01-F02 — Cleaning and Validation)
"""

from __future__ import annotations

from .audit import (
    FlaggedPoint,
    LookaheadCost,
    ModeComparison,
    SweepPoint,
    compare_modes,
    flagged_indexes,
    flagged_report,
    lookahead_cost,
    threshold_sweep,
)
from .core import HampelPoint, Mode, Repair, hampel_filter

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "HampelPoint",
    "Mode",
    "Repair",
    "hampel_filter",
    "FlaggedPoint",
    "LookaheadCost",
    "ModeComparison",
    "SweepPoint",
    "compare_modes",
    "flagged_indexes",
    "flagged_report",
    "lookahead_cost",
    "threshold_sweep",
]
