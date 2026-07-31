"""Quickstart: masking, the look-ahead trap, and calibrating the threshold.

Run:  python examples/quickstart.py
"""

import json
import math
from pathlib import Path
from statistics import mean, pstdev

from fintech_hampel_filter import (
    compare_modes,
    flagged_report,
    hampel_filter,
    lookahead_cost,
    threshold_sweep,
)

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "fixtures.json").read_text(encoding="utf-8")
)
P = FIXTURE["parameters"]
OPTS = dict(
    window_radius=P["window_radius"], threshold=P["threshold"],
    scale=P["scale"], min_history=P["min_history"],
)
VALUES = FIXTURE["canonical"]["values"]

# 1) Why median/MAD and not mean/std: masking.
# A rolling window of six: five ordinary ticks near 100, then one bad print.
series = [99.9, 100.1, 100.0, 99.8, 100.2, 1000.0]
z = abs(series[-1] - mean(series)) / pstdev(series)
point = hampel_filter(series, window_radius=5, min_history=3)[-1]
print("one bad tick in a six-tick window:")
print(f"  classical z-score : {z:.2f}  -> BELOW 3, so a z-filter misses it entirely")
print(f"  Hampel score      : {point['score']:.1f}  -> flagged={point['flagged']}")
print(f"  the outlier dragged the mean to {mean(series):.1f} and inflated sigma to "
      f"{pstdev(series):.1f} — that is the masking.")
print(f"  the median is still {point['median']:.2f} and the MAD {point['mad']:.2f}, "
      f"so the outlier cannot hide behind its own effect.")

# 2) One point per input; detection by default.
points = hampel_filter(VALUES, mode="causal", **OPTS)
print(f"\n{len(VALUES)} ticks in -> {len(points)} points out")
for row in flagged_report(points):
    print(f"  index {row['index']}: value {row['value']} vs window median {row['median']:.2f}  "
          f"score {row['score']:.1f}  suggests {row['suggested_replacement']:.2f}")
print(f"  outputs unchanged: {all(p['output'] == p['value'] for p in points)}")

# 3) Statuses that are not flags.
missing = points[FIXTURE["canonical"]["missing_index"]]
print(f"\nindex {missing['index']}: status '{missing['status']}' — a gap, not an outlier")
print(f"index 0: status '{points[0]['status']}' — no dispersion to judge against yet")

# 4) The look-ahead trap.
print("\nmode comparison on a series where they disagree:")
zero_mad = FIXTURE["zero_mad"]["values"]
comparison = compare_modes(zero_mad, **OPTS)
print(f"  values          {zero_mad}")
print(f"  causal flags    {comparison['causal_flags']}")
print(f"  centered flags  {comparison['centered_flags']}")
print(f"  causal_only     {comparison['causal_only']}  <- the future exonerated it")

spike = [100.0, 500.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
cost = lookahead_cost(spike, window_radius=3, min_history=3, threshold=3.0)
print(f"\nan early spike {spike[:3]}...:")
print(f"  centered finds {cost['centered_flag_count']}, causal finds {cost['causal_flag_count']}")
print(f"  realtime recall     {cost['realtime_recall']:.0%}")
print(f"  lookahead dependence {cost['lookahead_dependence']:.0%}  <- flags a live system CANNOT make")

# 5) Calibrating the threshold on your own tape.
print("\nthreshold sweep (causal):")
print("  threshold  flagged  eligible  flag rate  newly forgiven")
for row in threshold_sweep(VALUES, [1, 2, 3, 10, 1000],
                           window_radius=3, min_history=3, scale=1.4826):
    print(f"  {row['threshold']:>9}  {row['flagged_count']:>7}  {row['eligible_count']:>8}  "
          f"{row['flag_rate']:>9.1%}  {row['newly_forgiven']:>14}")
print("  -> the flag rate is over ELIGIBLE points; gaps and warm-up never count as clean")

# 6) Repair is opt-in.
repaired = hampel_filter(VALUES, mode="causal", repair="median", **OPTS)
changed = [p["index"] for p in repaired if p["output"] != p["value"]]
print(f"\nrepair='median' rewrites only {changed} — everything else is untouched")
