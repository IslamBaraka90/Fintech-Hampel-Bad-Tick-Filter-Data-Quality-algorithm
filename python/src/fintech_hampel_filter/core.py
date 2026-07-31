"""Auditable causal and retrospective Hampel bad-tick diagnostics.

    score = |value - median(window)| / (scale * MAD(window))
    flagged = score > threshold

A Hampel filter is a **robust z-score in a rolling window**: it replaces the mean
with the median and the standard deviation with the Median Absolute Deviation. That
substitution is the point. A single bad tick pulls the mean toward itself and
inflates the standard deviation, so a classical z-score *hides* the very outlier it
is meant to catch — the failure mode known as **masking**. The median and MAD have a
50% breakdown point: half the window can be garbage before either budges.

The default ``scale`` of 1.4826 is not arbitrary. It makes ``scale * MAD`` a
consistent estimator of the standard deviation for normally distributed data, so a
``threshold`` of 3 means roughly "three sigma" and carries the intuition people
already have — while remaining robust when the data isn't normal.

Two properties decide whether this is safe to use in production.

**Mode is a look-ahead decision, not a tuning knob.** ``causal`` mode uses
``max(0, i-k)..i`` — only the past. ``centered`` mode uses
``max(0, i-k)..min(n-1, i+k)``, which includes **future** observations. Centered is
the better detector and is correct for cleaning a historical archive; using it to
build features for a backtest silently injects look-ahead bias and inflates results.
Every point reports ``lookahead_used`` so the choice is auditable rather than
implicit, and the ``audit`` module makes the difference between the two modes
explicit.

**Detection is the default; repair is opt-in.** With ``repair="none"`` (the
default) ``output`` is the original value and ``suggested_replacement`` merely
proposes the window median. Nothing is overwritten unless a caller asks for
``repair="median"``, because silently replacing market data destroys the evidence
that a feed is misbehaving.

Two edge cases are modeled explicitly rather than papered over. A window whose MAD
is **zero** (every observation identical) makes the score undefined: a matching
value scores ``0.0`` (``zero_mad_match``) and any deviation scores infinity
(``zero_mad_deviation``) — dividing by zero or silently passing would both be wrong.
And a point with fewer than ``min_history`` observations is ``insufficient_history``
rather than flagged, because a two-point window has no dispersion to speak of.

Companion article (canonical): https://thefintechbuilder.com/market-data-engineering/cleaning-and-validation/hampel-bad-tick-filter/
Catalog topic id: D01-F02-A02  (Domain D01 — Market Data Engineering / Family D01-F02 — Cleaning and Validation)
"""

from __future__ import annotations

from math import isfinite
from statistics import median
from typing import Iterable, Literal, TypedDict

Mode = Literal["causal", "centered"]
Repair = Literal["none", "median"]


class HampelPoint(TypedDict):
    index: int
    value: float | None
    mode: Mode
    window_start: int
    window_end: int
    window_count: int
    median: float | None
    mad: float | None
    scaled_mad: float | None
    score: float | None
    threshold: float
    flagged: bool
    status: str
    lookahead_used: bool
    suggested_replacement: float | None
    output: float | None


def _is_positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def hampel_filter(
    values: Iterable[float | None],
    *,
    window_radius: int = 3,
    threshold: float = 3.0,
    scale: float = 1.4826,
    min_history: int = 3,
    mode: Mode = "causal",
    repair: Repair = "none",
) -> list[HampelPoint]:
    """Flag local deviations while preserving the original observation by default.

    Causal mode uses indexes ``max(0, i-k)..i``. Centered mode uses
    ``max(0, i-k)..min(n-1, i+k)`` and is retrospective. Missing values are
    excluded from statistics and preserved. A non-matching value with zero MAD
    receives an infinite score; a matching value receives zero.
    """

    data = list(values)
    if not _is_positive_integer(window_radius):
        raise ValueError("window_radius must be a positive integer")
    if not _is_positive_integer(min_history):
        raise ValueError("min_history must be a positive integer")
    if mode not in ("causal", "centered"):
        raise ValueError("mode must be 'causal' or 'centered'")
    if repair not in ("none", "median"):
        raise ValueError("repair must be 'none' or 'median'")
    if isinstance(threshold, bool) or not isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    if isinstance(scale, bool) or not isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    for value in data:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value)
        ):
            raise ValueError("values must be finite numbers or None")

    output: list[HampelPoint] = []
    for index, raw_value in enumerate(data):
        start = max(0, index - window_radius)
        end_exclusive = index + 1 if mode == "causal" else min(len(data), index + window_radius + 1)
        window = [float(value) for value in data[start:end_exclusive] if value is not None]
        lookahead_used = mode == "centered" and end_exclusive > index + 1

        center: float | None = median(window) if window else None
        mad: float | None = (
            median(abs(value - center) for value in window) if center is not None else None
        )
        scaled_mad = scale * mad if mad is not None else None
        score: float | None = None
        flagged = False
        status = "eligible"

        if raw_value is None:
            status = "missing"
            center = mad = scaled_mad = None
        elif len(window) < min_history:
            status = "insufficient_history"
        else:
            deviation = abs(float(raw_value) - center)  # type: ignore[arg-type]
            if scaled_mad == 0:
                score = 0.0 if deviation == 0 else float("inf")
                status = "zero_mad_match" if deviation == 0 else "zero_mad_deviation"
            else:
                score = deviation / scaled_mad  # type: ignore[operator]
            flagged = score > threshold

        suggested = center if flagged else None
        original = float(raw_value) if raw_value is not None else None
        repaired = suggested if repair == "median" and flagged else original
        output.append(
            {
                "index": index,
                "value": original,
                "mode": mode,
                "window_start": start,
                "window_end": end_exclusive - 1,
                "window_count": len(window),
                "median": center,
                "mad": mad,
                "scaled_mad": scaled_mad,
                "score": score,
                "threshold": float(threshold),
                "flagged": flagged,
                "status": status,
                "lookahead_used": lookahead_used,
                "suggested_replacement": suggested,
                "output": repaired,
            }
        )
    return output
