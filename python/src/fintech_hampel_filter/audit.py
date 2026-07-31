"""Making the look-ahead choice visible, and calibrating the threshold.

``mode="centered"`` is the better detector and it is also the one that will quietly
ruin a backtest. It reads ``k`` observations *ahead* of the point it is judging, so
a feature built with it knows things the market had not yet revealed. The filter
already records ``lookahead_used`` per point; this module turns that flag into an
answer to the question a reviewer actually asks:

    "Which of these flags would I still have had, in real time?"

:func:`compare_modes` runs both modes over the same series and classifies every
flag as **agreed**, **lookahead_only** (centered caught it, causal did not — the
flags a live system would have missed) or **causal_only** (causal flagged it and the
future exonerated it — a false positive centered mode avoids). :func:`lookahead_cost`
condenses that into the numbers you would put in a design note.

The other operational question is where to put ``threshold``. It has no universal
answer: 3 is conventional because ``1.4826 * MAD`` approximates a standard deviation
under normality, but a thin-book instrument at three sigma will flag half its ticks.
:func:`threshold_sweep` re-runs the filter across candidates so the trade-off is
measured on your own tape rather than assumed.

:func:`flagged_report` is the small convenience the other three are built on: the
flagged points, with their score, window and suggested replacement, ready to file.

Nothing here repairs anything. The filter's default is detection, and these helpers
preserve it.
"""

from __future__ import annotations

from math import isfinite
from typing import Any, Iterable, Literal, Mapping, Sequence, TypedDict

from .core import Mode, hampel_filter


Agreement = Literal["agreed", "lookahead_only", "causal_only"]


class FlaggedPoint(TypedDict):
    index: int
    value: float | None
    score: float | None
    median: float | None
    scaled_mad: float | None
    window_start: int
    window_end: int
    window_count: int
    status: str
    suggested_replacement: float | None
    lookahead_used: bool


class ModeComparison(TypedDict):
    causal_flags: list[int]
    centered_flags: list[int]
    #: Flagged by both — a live system would have caught these.
    agreed: list[int]
    #: Centered only. These are exactly the flags look-ahead bought you.
    lookahead_only: list[int]
    #: Causal only — the future exonerated them.
    causal_only: list[int]
    #: Points where centered mode actually read ahead (all but the tail).
    lookahead_points: int


class LookaheadCost(TypedDict):
    total_points: int
    causal_flag_count: int
    centered_flag_count: int
    agreed_count: int
    lookahead_only_count: int
    causal_only_count: int
    #: agreed / centered_flags — the share of centered's flags a live system
    #: would also have raised. None when centered flagged nothing.
    realtime_recall: float | None
    #: lookahead_only / centered_flags — the share that depended on the future.
    lookahead_dependence: float | None


class SweepPoint(TypedDict):
    threshold: float
    flagged_count: int
    eligible_count: int
    #: flagged / eligible, in [0, 1]. None when nothing was eligible.
    flag_rate: float | None
    #: Points this threshold newly forgave versus the previous, stricter one.
    newly_forgiven: int


def _points(points: Iterable[Any], label: str = "points") -> list[Mapping[str, Any]]:
    if isinstance(points, (str, bytes)) or not isinstance(points, Iterable):
        raise TypeError(f"{label} must be an iterable of Hampel points.")
    items = list(points)
    for index, point in enumerate(items):
        if not isinstance(point, Mapping) or "flagged" not in point or "index" not in point:
            raise TypeError(f"{label}[{index}] must be a point from hampel_filter().")
    return items


def flagged_indexes(points: Iterable[Mapping[str, Any]]) -> list[int]:
    """The indexes the filter flagged, in order."""

    return [point["index"] for point in _points(points) if point["flagged"]]


def flagged_report(points: Iterable[Mapping[str, Any]]) -> list[FlaggedPoint]:
    """The flagged points with everything needed to justify the decision.

    The window bounds and the statistics that produced the score travel with each
    row, because "tick 412 was flagged" is not reviewable and "tick 412 sat 80 MADs
    from the median of its four-tick window" is.
    """

    return [
        {
            "index": point["index"],
            "value": point["value"],
            "score": point["score"],
            "median": point["median"],
            "scaled_mad": point["scaled_mad"],
            "window_start": point["window_start"],
            "window_end": point["window_end"],
            "window_count": point["window_count"],
            "status": point["status"],
            "suggested_replacement": point["suggested_replacement"],
            "lookahead_used": point["lookahead_used"],
        }
        for point in _points(points)
        if point["flagged"]
    ]


def compare_modes(values: Sequence[float | None], **options: Any) -> ModeComparison:
    """Run both modes over one series and classify where they disagree.

    ``lookahead_only`` is the interesting list: those flags exist *only* because
    centered mode could see the future. A live system running the same threshold
    would have missed every one of them, so a backtest that used centered mode is
    reporting detections it could not have made.

    ``causal_only`` is the mirror — causal flagged a point that later observations
    showed to be fine. Whether that is a false positive or a legitimately cautious
    real-time call is a judgement, which is why both lists are returned rather than
    scored.

    ``options`` are forwarded to :func:`~.core.hampel_filter` unchanged, minus
    ``mode``, so the comparison always uses identical parameters on both sides.
    """

    if "mode" in options:
        raise ValueError("compare_modes sets mode itself; pass the other options only.")

    causal = hampel_filter(values, mode="causal", **options)
    centered = hampel_filter(values, mode="centered", **options)

    causal_flags = flagged_indexes(causal)
    centered_flags = flagged_indexes(centered)
    causal_set, centered_set = set(causal_flags), set(centered_flags)

    return {
        "causal_flags": causal_flags,
        "centered_flags": centered_flags,
        "agreed": sorted(causal_set & centered_set),
        "lookahead_only": sorted(centered_set - causal_set),
        "causal_only": sorted(causal_set - centered_set),
        "lookahead_points": sum(1 for point in centered if point["lookahead_used"]),
    }


def lookahead_cost(values: Sequence[float | None], **options: Any) -> LookaheadCost:
    """Quantify how much of centered mode's detection depended on the future.

    ``realtime_recall`` answers "if I ship the causal filter, what fraction of the
    bad ticks my archive-cleaning process finds will I still catch live?".
    ``lookahead_dependence`` is its complement — the share of centered's flags that
    a real-time system structurally cannot reproduce.

    A high dependence is not a bug; it is the honest cost of the mode, and the
    reason it should never be chosen by accident.
    """

    comparison = compare_modes(values, **options)
    centered_count = len(comparison["centered_flags"])
    return {
        "total_points": len(list(values)),
        "causal_flag_count": len(comparison["causal_flags"]),
        "centered_flag_count": centered_count,
        "agreed_count": len(comparison["agreed"]),
        "lookahead_only_count": len(comparison["lookahead_only"]),
        "causal_only_count": len(comparison["causal_only"]),
        "realtime_recall": len(comparison["agreed"]) / centered_count if centered_count else None,
        "lookahead_dependence": (
            len(comparison["lookahead_only"]) / centered_count if centered_count else None
        ),
    }


def threshold_sweep(
    values: Sequence[float | None],
    threshold_candidates: Sequence[float],
    *,
    mode: Mode = "causal",
    **options: Any,
) -> list[SweepPoint]:
    """Re-run the filter across candidate thresholds to calibrate sensitivity.

    ``flag_rate`` is measured against **eligible** points only — those with enough
    history and a real value. Dividing by the raw length would let a series full of
    gaps look reassuringly clean.

    Candidates are sorted ascending so ``newly_forgiven`` always compares against the
    next-strictest threshold regardless of the order supplied. Read the sweep by
    looking for where the flag count stops falling steeply: below that, you are
    catching genuine outliers; above it, you are mostly raising the bar past them.
    """

    if "threshold" in options:
        raise ValueError("threshold_sweep sets threshold itself; pass the candidates only.")
    if isinstance(threshold_candidates, (str, bytes)) or not isinstance(threshold_candidates, Sequence):
        raise TypeError("threshold_candidates must be a sequence of numbers.")
    if not threshold_candidates:
        raise ValueError("threshold_candidates must not be empty.")
    for candidate in threshold_candidates:
        if isinstance(candidate, bool) or not isinstance(candidate, (int, float)) or not isfinite(candidate) or candidate <= 0:
            raise ValueError("every threshold candidate must be finite and positive.")

    sweep: list[SweepPoint] = []
    previous_flagged: int | None = None
    for candidate in sorted(float(value) for value in threshold_candidates):
        points = hampel_filter(values, mode=mode, threshold=candidate, **options)
        flagged = sum(1 for point in points if point["flagged"])
        eligible = sum(1 for point in points if point["status"] != "missing" and point["status"] != "insufficient_history")
        sweep.append(
            {
                "threshold": candidate,
                "flagged_count": flagged,
                "eligible_count": eligible,
                "flag_rate": flagged / eligible if eligible else None,
                "newly_forgiven": 0 if previous_flagged is None else previous_flagged - flagged,
            }
        )
        previous_flagged = flagged
    return sweep


__all__ = [
    "Agreement",
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
