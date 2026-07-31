"""Tests for the audit surface: mode comparison, look-ahead cost, calibration."""

from __future__ import annotations

import pytest
from conftest import CANONICAL, ZERO_MAD, params

from fintech_hampel_filter import (
    compare_modes,
    flagged_indexes,
    flagged_report,
    hampel_filter,
    lookahead_cost,
    threshold_sweep,
)


def run(values=None, **overrides):
    return hampel_filter(CANONICAL["values"] if values is None else values, **params(**overrides))


# --------------------------------------------------------------------------- #
# Flagged report
# --------------------------------------------------------------------------- #
def test_the_report_lists_only_the_flags():
    report = flagged_report(run())
    assert [row["index"] for row in report] == [5, 12]


def test_the_report_carries_the_evidence_for_each_flag():
    """'Tick 5 was flagged' is not reviewable; the window and score are."""

    row = flagged_report(run())[0]
    assert row["value"] == 112.0
    assert row["window_start"] == 2 and row["window_end"] == 5
    assert row["window_count"] == 4
    assert row["median"] == pytest.approx(100.05)
    assert row["score"] == pytest.approx(CANONICAL["causal_checkpoint"]["score"])
    assert row["status"] == "eligible"
    assert row["suggested_replacement"] == row["median"]
    assert row["lookahead_used"] is False


def test_a_clean_series_reports_nothing():
    assert flagged_report(hampel_filter([100.0] * 10, **params())) == []


def test_the_report_rejects_non_points():
    with pytest.raises(TypeError):
        flagged_report([{"not": "a point"}])
    with pytest.raises(TypeError):
        flagged_report("points")


def test_flagged_indexes_agrees_with_the_report():
    points = run()
    assert flagged_indexes(points) == [row["index"] for row in flagged_report(points)]


# --------------------------------------------------------------------------- #
# Mode comparison
# --------------------------------------------------------------------------- #
def test_the_canonical_series_needs_no_lookahead():
    """Both modes find the same two bad ticks — the happy case."""

    comparison = compare_modes(CANONICAL["values"], **params())
    assert comparison["causal_flags"] == [5, 12]
    assert comparison["centered_flags"] == [5, 12]
    assert comparison["agreed"] == [5, 12]
    assert comparison["lookahead_only"] == []
    assert comparison["causal_only"] == []


def test_a_causal_only_flag_is_surfaced():
    """The zero-MAD series: causal flags a step the future explains away."""

    comparison = compare_modes(ZERO_MAD["values"], **params())
    assert comparison["causal_flags"] == [4]
    assert comparison["centered_flags"] == []
    assert comparison["causal_only"] == [4]
    assert comparison["lookahead_only"] == []
    assert comparison["agreed"] == []


def test_a_lookahead_only_flag_is_surfaced():
    """A spike in the first few ticks: only a forward-looking window can judge it."""

    values = [100.0, 500.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    comparison = compare_modes(values, window_radius=3, min_history=3, threshold=3.0)
    assert 1 in comparison["centered_flags"]
    assert 1 not in comparison["causal_flags"]     # index 1 has no history yet
    assert 1 in comparison["lookahead_only"]


def test_the_comparison_counts_where_lookahead_was_actually_used():
    """Only the final point has no future left to read."""

    comparison = compare_modes(CANONICAL["values"], **params())
    assert comparison["lookahead_points"] == len(CANONICAL["values"]) - 1


def test_the_flag_lists_are_sorted_and_disjoint():
    comparison = compare_modes(ZERO_MAD["values"], **params())
    for key in ("agreed", "lookahead_only", "causal_only"):
        assert comparison[key] == sorted(comparison[key])
    assert not set(comparison["lookahead_only"]) & set(comparison["causal_only"])
    assert not set(comparison["agreed"]) & set(comparison["lookahead_only"])


def test_comparing_forbids_passing_mode():
    """The whole point is that it sets both — accepting `mode` would be nonsense."""

    with pytest.raises(ValueError, match="sets mode itself"):
        compare_modes(CANONICAL["values"], mode="causal", **params())


def test_the_comparison_uses_identical_parameters_on_both_sides():
    comparison = compare_modes(CANONICAL["values"], **params(threshold=1.0))
    assert comparison["causal_flags"] == flagged_indexes(run(mode="causal", threshold=1.0))
    assert comparison["centered_flags"] == flagged_indexes(run(mode="centered", threshold=1.0))


# --------------------------------------------------------------------------- #
# Look-ahead cost
# --------------------------------------------------------------------------- #
def test_perfect_agreement_means_full_realtime_recall():
    cost = lookahead_cost(CANONICAL["values"], **params())
    assert cost["total_points"] == len(CANONICAL["values"])
    assert cost["causal_flag_count"] == cost["centered_flag_count"] == 2
    assert cost["agreed_count"] == 2
    assert cost["realtime_recall"] == 1.0
    assert cost["lookahead_dependence"] == 0.0


def test_a_lookahead_dependent_series_scores_below_one():
    values = [100.0, 500.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    cost = lookahead_cost(values, window_radius=3, min_history=3, threshold=3.0)
    assert cost["lookahead_only_count"] >= 1
    assert cost["realtime_recall"] < 1.0
    assert cost["lookahead_dependence"] > 0.0
    assert cost["realtime_recall"] + cost["lookahead_dependence"] == pytest.approx(1.0)


def test_no_centered_flags_leaves_the_ratios_undefined():
    """0/0 is undefined, not 1.0 — there is nothing to have recalled."""

    cost = lookahead_cost(ZERO_MAD["values"], **params())
    assert cost["centered_flag_count"] == 0
    assert cost["realtime_recall"] is None
    assert cost["lookahead_dependence"] is None
    assert cost["causal_only_count"] == 1


def test_the_cost_counts_agree_with_the_comparison():
    values = [100.0, 500.0, 100.0, 100.0, 100.0, 100.0]
    options = {"window_radius": 3, "min_history": 3, "threshold": 3.0}
    cost = lookahead_cost(values, **options)
    comparison = compare_modes(values, **options)
    assert cost["agreed_count"] == len(comparison["agreed"])
    assert cost["lookahead_only_count"] == len(comparison["lookahead_only"])
    assert cost["causal_only_count"] == len(comparison["causal_only"])


# --------------------------------------------------------------------------- #
# Threshold sweep
# --------------------------------------------------------------------------- #
def test_the_sweep_shows_flags_falling_as_the_threshold_rises():
    sweep = threshold_sweep(
        CANONICAL["values"], [1, 3, 10, 1000],
        window_radius=3, min_history=3, scale=1.4826,
    )
    counts = [point["flagged_count"] for point in sweep]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > counts[-1]


def test_the_sweep_forbids_passing_threshold():
    """It sets the threshold itself; accepting one would be nonsense."""

    with pytest.raises(ValueError, match="sets threshold itself"):
        threshold_sweep(CANONICAL["values"], [1, 3], threshold=5.0, window_radius=3)


def test_the_sweep_reports_what_each_step_newly_forgives():
    sweep = threshold_sweep(
        CANONICAL["values"], [1, 1000],
        window_radius=3, min_history=3, scale=1.4826,
    )
    assert sweep[0]["newly_forgiven"] == 0
    assert sweep[1]["newly_forgiven"] == sweep[0]["flagged_count"] - sweep[1]["flagged_count"]


def test_candidates_are_sorted():
    options = {"window_radius": 3, "min_history": 3, "scale": 1.4826}
    assert threshold_sweep(CANONICAL["values"], [10, 1, 3], **options) == threshold_sweep(
        CANONICAL["values"], [1, 3, 10], **options
    )


def test_the_flag_rate_is_measured_against_eligible_points_only():
    """A series full of gaps must not look clean just because it is empty."""

    options = {"window_radius": 3, "min_history": 3, "scale": 1.4826}
    sweep = threshold_sweep(CANONICAL["values"], [3.0], **options)
    point = sweep[0]
    # 17 values, 1 missing, 2 without enough history.
    assert point["eligible_count"] == 14
    assert point["eligible_count"] < len(CANONICAL["values"])
    assert point["flag_rate"] == pytest.approx(point["flagged_count"] / point["eligible_count"])


def test_the_sweep_can_run_in_either_mode():
    options = {"window_radius": 3, "min_history": 3, "scale": 1.4826}
    causal = threshold_sweep(ZERO_MAD["values"], [3.0], mode="causal", **options)
    centered = threshold_sweep(ZERO_MAD["values"], [3.0], mode="centered", **options)
    assert causal[0]["flagged_count"] == 1
    assert centered[0]["flagged_count"] == 0


def test_a_series_with_no_eligible_points_has_no_flag_rate():
    sweep = threshold_sweep([None, None], [3.0], window_radius=3, min_history=3)
    assert sweep[0]["eligible_count"] == 0
    assert sweep[0]["flag_rate"] is None


@pytest.mark.parametrize(
    "candidates", [[], [0], [-1], [float("nan")], [float("inf")], "3"],
    ids=["empty", "zero", "negative", "nan", "inf", "string"],
)
def test_the_sweep_rejects_bad_candidates(candidates):
    with pytest.raises((TypeError, ValueError)):
        threshold_sweep(CANONICAL["values"], candidates, window_radius=3, min_history=3)
