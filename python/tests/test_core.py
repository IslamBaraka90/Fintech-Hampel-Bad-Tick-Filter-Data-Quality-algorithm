"""Contract tests for the Hampel bad-tick filter.

The shared fixture is the cross-language acceptance anchor: a 17-tick series around
100 with two injected bad ticks (112.0 at index 5, 86.0 at index 12) and a gap at
index 8, plus a zero-MAD series that separates the two modes.
"""

from __future__ import annotations

import copy
import math

import pytest
from conftest import CANONICAL, PARAMS, ZERO_MAD, params

from fintech_hampel_filter import flagged_indexes, hampel_filter

TOL = 1e-9


def run(values=None, **overrides):
    return hampel_filter(CANONICAL["values"] if values is None else values, **params(**overrides))


# --------------------------------------------------------------------------- #
# The shared fixture
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["causal", "centered"])
def test_canonical_flags_match_the_fixture(mode: str):
    assert flagged_indexes(run(mode=mode)) == CANONICAL[f"expected_{mode}_flags"] == [5, 12]


@pytest.mark.parametrize("mode", ["causal", "centered"])
def test_zero_mad_flags_match_the_fixture(mode: str):
    got = flagged_indexes(run(ZERO_MAD["values"], mode=mode))
    assert got == ZERO_MAD[f"expected_{mode}_flags"]


def test_the_causal_checkpoint_reproduces_exactly():
    """Every statistic that produced the flag, pinned."""

    expected = CANONICAL["causal_checkpoint"]
    point = run(mode="causal")[expected["index"]]
    assert point["window_start"] == expected["window_start"]
    assert point["window_end"] == expected["window_end"]
    assert point["window_count"] == expected["window_count"]
    assert point["median"] == pytest.approx(expected["median"])
    assert point["mad"] == pytest.approx(expected["mad"])
    assert point["score"] == pytest.approx(expected["score"])
    assert point["status"] == expected["status"]
    assert point["flagged"] is expected["flagged"]


def test_one_point_per_input_value():
    points = run()
    assert len(points) == len(CANONICAL["values"])
    assert [p["index"] for p in points] == list(range(len(CANONICAL["values"])))


def test_an_empty_series_produces_no_points():
    assert hampel_filter([], **params()) == []


# --------------------------------------------------------------------------- #
# Robustness — why median/MAD instead of mean/std
# --------------------------------------------------------------------------- #
def test_a_single_outlier_does_not_mask_itself():
    """The masking failure a classical z-score suffers from."""

    values = [100.0] * 10 + [1000.0]
    point = hampel_filter(values, window_radius=5, min_history=3, mode="causal")[-1]
    assert point["median"] == 100.0        # the outlier did not move the centre
    assert point["flagged"] is True

    # A mean/std z-score on the same window would score it well under 3.
    window = values[-6:]
    mean = sum(window) / len(window)
    variance = sum((v - mean) ** 2 for v in window) / len(window)
    z = abs(values[-1] - mean) / math.sqrt(variance)
    assert z < 3.0                          # ... and would therefore miss it


def test_the_filter_survives_a_window_that_is_almost_all_garbage():
    """Median and MAD have a 50% breakdown point."""

    values = [100.0, 100.0, 100.0, 500.0, 500.0]   # 2 of 5 corrupted
    point = hampel_filter(values, window_radius=4, min_history=3, mode="causal")[-1]
    assert point["median"] == 100.0
    assert point["flagged"] is True


def test_the_default_scale_makes_the_threshold_read_as_sigma():
    assert PARAMS["scale"] == 1.4826
    point = run()[5]
    assert point["scaled_mad"] == pytest.approx(1.4826 * point["mad"])


# --------------------------------------------------------------------------- #
# Mode: the look-ahead decision
# --------------------------------------------------------------------------- #
def test_causal_windows_never_reach_forward():
    for point in run(mode="causal"):
        assert point["window_end"] == point["index"]
        assert point["lookahead_used"] is False


def test_centered_windows_do_reach_forward():
    points = run(mode="centered")
    radius = PARAMS["window_radius"]
    assert points[0]["window_end"] == radius
    assert points[0]["lookahead_used"] is True
    # ... except at the very end, where there is no future left.
    assert points[-1]["lookahead_used"] is False
    assert points[-1]["window_end"] == len(CANONICAL["values"]) - 1


def test_the_two_modes_can_disagree():
    """The zero-MAD fixture is exactly such a case."""

    causal = flagged_indexes(run(ZERO_MAD["values"], mode="causal"))
    centered = flagged_indexes(run(ZERO_MAD["values"], mode="centered"))
    assert causal == [4]
    assert centered == []
    # Causal saw a step it could not yet know was permanent; centered saw the rest.


def test_window_bounds_are_clamped_at_both_ends():
    points = run(mode="centered")
    assert points[0]["window_start"] == 0
    assert points[-1]["window_end"] == len(CANONICAL["values"]) - 1


# --------------------------------------------------------------------------- #
# Detection by default
# --------------------------------------------------------------------------- #
def test_repair_none_leaves_the_original_value():
    for point in run(repair="none"):
        assert point["output"] == point["value"]


def test_a_flag_only_suggests_a_replacement():
    flagged = [p for p in run(repair="none") if p["flagged"]]
    assert flagged
    for point in flagged:
        assert point["suggested_replacement"] == point["median"]
        assert point["output"] == point["value"]    # unchanged


def test_repair_median_is_opt_in_and_only_touches_flags():
    points = run(repair="median")
    for point in points:
        if point["flagged"]:
            assert point["output"] == point["median"]
        else:
            assert point["output"] == point["value"]


def test_unflagged_points_suggest_nothing():
    for point in run():
        if not point["flagged"]:
            assert point["suggested_replacement"] is None


def test_input_is_never_mutated():
    values = copy.deepcopy(CANONICAL["values"])
    run(values)
    assert values == CANONICAL["values"]


# --------------------------------------------------------------------------- #
# Missing values and short history
# --------------------------------------------------------------------------- #
def test_a_missing_value_keeps_its_position_and_is_never_flagged():
    index = CANONICAL["missing_index"]
    point = run()[index]
    assert point["status"] == "missing"
    assert point["value"] is None
    assert point["flagged"] is False
    assert point["score"] is None
    assert point["median"] is None
    assert point["output"] is None


def test_missing_values_are_excluded_from_the_statistics():
    with_gap = [100.0, 100.0, None, 100.0, 100.0]
    point = hampel_filter(with_gap, window_radius=4, min_history=3, mode="causal")[-1]
    assert point["window_count"] == 4      # five slots, four real values
    assert point["median"] == 100.0


def test_early_points_lack_history():
    points = run(mode="causal", min_history=3)
    assert points[0]["status"] == "insufficient_history"
    assert points[1]["status"] == "insufficient_history"
    assert points[2]["status"] == "eligible"
    assert all(not p["flagged"] for p in points[:2])


def test_min_history_is_measured_in_real_values_not_slots():
    """Four slots with two gaps is two observations, not four."""

    values = [None, None, 100.0, 101.0, 99.0]
    points = hampel_filter(values, window_radius=4, min_history=3, mode="causal")
    assert points[3]["window_count"] == 2
    assert points[3]["status"] == "insufficient_history"   # only 2 real values so far
    assert points[4]["window_count"] == 3
    assert points[4]["status"] == "eligible"               # now 3


# --------------------------------------------------------------------------- #
# The zero-MAD edge case
# --------------------------------------------------------------------------- #
def test_a_matching_value_in_a_zero_mad_window_scores_zero():
    points = hampel_filter([5.0, 5.0, 5.0, 5.0], window_radius=3, min_history=3, mode="causal")
    last = points[-1]
    assert last["mad"] == 0.0
    assert last["status"] == "zero_mad_match"
    assert last["score"] == 0.0
    assert last["flagged"] is False


def test_a_deviating_value_in_a_zero_mad_window_scores_infinity():
    """Dividing by zero is wrong; so is silently passing it."""

    points = hampel_filter([5.0, 5.0, 5.0, 9.0], window_radius=3, min_history=3, mode="causal")
    last = points[-1]
    assert last["scaled_mad"] == 0.0
    assert last["status"] == "zero_mad_deviation"
    assert last["score"] == float("inf")
    assert last["flagged"] is True


def test_an_infinite_score_is_flagged_at_any_finite_threshold():
    for threshold in (0.5, 3.0, 1e9):
        points = hampel_filter(
            [5.0, 5.0, 5.0, 9.0], window_radius=3, min_history=3, threshold=threshold, mode="causal"
        )
        assert points[-1]["flagged"] is True


# --------------------------------------------------------------------------- #
# Threshold semantics
# --------------------------------------------------------------------------- #
def test_the_boundary_is_strict():
    """score > threshold, so a score exactly at the threshold is not flagged."""

    # window [0, 1, 2]: median 1, MAD 1, so |2 - 1| / (1 * 1) == exactly 1.0.
    points = hampel_filter([0.0, 1.0, 2.0], window_radius=3, min_history=3, scale=1.0, threshold=1.0, mode="causal")
    scored = points[-1]
    assert scored["median"] == 1.0
    assert scored["mad"] == 1.0
    assert scored["score"] == pytest.approx(1.0)
    assert scored["flagged"] is False           # `>` not `>=`

    # A hair under the threshold does flag it.
    looser = hampel_filter([0.0, 1.0, 2.0], window_radius=3, min_history=3, scale=1.0, threshold=0.999, mode="causal")
    assert looser[-1]["flagged"] is True


def test_a_lower_threshold_flags_more():
    strict = len(flagged_indexes(run(threshold=1.0)))
    loose = len(flagged_indexes(run(threshold=50.0)))
    assert strict >= loose


def test_every_point_echoes_the_threshold_in_force():
    for point in run(threshold=7.5):
        assert point["threshold"] == 7.5


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "overrides",
    [
        {"window_radius": 0}, {"window_radius": -1}, {"window_radius": 1.5}, {"window_radius": True},
        {"min_history": 0}, {"min_history": 2.5},
        {"threshold": 0}, {"threshold": -1}, {"threshold": float("nan")}, {"threshold": float("inf")},
        {"scale": 0}, {"scale": -1}, {"scale": float("inf")},
        {"mode": "future"}, {"repair": "mean"},
    ],
    ids=["r0", "r-neg", "r-float", "r-bool", "mh0", "mh-float",
         "t0", "t-neg", "t-nan", "t-inf", "s0", "s-neg", "s-inf", "bad-mode", "bad-repair"],
)
def test_invalid_options_raise(overrides: dict):
    with pytest.raises(ValueError):
        run(**overrides)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), "100", True, []])
def test_non_finite_values_raise(bad):
    with pytest.raises(ValueError, match="finite numbers or None"):
        hampel_filter([100.0, 100.0, bad], **params())


def test_an_all_missing_series_is_all_missing():
    points = hampel_filter([None, None, None], **params())
    assert all(p["status"] == "missing" for p in points)
    assert all(not p["flagged"] for p in points)
