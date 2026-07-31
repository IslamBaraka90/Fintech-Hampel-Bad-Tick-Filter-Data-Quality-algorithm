/**
 * Contract tests for the Hampel bad-tick filter.
 *
 * The shared fixture is the cross-language acceptance anchor: a 17-tick series
 * around 100 with two injected bad ticks (112.0 at index 5, 86.0 at index 12) and a
 * gap at index 8, plus a zero-MAD series that separates the two modes.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { flaggedIndexes } from "../src/audit.ts";
import { hampelFilter } from "../src/hampelFilter.ts";
import { CANONICAL, ZERO_MAD, params, run } from "./fixtures.ts";

const close = (a: number, b: number, tol = 1e-6) => Math.abs(a - b) < tol;

// --- the shared fixture ----------------------------------------------------- //
for (const mode of ["causal", "centered"] as const) {
  test(`canonical flags match the fixture (${mode})`, () => {
    assert.deepEqual(flaggedIndexes(run(null, { mode })), CANONICAL[`expected_${mode}_flags`]);
    assert.deepEqual(flaggedIndexes(run(null, { mode })), [5, 12]);
  });

  test(`zero-MAD flags match the fixture (${mode})`, () => {
    assert.deepEqual(
      flaggedIndexes(run(ZERO_MAD.values, { mode })),
      ZERO_MAD[`expected_${mode}_flags`],
    );
  });
}

test("the causal checkpoint reproduces exactly", () => {
  const expected = CANONICAL.causal_checkpoint;
  const point = run(null, { mode: "causal" })[expected.index]!;
  assert.equal(point.windowStart, expected.window_start);
  assert.equal(point.windowEnd, expected.window_end);
  assert.equal(point.windowCount, expected.window_count);
  assert.ok(close(point.median!, expected.median));
  assert.ok(close(point.mad!, expected.mad));
  assert.ok(close(point.score!, expected.score));
  assert.equal(point.status, expected.status);
  assert.equal(point.flagged, expected.flagged);
});

test("one point per input value", () => {
  const points = run();
  assert.equal(points.length, CANONICAL.values.length);
  assert.deepEqual(points.map((p) => p.index), CANONICAL.values.map((_: unknown, i: number) => i));
});

test("an empty series produces no points", () => {
  assert.deepEqual(hampelFilter([], params()), []);
});

// --- robustness ------------------------------------------------------------- //
test("a single outlier does not mask itself", () => {
  const values = [...Array(10).fill(100), 1000];
  const point = hampelFilter(values, { windowRadius: 5, minHistory: 3, mode: "causal" }).at(-1)!;
  assert.equal(point.median, 100);
  assert.equal(point.flagged, true);

  // A mean/std z-score on the same window would score it well under 3.
  const window = values.slice(-6);
  const mean = window.reduce((a, b) => a + b, 0) / window.length;
  const variance = window.reduce((a, b) => a + (b - mean) ** 2, 0) / window.length;
  assert.ok(Math.abs(values.at(-1)! - mean) / Math.sqrt(variance) < 3);
});

test("the filter survives a window that is almost all garbage", () => {
  const point = hampelFilter([100, 100, 100, 500, 500], {
    windowRadius: 4, minHistory: 3, mode: "causal",
  }).at(-1)!;
  assert.equal(point.median, 100);
  assert.equal(point.flagged, true);
});

test("the default scale makes the threshold read as sigma", () => {
  const point = run()[5]!;
  assert.ok(close(point.scaledMad!, 1.4826 * point.mad!));
});

// --- mode ------------------------------------------------------------------- //
test("causal windows never reach forward", () => {
  for (const point of run(null, { mode: "causal" })) {
    assert.equal(point.windowEnd, point.index);
    assert.equal(point.lookaheadUsed, false);
  }
});

test("centered windows do reach forward", () => {
  const points = run(null, { mode: "centered" });
  assert.equal(points[0]!.windowEnd, params().windowRadius);
  assert.equal(points[0]!.lookaheadUsed, true);
  assert.equal(points.at(-1)!.lookaheadUsed, false);
  assert.equal(points.at(-1)!.windowEnd, CANONICAL.values.length - 1);
});

test("the two modes can disagree", () => {
  assert.deepEqual(flaggedIndexes(run(ZERO_MAD.values, { mode: "causal" })), [4]);
  assert.deepEqual(flaggedIndexes(run(ZERO_MAD.values, { mode: "centered" })), []);
});

test("window bounds are clamped at both ends", () => {
  const points = run(null, { mode: "centered" });
  assert.equal(points[0]!.windowStart, 0);
  assert.equal(points.at(-1)!.windowEnd, CANONICAL.values.length - 1);
});

// --- detection by default --------------------------------------------------- //
test("repair none leaves the original value", () => {
  for (const point of run(null, { repair: "none" })) assert.equal(point.output, point.value);
});

test("a flag only suggests a replacement", () => {
  const flagged = run(null, { repair: "none" }).filter((p) => p.flagged);
  assert.ok(flagged.length > 0);
  for (const point of flagged) {
    assert.equal(point.suggestedReplacement, point.median);
    assert.equal(point.output, point.value);
  }
});

test("repair median is opt-in and only touches flags", () => {
  for (const point of run(null, { repair: "median" })) {
    assert.equal(point.output, point.flagged ? point.median : point.value);
  }
});

test("unflagged points suggest nothing", () => {
  for (const point of run()) {
    if (!point.flagged) assert.equal(point.suggestedReplacement, null);
  }
});

test("input is never mutated", () => {
  const values = structuredClone(CANONICAL.values);
  const before = JSON.stringify(values);
  run(values);
  assert.equal(JSON.stringify(values), before);
});

// --- missing values and short history --------------------------------------- //
test("a missing value keeps its position and is never flagged", () => {
  const point = run()[CANONICAL.missing_index]!;
  assert.equal(point.status, "missing");
  assert.equal(point.value, null);
  assert.equal(point.flagged, false);
  assert.equal(point.score, null);
  assert.equal(point.median, null);
  assert.equal(point.output, null);
});

test("missing values are excluded from the statistics", () => {
  const point = hampelFilter([100, 100, null, 100, 100], {
    windowRadius: 4, minHistory: 3, mode: "causal",
  }).at(-1)!;
  assert.equal(point.windowCount, 4);
  assert.equal(point.median, 100);
});

test("early points lack history", () => {
  const points = run(null, { mode: "causal", minHistory: 3 });
  assert.equal(points[0]!.status, "insufficient_history");
  assert.equal(points[1]!.status, "insufficient_history");
  assert.equal(points[2]!.status, "eligible");
});

test("min history is measured in real values not slots", () => {
  const points = hampelFilter([null, null, 100, 101, 99], {
    windowRadius: 4, minHistory: 3, mode: "causal",
  });
  assert.equal(points[3]!.windowCount, 2);
  assert.equal(points[3]!.status, "insufficient_history");
  assert.equal(points[4]!.windowCount, 3);
  assert.equal(points[4]!.status, "eligible");
});

// --- the zero-MAD edge case ------------------------------------------------- //
test("a matching value in a zero-MAD window scores zero", () => {
  const last = hampelFilter([5, 5, 5, 5], { windowRadius: 3, minHistory: 3, mode: "causal" }).at(-1)!;
  assert.equal(last.mad, 0);
  assert.equal(last.status, "zero_mad_match");
  assert.equal(last.score, 0);
  assert.equal(last.flagged, false);
});

test("a deviating value in a zero-MAD window scores infinity", () => {
  const last = hampelFilter([5, 5, 5, 9], { windowRadius: 3, minHistory: 3, mode: "causal" }).at(-1)!;
  assert.equal(last.scaledMad, 0);
  assert.equal(last.status, "zero_mad_deviation");
  assert.equal(last.score, Infinity);
  assert.equal(last.flagged, true);
});

test("an infinite score is flagged at any finite threshold", () => {
  for (const threshold of [0.5, 3, 1e9]) {
    const last = hampelFilter([5, 5, 5, 9], {
      windowRadius: 3, minHistory: 3, threshold, mode: "causal",
    }).at(-1)!;
    assert.equal(last.flagged, true);
  }
});

// --- threshold semantics ---------------------------------------------------- //
test("the boundary is strict", () => {
  // window [0, 1, 2]: median 1, MAD 1, so |2 - 1| / (1 * 1) === exactly 1.
  const scored = hampelFilter([0, 1, 2], {
    windowRadius: 3, minHistory: 3, scale: 1, threshold: 1, mode: "causal",
  }).at(-1)!;
  assert.equal(scored.median, 1);
  assert.equal(scored.mad, 1);
  assert.ok(close(scored.score!, 1));
  assert.equal(scored.flagged, false); // `>` not `>=`

  const looser = hampelFilter([0, 1, 2], {
    windowRadius: 3, minHistory: 3, scale: 1, threshold: 0.999, mode: "causal",
  }).at(-1)!;
  assert.equal(looser.flagged, true);
});

test("a lower threshold flags more", () => {
  assert.ok(
    flaggedIndexes(run(null, { threshold: 1 })).length >=
      flaggedIndexes(run(null, { threshold: 50 })).length,
  );
});

test("every point echoes the threshold in force", () => {
  for (const point of run(null, { threshold: 7.5 })) assert.equal(point.threshold, 7.5);
});

// --- validation ------------------------------------------------------------- //
for (const [name, overrides] of Object.entries({
  "radius 0": { windowRadius: 0 },
  "radius negative": { windowRadius: -1 },
  "radius float": { windowRadius: 1.5 },
  "minHistory 0": { minHistory: 0 },
  "minHistory float": { minHistory: 2.5 },
  "threshold 0": { threshold: 0 },
  "threshold negative": { threshold: -1 },
  "threshold nan": { threshold: NaN },
  "threshold inf": { threshold: Infinity },
  "scale 0": { scale: 0 },
  "scale negative": { scale: -1 },
  "bad mode": { mode: "future" },
  "bad repair": { repair: "mean" },
})) {
  test(`invalid option raises (${name})`, () => {
    assert.throws(() => run(null, overrides as never));
  });
}

for (const bad of [NaN, Infinity, -Infinity, "100", true, []] as unknown[]) {
  test(`non-finite value raises (${String(bad)})`, () => {
    assert.throws(() => hampelFilter([100, 100, bad] as (number | null)[], params()));
  });
}

test("an all-missing series is all missing", () => {
  const points = hampelFilter([null, null, null], params());
  assert.ok(points.every((p) => p.status === "missing" && !p.flagged));
});
