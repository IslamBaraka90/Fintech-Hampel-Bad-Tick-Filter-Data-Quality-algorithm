/** Tests for the audit surface: mode comparison, look-ahead cost, calibration. */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  compareModes,
  flaggedIndexes,
  flaggedReport,
  lookaheadCost,
  thresholdSweep,
  type SweepOptions,
} from "../src/audit.ts";
import { hampelFilter, type HampelPoint } from "../src/hampelFilter.ts";
import { CANONICAL, ZERO_MAD, params, run } from "./fixtures.ts";

const close = (a: number, b: number, tol = 1e-6) => Math.abs(a - b) < tol;
const SWEEP_OPTIONS: SweepOptions = { windowRadius: 3, minHistory: 3, scale: 1.4826 };

// --- flagged report --------------------------------------------------------- //
test("the report lists only the flags", () => {
  assert.deepEqual(flaggedReport(run()).map((row) => row.index), [5, 12]);
});

test("the report carries the evidence for each flag", () => {
  const row = flaggedReport(run())[0]!;
  assert.equal(row.value, 112);
  assert.equal(row.windowStart, 2);
  assert.equal(row.windowEnd, 5);
  assert.equal(row.windowCount, 4);
  assert.ok(close(row.median!, 100.05));
  assert.ok(close(row.score!, CANONICAL.causal_checkpoint.score));
  assert.equal(row.status, "eligible");
  assert.equal(row.suggestedReplacement, row.median);
  assert.equal(row.lookaheadUsed, false);
});

test("a clean series reports nothing", () => {
  assert.deepEqual(flaggedReport(hampelFilter(Array(10).fill(100), params())), []);
});

test("the report rejects non-points", () => {
  assert.throws(() => flaggedReport([{ not: "a point" }] as unknown as HampelPoint[]), TypeError);
  assert.throws(() => flaggedReport("points" as unknown as HampelPoint[]), TypeError);
});

test("flaggedIndexes agrees with the report", () => {
  const points = run();
  assert.deepEqual(flaggedIndexes(points), flaggedReport(points).map((row) => row.index));
});

// --- mode comparison -------------------------------------------------------- //
test("the canonical series needs no look-ahead", () => {
  const comparison = compareModes(CANONICAL.values, params());
  assert.deepEqual(comparison.causalFlags, [5, 12]);
  assert.deepEqual(comparison.centeredFlags, [5, 12]);
  assert.deepEqual(comparison.agreed, [5, 12]);
  assert.deepEqual(comparison.lookaheadOnly, []);
  assert.deepEqual(comparison.causalOnly, []);
});

test("a causal-only flag is surfaced", () => {
  const comparison = compareModes(ZERO_MAD.values, params());
  assert.deepEqual(comparison.causalFlags, [4]);
  assert.deepEqual(comparison.centeredFlags, []);
  assert.deepEqual(comparison.causalOnly, [4]);
  assert.deepEqual(comparison.lookaheadOnly, []);
  assert.deepEqual(comparison.agreed, []);
});

test("a lookahead-only flag is surfaced", () => {
  const values = [100, 500, 100, 100, 100, 100, 100, 100];
  const comparison = compareModes(values, { windowRadius: 3, minHistory: 3, threshold: 3 });
  assert.ok(comparison.centeredFlags.includes(1));
  assert.ok(!comparison.causalFlags.includes(1));
  assert.ok(comparison.lookaheadOnly.includes(1));
});

test("the comparison counts where look-ahead was actually used", () => {
  const comparison = compareModes(CANONICAL.values, params());
  assert.equal(comparison.lookaheadPoints, CANONICAL.values.length - 1);
});

test("the flag lists are sorted and disjoint", () => {
  const comparison = compareModes(ZERO_MAD.values, params());
  for (const key of ["agreed", "lookaheadOnly", "causalOnly"] as const) {
    assert.deepEqual(comparison[key], [...comparison[key]].sort((a, b) => a - b));
  }
  const lookahead = new Set(comparison.lookaheadOnly);
  assert.ok(!comparison.causalOnly.some((index) => lookahead.has(index)));
  assert.ok(!comparison.agreed.some((index) => lookahead.has(index)));
});

test("comparing forbids passing mode", () => {
  assert.throws(
    () => compareModes(CANONICAL.values, { ...params(), mode: "causal" } as never),
    /sets mode itself/,
  );
});

test("the comparison uses identical parameters on both sides", () => {
  const comparison = compareModes(CANONICAL.values, params({ threshold: 1 }));
  assert.deepEqual(comparison.causalFlags, flaggedIndexes(run(null, { mode: "causal", threshold: 1 })));
  assert.deepEqual(
    comparison.centeredFlags,
    flaggedIndexes(run(null, { mode: "centered", threshold: 1 })),
  );
});

// --- look-ahead cost -------------------------------------------------------- //
test("perfect agreement means full realtime recall", () => {
  const cost = lookaheadCost(CANONICAL.values, params());
  assert.equal(cost.totalPoints, CANONICAL.values.length);
  assert.equal(cost.causalFlagCount, 2);
  assert.equal(cost.centeredFlagCount, 2);
  assert.equal(cost.agreedCount, 2);
  assert.equal(cost.realtimeRecall, 1);
  assert.equal(cost.lookaheadDependence, 0);
});

test("a lookahead-dependent series scores below one", () => {
  const values = [100, 500, 100, 100, 100, 100, 100, 100];
  const cost = lookaheadCost(values, { windowRadius: 3, minHistory: 3, threshold: 3 });
  assert.ok(cost.lookaheadOnlyCount >= 1);
  assert.ok(cost.realtimeRecall! < 1);
  assert.ok(cost.lookaheadDependence! > 0);
  assert.ok(close(cost.realtimeRecall! + cost.lookaheadDependence!, 1));
});

test("no centered flags leaves the ratios undefined", () => {
  const cost = lookaheadCost(ZERO_MAD.values, params());
  assert.equal(cost.centeredFlagCount, 0);
  assert.equal(cost.realtimeRecall, null);
  assert.equal(cost.lookaheadDependence, null);
  assert.equal(cost.causalOnlyCount, 1);
});

test("the cost counts agree with the comparison", () => {
  const values = [100, 500, 100, 100, 100, 100];
  const options = { windowRadius: 3, minHistory: 3, threshold: 3 };
  const cost = lookaheadCost(values, options);
  const comparison = compareModes(values, options);
  assert.equal(cost.agreedCount, comparison.agreed.length);
  assert.equal(cost.lookaheadOnlyCount, comparison.lookaheadOnly.length);
  assert.equal(cost.causalOnlyCount, comparison.causalOnly.length);
});

// --- threshold sweep -------------------------------------------------------- //
test("the sweep shows flags falling as the threshold rises", () => {
  const sweep = thresholdSweep(CANONICAL.values, [1, 3, 10, 1000], SWEEP_OPTIONS);
  const counts = sweep.map((point) => point.flaggedCount);
  assert.deepEqual(counts, [...counts].sort((a, b) => b - a));
  assert.ok(counts[0]! > counts.at(-1)!);
});

test("the sweep forbids passing threshold", () => {
  assert.throws(
    () => thresholdSweep(CANONICAL.values, [1, 3], { threshold: 5, windowRadius: 3 } as never),
    /sets threshold itself/,
  );
});

test("the sweep reports what each step newly forgives", () => {
  const sweep = thresholdSweep(CANONICAL.values, [1, 1000], SWEEP_OPTIONS);
  assert.equal(sweep[0]!.newlyForgiven, 0);
  assert.equal(sweep[1]!.newlyForgiven, sweep[0]!.flaggedCount - sweep[1]!.flaggedCount);
});

test("candidates are sorted", () => {
  assert.deepEqual(
    thresholdSweep(CANONICAL.values, [10, 1, 3], SWEEP_OPTIONS),
    thresholdSweep(CANONICAL.values, [1, 3, 10], SWEEP_OPTIONS),
  );
});

test("the flag rate is measured against eligible points only", () => {
  const point = thresholdSweep(CANONICAL.values, [3], SWEEP_OPTIONS)[0]!;
  // 17 values, 1 missing, 2 without enough history.
  assert.equal(point.eligibleCount, 14);
  assert.ok(point.eligibleCount < CANONICAL.values.length);
  assert.ok(close(point.flagRate!, point.flaggedCount / point.eligibleCount));
});

test("the sweep can run in either mode", () => {
  assert.equal(
    thresholdSweep(ZERO_MAD.values, [3], { ...SWEEP_OPTIONS, mode: "causal" })[0]!.flaggedCount,
    1,
  );
  assert.equal(
    thresholdSweep(ZERO_MAD.values, [3], { ...SWEEP_OPTIONS, mode: "centered" })[0]!.flaggedCount,
    0,
  );
});

test("a series with no eligible points has no flag rate", () => {
  const sweep = thresholdSweep([null, null], [3], { windowRadius: 3, minHistory: 3 });
  assert.equal(sweep[0]!.eligibleCount, 0);
  assert.equal(sweep[0]!.flagRate, null);
});

for (const [name, candidates] of Object.entries({
  empty: [],
  zero: [0],
  negative: [-1],
  nan: [NaN],
  inf: [Infinity],
  string: "3",
})) {
  test(`the sweep rejects bad candidates (${name})`, () => {
    assert.throws(() =>
      thresholdSweep(CANONICAL.values, candidates as number[], { windowRadius: 3, minHistory: 3 }),
    );
  });
}
