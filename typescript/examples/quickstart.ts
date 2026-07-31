/**
 * Quickstart: masking, the look-ahead trap, and calibrating the threshold.
 *
 * Run:  node --experimental-strip-types examples/quickstart.ts
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import {
  compareModes,
  flaggedReport,
  lookaheadCost,
  thresholdSweep,
} from "../src/audit.ts";
import { hampelFilter, type HampelOptions } from "../src/hampelFilter.ts";

const FIXTURE = JSON.parse(
  readFileSync(fileURLToPath(new URL("../test/fixtures/fixtures.json", import.meta.url)), "utf8"),
);
const P = FIXTURE.parameters;
const OPTS: HampelOptions = {
  windowRadius: P.window_radius,
  threshold: P.threshold,
  scale: P.scale,
  minHistory: P.min_history,
};
const VALUES: (number | null)[] = FIXTURE.canonical.values;

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
const pstdev = (xs: number[]) => {
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length);
};
const pad = (value: unknown, width: number) => String(value).padStart(width);

// 1) Why median/MAD and not mean/std: masking.
// A rolling window of six: five ordinary ticks near 100, then one bad print.
const series = [99.9, 100.1, 100.0, 99.8, 100.2, 1000.0];
const z = Math.abs(series.at(-1)! - mean(series)) / pstdev(series);
const point = hampelFilter(series, { windowRadius: 5, minHistory: 3 }).at(-1)!;
console.log("one bad tick in a six-tick window:");
console.log(`  classical z-score : ${z.toFixed(2)}  -> BELOW 3, so a z-filter misses it entirely`);
console.log(`  Hampel score      : ${point.score!.toFixed(1)}  -> flagged=${point.flagged}`);
console.log(
  `  the outlier dragged the mean to ${mean(series).toFixed(1)} and inflated sigma to ` +
    `${pstdev(series).toFixed(1)} - that is the masking.`,
);
console.log(
  `  the median is still ${point.median!.toFixed(2)} and the MAD ${point.mad!.toFixed(2)}, ` +
    "so the outlier cannot hide behind its own effect.",
);

// 2) One point per input; detection by default.
const points = hampelFilter(VALUES, { ...OPTS, mode: "causal" });
console.log(`\n${VALUES.length} ticks in -> ${points.length} points out`);
for (const row of flaggedReport(points)) {
  console.log(
    `  index ${row.index}: value ${row.value} vs window median ${row.median!.toFixed(2)}  ` +
      `score ${row.score!.toFixed(1)}  suggests ${row.suggestedReplacement!.toFixed(2)}`,
  );
}
console.log(`  outputs unchanged: ${points.every((p) => p.output === p.value)}`);

// 3) Statuses that are not flags.
const missing = points[FIXTURE.canonical.missing_index]!;
console.log(`\nindex ${missing.index}: status '${missing.status}' - a gap, not an outlier`);
console.log(`index 0: status '${points[0]!.status}' - no dispersion to judge against yet`);

// 4) The look-ahead trap.
console.log("\nmode comparison on a series where they disagree:");
const zeroMad: (number | null)[] = FIXTURE.zero_mad.values;
const comparison = compareModes(zeroMad, OPTS);
console.log(`  values          [${zeroMad}]`);
console.log(`  causal flags    [${comparison.causalFlags}]`);
console.log(`  centered flags  [${comparison.centeredFlags}]`);
console.log(`  causalOnly      [${comparison.causalOnly}]  <- the future exonerated it`);

const spike = [100, 500, 100, 100, 100, 100, 100, 100];
const cost = lookaheadCost(spike, { windowRadius: 3, minHistory: 3, threshold: 3 });
console.log(`\nan early spike [${spike.slice(0, 3)}]...:`);
console.log(`  centered finds ${cost.centeredFlagCount}, causal finds ${cost.causalFlagCount}`);
console.log(`  realtime recall      ${(cost.realtimeRecall! * 100).toFixed(0)}%`);
console.log(
  `  lookahead dependence ${(cost.lookaheadDependence! * 100).toFixed(0)}%  ` +
    "<- flags a live system CANNOT make",
);

// 5) Calibrating the threshold on your own tape.
console.log("\nthreshold sweep (causal):");
console.log("  threshold  flagged  eligible  flag rate  newly forgiven");
for (const row of thresholdSweep(VALUES, [1, 2, 3, 10, 1000], {
  windowRadius: 3, minHistory: 3, scale: 1.4826,
})) {
  console.log(
    `  ${pad(row.threshold.toFixed(1), 9)}  ${pad(row.flaggedCount, 7)}  ` +
      `${pad(row.eligibleCount, 8)}  ${pad((row.flagRate! * 100).toFixed(1) + "%", 9)}  ` +
      `${pad(row.newlyForgiven, 14)}`,
  );
}
console.log("  -> the flag rate is over ELIGIBLE points; gaps and warm-up never count as clean");

// 6) Repair is opt-in.
const repaired = hampelFilter(VALUES, { ...OPTS, mode: "causal", repair: "median" });
const changed = repaired.filter((p) => p.output !== p.value).map((p) => p.index);
console.log(`\nrepair='median' rewrites only [${changed}] - everything else is untouched`);
