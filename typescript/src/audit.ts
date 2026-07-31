/**
 * Making the look-ahead choice visible, and calibrating the threshold.
 *
 * `mode: "centered"` is the better detector and it is also the one that will quietly
 * ruin a backtest. It reads `k` observations *ahead* of the point it is judging, so a
 * feature built with it knows things the market had not yet revealed. The filter
 * already records `lookaheadUsed` per point; this module turns that flag into an
 * answer to the question a reviewer actually asks:
 *
 *     "Which of these flags would I still have had, in real time?"
 *
 * `compareModes` runs both modes over the same series and classifies every flag as
 * **agreed**, **lookaheadOnly** (centered caught it, causal did not — the flags a
 * live system would have missed) or **causalOnly** (causal flagged it and the future
 * exonerated it). `lookaheadCost` condenses that into the numbers you would put in a
 * design note.
 *
 * The other operational question is where to put `threshold`. It has no universal
 * answer: 3 is conventional because `1.4826 * MAD` approximates a standard deviation
 * under normality, but a thin-book instrument at three sigma will flag half its
 * ticks. `thresholdSweep` re-runs the filter across candidates so the trade-off is
 * measured on your own tape rather than assumed.
 *
 * Nothing here repairs anything. The filter's default is detection, and these
 * helpers preserve it.
 */

import {
  hampelFilter,
  type HampelMode,
  type HampelOptions,
  type HampelPoint,
} from "./hampelFilter.ts";

export interface FlaggedPoint {
  index: number;
  value: number | null;
  score: number | null;
  median: number | null;
  scaledMad: number | null;
  windowStart: number;
  windowEnd: number;
  windowCount: number;
  status: string;
  suggestedReplacement: number | null;
  lookaheadUsed: boolean;
}

export interface ModeComparison {
  causalFlags: number[];
  centeredFlags: number[];
  /** Flagged by both — a live system would have caught these. */
  agreed: number[];
  /** Centered only. These are exactly the flags look-ahead bought you. */
  lookaheadOnly: number[];
  /** Causal only — the future exonerated them. */
  causalOnly: number[];
  /** Points where centered mode actually read ahead (all but the last). */
  lookaheadPoints: number;
}

export interface LookaheadCost {
  totalPoints: number;
  causalFlagCount: number;
  centeredFlagCount: number;
  agreedCount: number;
  lookaheadOnlyCount: number;
  causalOnlyCount: number;
  /**
   * agreed / centeredFlags — the share of centered's flags a live system would also
   * have raised. null when centered flagged nothing.
   */
  realtimeRecall: number | null;
  /** lookaheadOnly / centeredFlags — the share that depended on the future. */
  lookaheadDependence: number | null;
}

export interface SweepPoint {
  threshold: number;
  flaggedCount: number;
  eligibleCount: number;
  /** flagged / eligible, in [0, 1]. null when nothing was eligible. */
  flagRate: number | null;
  /** Points this threshold newly forgave versus the previous, stricter one. */
  newlyForgiven: number;
}

/** Options accepted by the comparison helpers — every filter option except `mode`. */
export type ComparisonOptions = Omit<HampelOptions, "mode">;
/** Options accepted by the sweep — every filter option except `threshold`. */
export type SweepOptions = Omit<HampelOptions, "threshold">;

function assertPoints(points: readonly HampelPoint[], label = "points"): HampelPoint[] {
  if (!Array.isArray(points)) {
    throw new TypeError(`${label} must be an array of Hampel points.`);
  }
  points.forEach((point, index) => {
    if (
      point === null ||
      typeof point !== "object" ||
      typeof point.flagged !== "boolean" ||
      typeof point.index !== "number"
    ) {
      throw new TypeError(`${label}[${index}] must be a point from hampelFilter().`);
    }
  });
  return [...points];
}

/** The indexes the filter flagged, in order. */
export function flaggedIndexes(points: readonly HampelPoint[]): number[] {
  return assertPoints(points).filter((point) => point.flagged).map((point) => point.index);
}

/**
 * The flagged points with everything needed to justify the decision.
 *
 * The window bounds and the statistics that produced the score travel with each row,
 * because "tick 412 was flagged" is not reviewable and "tick 412 sat 80 MADs from the
 * median of its four-tick window" is.
 */
export function flaggedReport(points: readonly HampelPoint[]): FlaggedPoint[] {
  return assertPoints(points)
    .filter((point) => point.flagged)
    .map((point) => ({
      index: point.index,
      value: point.value,
      score: point.score,
      median: point.median,
      scaledMad: point.scaledMad,
      windowStart: point.windowStart,
      windowEnd: point.windowEnd,
      windowCount: point.windowCount,
      status: point.status,
      suggestedReplacement: point.suggestedReplacement,
      lookaheadUsed: point.lookaheadUsed,
    }));
}

/**
 * Run both modes over one series and classify where they disagree.
 *
 * `lookaheadOnly` is the interesting list: those flags exist *only* because centered
 * mode could see the future. A live system running the same threshold would have
 * missed every one of them, so a backtest that used centered mode is reporting
 * detections it could not have made.
 *
 * `causalOnly` is the mirror — causal flagged a point that later observations showed
 * to be fine. Whether that is a false positive or a legitimately cautious real-time
 * call is a judgement, which is why both lists are returned rather than scored.
 */
export function compareModes(
  values: readonly (number | null)[],
  options: ComparisonOptions = {},
): ModeComparison {
  if ("mode" in options) {
    throw new Error("compareModes sets mode itself; pass the other options only.");
  }

  const causal = hampelFilter([...values], { ...options, mode: "causal" });
  const centered = hampelFilter([...values], { ...options, mode: "centered" });

  const causalFlags = flaggedIndexes(causal);
  const centeredFlags = flaggedIndexes(centered);
  const causalSet = new Set(causalFlags);
  const centeredSet = new Set(centeredFlags);
  const ascending = (a: number, b: number) => a - b;

  return {
    causalFlags,
    centeredFlags,
    agreed: causalFlags.filter((index) => centeredSet.has(index)).sort(ascending),
    lookaheadOnly: centeredFlags.filter((index) => !causalSet.has(index)).sort(ascending),
    causalOnly: causalFlags.filter((index) => !centeredSet.has(index)).sort(ascending),
    lookaheadPoints: centered.filter((point) => point.lookaheadUsed).length,
  };
}

/**
 * Quantify how much of centered mode's detection depended on the future.
 *
 * `realtimeRecall` answers "if I ship the causal filter, what fraction of the bad
 * ticks my archive-cleaning process finds will I still catch live?".
 * `lookaheadDependence` is its complement — the share of centered's flags that a
 * real-time system structurally cannot reproduce.
 *
 * A high dependence is not a bug; it is the honest cost of the mode, and the reason
 * it should never be chosen by accident.
 */
export function lookaheadCost(
  values: readonly (number | null)[],
  options: ComparisonOptions = {},
): LookaheadCost {
  const comparison = compareModes(values, options);
  const centeredCount = comparison.centeredFlags.length;
  return {
    totalPoints: values.length,
    causalFlagCount: comparison.causalFlags.length,
    centeredFlagCount: centeredCount,
    agreedCount: comparison.agreed.length,
    lookaheadOnlyCount: comparison.lookaheadOnly.length,
    causalOnlyCount: comparison.causalOnly.length,
    realtimeRecall: centeredCount ? comparison.agreed.length / centeredCount : null,
    lookaheadDependence: centeredCount ? comparison.lookaheadOnly.length / centeredCount : null,
  };
}

/**
 * Re-run the filter across candidate thresholds to calibrate sensitivity.
 *
 * `flagRate` is measured against **eligible** points only — those with enough
 * history and a real value. Dividing by the raw length would let a series full of
 * gaps look reassuringly clean.
 *
 * Candidates are sorted ascending so `newlyForgiven` always compares against the
 * next-strictest threshold regardless of the order supplied.
 */
export function thresholdSweep(
  values: readonly (number | null)[],
  thresholdCandidates: readonly number[],
  options: SweepOptions = {},
): SweepPoint[] {
  if ("threshold" in options) {
    throw new Error("thresholdSweep sets threshold itself; pass the candidates only.");
  }
  if (!Array.isArray(thresholdCandidates)) {
    throw new TypeError("thresholdCandidates must be an array of numbers.");
  }
  if (thresholdCandidates.length === 0) {
    throw new Error("thresholdCandidates must not be empty.");
  }
  for (const candidate of thresholdCandidates) {
    if (typeof candidate !== "number" || !Number.isFinite(candidate) || candidate <= 0) {
      throw new Error("every threshold candidate must be finite and positive.");
    }
  }

  const sweep: SweepPoint[] = [];
  let previousFlagged: number | null = null;
  for (const candidate of [...thresholdCandidates].sort((a, b) => a - b)) {
    const points = hampelFilter([...values], { ...options, threshold: candidate });
    const flagged = points.filter((point) => point.flagged).length;
    const eligible = points.filter(
      (point) => point.status !== "missing" && point.status !== "insufficient_history",
    ).length;
    sweep.push({
      threshold: candidate,
      flaggedCount: flagged,
      eligibleCount: eligible,
      flagRate: eligible ? flagged / eligible : null,
      newlyForgiven: previousFlagged === null ? 0 : previousFlagged - flagged,
    });
    previousFlagged = flagged;
  }
  return sweep;
}

export type { HampelMode };
