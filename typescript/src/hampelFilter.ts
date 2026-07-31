/**
 * Auditable causal and retrospective Hampel bad-tick diagnostics.
 *
 *     score = |value − median(window)| / (scale × MAD(window))
 *     flagged = score > threshold
 *
 * A Hampel filter is a **robust z-score in a rolling window**: it replaces the mean
 * with the median and the standard deviation with the Median Absolute Deviation.
 * That substitution is the point. A single bad tick pulls the mean toward itself and
 * inflates the standard deviation, so a classical z-score *hides* the very outlier
 * it is meant to catch — the failure mode known as **masking**. The median and MAD
 * have a 50% breakdown point: half the window can be garbage before either budges.
 *
 * The default `scale` of 1.4826 is not arbitrary. It makes `scale * MAD` a
 * consistent estimator of the standard deviation for normally distributed data, so a
 * `threshold` of 3 means roughly "three sigma" and carries the intuition people
 * already have — while remaining robust when the data isn't normal.
 *
 * Two properties decide whether this is safe to use in production.
 *
 * **Mode is a look-ahead decision, not a tuning knob.** `causal` mode uses
 * `max(0, i-k)..i` — only the past. `centered` mode uses
 * `max(0, i-k)..min(n-1, i+k)`, which includes **future** observations. Centered is
 * the better detector and is correct for cleaning a historical archive; using it to
 * build features for a backtest silently injects look-ahead bias and inflates
 * results. Every point reports `lookaheadUsed` so the choice is auditable rather
 * than implicit, and `audit.ts` makes the difference between the two modes explicit.
 *
 * **Detection is the default; repair is opt-in.** With `repair: "none"` (the
 * default) `output` is the original value and `suggestedReplacement` merely proposes
 * the window median. Nothing is overwritten unless a caller asks for
 * `repair: "median"`, because silently replacing market data destroys the evidence
 * that a feed is misbehaving.
 *
 * Two edge cases are modeled explicitly rather than papered over. A window whose MAD
 * is **zero** (every observation identical) makes the score undefined: a matching
 * value scores `0` (`zero_mad_match`) and any deviation scores infinity
 * (`zero_mad_deviation`). And a point with fewer than `minHistory` observations is
 * `insufficient_history` rather than flagged, because a two-point window has no
 * dispersion to speak of.
 *
 * Companion article (canonical):
 * https://thefintechbuilder.com/market-data-engineering/cleaning-and-validation/hampel-bad-tick-filter/
 * Catalog topic id: D01-F02-A02 (Domain D01 — Market Data Engineering /
 * Family D01-F02 — Cleaning and Validation)
 */

export type HampelMode = "causal" | "centered";
export type RepairPolicy = "none" | "median";

export type HampelOptions = {
  windowRadius?: number;
  threshold?: number;
  scale?: number;
  minHistory?: number;
  mode?: HampelMode;
  repair?: RepairPolicy;
};

export type HampelPoint = {
  index: number;
  value: number | null;
  mode: HampelMode;
  windowStart: number;
  windowEnd: number;
  windowCount: number;
  median: number | null;
  mad: number | null;
  scaledMad: number | null;
  score: number | null;
  threshold: number;
  flagged: boolean;
  status: string;
  lookaheadUsed: boolean;
  suggestedReplacement: number | null;
  output: number | null;
};

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function hampelFilter(
  values: Array<number | null>,
  options: HampelOptions = {},
): HampelPoint[] {
  const {
    windowRadius = 3,
    threshold = 3,
    scale = 1.4826,
    minHistory = 3,
    mode = "causal",
    repair = "none",
  } = options;

  if (!Number.isInteger(windowRadius) || windowRadius < 1) {
    throw new Error("windowRadius must be a positive integer");
  }
  if (!Number.isInteger(minHistory) || minHistory < 1) {
    throw new Error("minHistory must be a positive integer");
  }
  if (mode !== "causal" && mode !== "centered") {
    throw new Error("mode must be 'causal' or 'centered'");
  }
  if (repair !== "none" && repair !== "median") {
    throw new Error("repair must be 'none' or 'median'");
  }
  if (!Number.isFinite(threshold) || threshold <= 0) {
    throw new Error("threshold must be finite and positive");
  }
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error("scale must be finite and positive");
  }
  if (values.some((value) => value !== null && !Number.isFinite(value))) {
    throw new Error("values must be finite numbers or null");
  }

  return values.map((value, index) => {
    const windowStart = Math.max(0, index - windowRadius);
    const endExclusive = mode === "causal"
      ? index + 1
      : Math.min(values.length, index + windowRadius + 1);
    const window = values
      .slice(windowStart, endExclusive)
      .filter((candidate): candidate is number => candidate !== null);
    const lookaheadUsed = mode === "centered" && endExclusive > index + 1;

    let center: number | null = window.length ? median(window) : null;
    let mad: number | null = center === null
      ? null
      : median(window.map((candidate) => Math.abs(candidate - center!)));
    let scaledMad: number | null = mad === null ? null : scale * mad;
    let score: number | null = null;
    let flagged = false;
    let status = "eligible";

    if (value === null) {
      status = "missing";
      center = null;
      mad = null;
      scaledMad = null;
    } else if (window.length < minHistory) {
      status = "insufficient_history";
    } else {
      const deviation = Math.abs(value - center!);
      if (scaledMad === 0) {
        score = deviation === 0 ? 0 : Number.POSITIVE_INFINITY;
        status = deviation === 0 ? "zero_mad_match" : "zero_mad_deviation";
      } else {
        score = deviation / scaledMad!;
      }
      flagged = score > threshold;
    }

    const suggestedReplacement = flagged ? center : null;
    const output = repair === "median" && flagged ? suggestedReplacement : value;
    return {
      index,
      value,
      mode,
      windowStart,
      windowEnd: endExclusive - 1,
      windowCount: window.length,
      median: center,
      mad,
      scaledMad,
      score,
      threshold,
      flagged,
      status,
      lookaheadUsed,
      suggestedReplacement,
      output,
    };
  });
}
