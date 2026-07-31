# Fintech Hampel Bad-Tick Filter — Data Quality Algorithm

> A canonical, well-specified, **cross-language (Python + TypeScript)** reference
> implementation of the **Hampel filter** — a robust z-score in a rolling window,
> using the median and MAD instead of the mean and standard deviation. That
> substitution is the point: a classical z-score **hides** the outlier it should
> catch, because the outlier inflates the very yardstick used to measure it. Ships
> an **audit** surface that makes the look-ahead cost of `centered` mode explicit and
> calibrates the threshold against your own tape.

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="TypeScript" src="https://img.shields.io/badge/typescript-5.7%2B-3178c6">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Tests" src="https://img.shields.io/badge/tests-80%20py%20%2F%2078%20ts-brightgreen">
</p>

**📖 Full article (canonical):** **[Hampel Bad-Tick Filter — The Fintech Builder](https://thefintechbuilder.com/market-data-engineering/cleaning-and-validation/hampel-bad-tick-filter/)**

This repository is the runnable, production-oriented companion to that article.
The article teaches the concept; this repo is the code you install and build on.

🧭 **Browse all algorithms:** [Awesome FinTech Algorithms](https://github.com/IslamBaraka90/Fintech-Algorithms-Awesome) — the full index of the library.
🗂️ **This algorithm's domain:** [Market Data Engineering](https://thefintechbuilder.com/domains/market-data-engineering/) › **Cleaning and Validation**
📥 **Just want to call it?** It also ships in the [`fintech-algorithms`](https://www.npmjs.com/package/fintech-algorithms) npm package — see [Two ways to use this](#two-ways-to-use-this).

| | |
|---|---|
| **Catalog topic** | `D01-F02-A02` |
| **Domain** | D01 — Market Data Engineering |
| **Family** | D01-F02 — Cleaning and Validation |
| **Difficulty** | 3 / 5 |
| **Languages** | Python, TypeScript |
| **Sibling** | [MAD Outlier Filter](https://github.com/IslamBaraka90/Fintech-MAD-Median-Absolute-Deviation-Outlier-Filter-algorithm) — the same statistic applied **globally** instead of in a window |

---

## Table of contents

- [What it computes](#what-it-computes)
- [Masking: why not a z-score](#masking-why-not-a-z-score)
- [Mode is a look-ahead decision](#mode-is-a-look-ahead-decision)
- [Detection by default](#detection-by-default)
- [Statuses](#statuses)
- [Two ways to use this](#two-ways-to-use-this)
- [Install](#install)
- [Quickstart](#quickstart)
- [Worked example (exact)](#worked-example-exact)
- [Audit: look-ahead cost and calibration](#audit-look-ahead-cost-and-calibration)
- [Options & point shape](#options--point-shape)
- [API reference](#api-reference)
- [Edge cases & limitations](#edge-cases--limitations)
- [Testing](#testing)
- [Related algorithms](#related-algorithms)
- [License](#license)

---

## What it computes

```
score   = |value − median(window)| / (scale × MAD(window))
flagged = score > threshold
```

Where `MAD(window) = median(|x − median(window)|)` — the Median Absolute Deviation.

The default `scale` of `1.4826` is not arbitrary: it makes `scale × MAD` a
consistent estimator of the standard deviation under normality, so a `threshold`
of 3 reads as "three sigma" and carries the intuition people already have — while
staying robust when the data isn't normal.

## Masking: why not a z-score

A classical z-score uses the mean and standard deviation. Both are **wrecked by the
very outlier they are supposed to find** — the outlier pulls the mean toward itself
and inflates the standard deviation, so the resulting score comes out small. The
failure has a name: **masking**.

Six ticks, five ordinary and one bad print of `1000`:

```
classical z-score : 2.24   <- BELOW 3, so a z-filter misses it entirely
Hampel score      : 4046.7 <- flagged

the outlier dragged the mean to 250.0 and inflated sigma to 335.4.
the median is still 100.05 and the MAD 0.15.
```

The median and MAD have a **50% breakdown point**: half the window can be garbage
before either moves. That is the entire reason this algorithm exists rather than
`abs(x - mean) / std`.

## Mode is a look-ahead decision

This is the part that quietly ruins backtests.

| mode | window | uses the future? |
|---|---|---|
| `causal` | `max(0, i−k) .. i` | no |
| `centered` | `max(0, i−k) .. min(n−1, i+k)` | **yes** |

`centered` is the better detector — it can see whether a spike reverted — and it is
the right choice for cleaning a historical archive. Use it to build features for a
backtest and you have injected look-ahead bias: the filter knew things the market
had not yet revealed, and your results are inflated.

So every point reports `lookahead_used`, and the
[`audit`](#audit-look-ahead-cost-and-calibration) module turns that into an answer
to the question a reviewer actually asks: *which of these flags would I still have
had, in real time?*

## Detection by default

`repair="none"` is the default. A flagged point sets `suggested_replacement` to the
window median and leaves `output` **equal to the original value**. Nothing is
overwritten unless a caller explicitly asks for `repair="median"`, because silently
replacing market data destroys the evidence that a feed is misbehaving.

```python
points = hampel_filter(ticks)                      # detection
points = hampel_filter(ticks, repair="median")     # opt-in repair, flags only
```

## Statuses

Not every non-flag is a clean tick, and the distinction matters:

| `status` | meaning |
|---|---|
| `eligible` | scored normally |
| `missing` | the value was `None` — a gap, never a flag, position preserved |
| `insufficient_history` | fewer than `min_history` real observations in the window |
| `zero_mad_match` | every value in the window is identical and this one matches → score `0.0` |
| `zero_mad_deviation` | every value identical and this one differs → score `inf`, flagged |

The zero-MAD pair is the edge case most implementations get wrong. Dividing by a
MAD of zero is an error, and quietly passing the point is worse — so a *matching*
value scores `0.0` and any *deviation* scores infinity, which is flagged at every
finite threshold.

`min_history` counts **real observations**, not window slots: four slots holding two
gaps is two observations.

## Two ways to use this

| | [`fintech-algorithms`](https://www.npmjs.com/package/fintech-algorithms) (npm) | this repo |
|---|---|---|
| Scope | the whole library, one install | this algorithm, in depth |
| Languages | TypeScript | Python **and** TypeScript |
| Surface | `hampelFilter()` | the filter **plus** the audit module |
| Best for | calling it in an app | reading, porting, extending, or auditing it |

```ts
// the fast path
import { hampelFilter } from "fintech-algorithms/market-data-engineering/cleaning-and-validation/hampel-bad-tick-filter";
```

## Install

**Python**

```bash
pip install fintech-hampel-filter
```

**TypeScript / JavaScript (Node ≥ 20)**

```bash
npm install fintech-hampel-filter
```

## Quickstart

**Python**

```python
from fintech_hampel_filter import flagged_report, hampel_filter

points = hampel_filter(
    ticks,                    # floats, or None for a gap
    window_radius=3,
    threshold=3.0,
    mode="causal",            # only the past — safe for live and for backtests
)

for row in flagged_report(points):
    quarantine(row["index"], row["value"], row["score"])
```

**TypeScript**

```ts
import { flaggedReport, hampelFilter } from "fintech-hampel-filter";

const points = hampelFilter(ticks, { windowRadius: 3, threshold: 3, mode: "causal" });
```

## Worked example (exact)

17 ticks around 100 with two injected bad prints and one gap, at
`window_radius=3, threshold=3.0, scale=1.4826, min_history=3`:

| index | value | what happens |
|---|--:|---|
| 0–1 | 100.0, 100.1 | `insufficient_history` |
| **5** | **112.0** | **flagged** — median 100.05, MAD 0.1, score **80.6016457575** |
| 8 | `null` | `missing` — position kept, never flagged |
| **12** | **86.0** | **flagged** — median 99.95, score 94.1 |

Both language suites assert the full flag list `[5, 12]` in **both** modes, plus
every statistic of the index-5 checkpoint.

## Audit: look-ahead cost and calibration

This repo's "beyond the tutorial" surface.

| helper | answers |
|---|---|
| `compare_modes` | where do causal and centered disagree, and why? |
| `lookahead_cost` | how much of centered's detection could a live system never reproduce? |
| `threshold_sweep` | what threshold should I actually run? |
| `flagged_report` | the flags, with the evidence that justifies each one |

**The look-ahead cost**, on a series with a spike at index 1:

```
centered finds 1, causal finds 0
realtime recall      0%
lookahead dependence 100%   <- flags a live system CANNOT make
```

`compare_modes` splits every flag three ways — `agreed`, `lookahead_only` (centered
only; exactly what the future bought you) and `causal_only` (causal flagged it and
later data exonerated it). Both lists are returned rather than scored, because
whether a `causal_only` flag is a false positive or a legitimately cautious
real-time call is a judgement, not arithmetic.

**Calibration:**

```
  threshold  flagged  eligible  flag rate  newly forgiven
        1.0        4        14      28.6%               0
        2.0        3        14      21.4%               1
        3.0        2        14      14.3%               1
     1000.0        0        14       0.0%               2
```

`flag_rate` is measured against **eligible** points only — gaps and warm-up never
count as clean, which they would if the denominator were the raw length.

Both helpers refuse to be handed the parameter they set themselves
(`compare_modes` rejects `mode`, `threshold_sweep` rejects `threshold`), because
accepting it would silently make the comparison meaningless.

## Options & point shape

**Options:** `window_radius` (default `3`), `threshold` (`3.0`), `scale` (`1.4826`),
`min_history` (`3`), `mode` (`"causal"`), `repair` (`"none"`). TypeScript uses the
camelCase equivalents.

**Point:** `index`, `value`, `mode`, `window_start`, `window_end`, `window_count`,
`median`, `mad`, `scaled_mad`, `score`, `threshold`, `flagged`, `status`,
`lookahead_used`, `suggested_replacement`, `output`.

Note the two languages differ in field casing — Python `snake_case`, TypeScript
`camelCase` — following each ecosystem's convention. Semantics and the fixture are
identical.

## Edge cases & limitations

- **The boundary is strict.** `score > threshold`, so a score of exactly the
  threshold is *not* flagged. The suites pin this with a window of `[0, 1, 2]`,
  where the score is exactly `1.0`.
- **An infinite score is flagged at any finite threshold** — including `1e9`.
- **Missing values keep their position** and are excluded from the statistics, so a
  window's `window_count` can be smaller than its span.
- **One point per input value**, always, in order.
- **Input is never mutated**, in either language.
- **`centered` mode is retrospective.** Never use it to build backtest features.
- **This finds *local* outliers.** A slow drift or a whole-day level shift will not
  be flagged — the window moves with it. For a cross-sectional view, use the
  [MAD filter](https://github.com/IslamBaraka90/Fintech-MAD-Median-Absolute-Deviation-Outlier-Filter-algorithm).
- **A flag is not proof of a bad tick.** Real markets gap.

## Testing

**Python** (80 tests)

```bash
cd python && pip install -e ".[dev]" && pytest
```

**TypeScript** (78 tests, zero runtime dependencies)

```bash
cd typescript && npm install && npm test && npm run build
```

Both suites walk the same `fixtures.json`, assert the flag lists in both modes and
every statistic of the causal checkpoint, and independently verify the masking
claim by computing a mean/std z-score on the same window and asserting it falls
below the threshold.

## Related algorithms

- `D01-F02-A03` — [MAD Outlier Filter](https://github.com/IslamBaraka90/Fintech-MAD-Median-Absolute-Deviation-Outlier-Filter-algorithm): the same median/MAD statistic applied to a whole batch instead of a rolling window. Hampel finds a tick that disagrees with its *neighbours*; MAD finds one that disagrees with its *peers*.
- `D01-F02-A01` — [OHLC Consistency Validator](https://github.com/IslamBaraka90/Fintech-OHLC-Consistency-Validator-Data-Quality-algorithm): structural validity, where this covers plausibility
- `D01-F04-A01` — [Missing-Bar Gap Classifier](https://github.com/IslamBaraka90/Fintech-Missing-Bar-Gap-Classifier-algorithm)
- `D01-F02-A04` — [Stale-Quote Detector](https://thefintechbuilder.com/market-data-engineering/cleaning-and-validation/stale-quote-detector/)

Full index: **[Awesome FinTech Algorithms](https://github.com/IslamBaraka90/Fintech-Algorithms-Awesome)**.

## License

[MIT](./LICENSE) © The Fintech Builder. Part of the
[100 FinTech Algorithms](https://thefintechbuilder.com) library.
