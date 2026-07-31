/** Shared fixture access. The same JSON backs the Python suite. */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { hampelFilter, type HampelOptions, type HampelPoint } from "../src/hampelFilter.ts";

export const FIXTURE = JSON.parse(
  readFileSync(fileURLToPath(new URL("./fixtures/fixtures.json", import.meta.url)), "utf8"),
);
const RAW = FIXTURE.parameters as {
  window_radius: number;
  threshold: number;
  scale: number;
  min_history: number;
};
export const CANONICAL = FIXTURE.canonical;
export const ZERO_MAD = FIXTURE.zero_mad;

/** The fixture's parameters in the TypeScript naming, optionally overridden. */
export function params(overrides: Partial<HampelOptions> = {}): HampelOptions {
  return {
    windowRadius: RAW.window_radius,
    threshold: RAW.threshold,
    scale: RAW.scale,
    minHistory: RAW.min_history,
    ...overrides,
  };
}

export function run(
  values?: (number | null)[] | null,
  overrides: Partial<HampelOptions> = {},
): HampelPoint[] {
  return hampelFilter(values ?? CANONICAL.values, params(overrides));
}
