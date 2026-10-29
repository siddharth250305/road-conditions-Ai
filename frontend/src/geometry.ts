// Plain maths for the ruler and the study charts, kept apart from React so it can be tested.

/** Vertical centre of a ruler row, in percent of the ruler height. */
export function rulerCenter(index: number, count: number): number {
  return ((index + 0.5) / count) * 100;
}

/**
 * Nudge label positions apart so none are closer than minGap, keeping their order.
 * Used for the country names at the end of the slope chart lines.
 */
export function spreadLabels(positions: number[], minGap: number): number[] {
  const order = positions.map((p, i) => ({ p, i })).sort((a, b) => a.p - b.p);
  const placed: number[] = [];
  for (const { p } of order) {
    const last = placed[placed.length - 1];
    placed.push(last === undefined ? p : Math.max(p, last + minGap));
  }
  const out = new Array<number>(positions.length);
  order.forEach(({ i }, k) => {
    out[i] = placed[k]!;
  });
  return out;
}

/** An upper bound for a 0-based score axis: the data maximum plus headroom, in 0.1 steps. */
export function axisMax(values: number[]): number {
  const top = Math.max(0.1, ...values.filter((v) => Number.isFinite(v)));
  return Math.min(1, Math.ceil((top * 1.1) / 0.1) * 0.1);
}
