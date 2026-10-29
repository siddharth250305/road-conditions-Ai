import { axisMax, rulerCenter, spreadLabels } from "../geometry";

describe("rulerCenter", () => {
  it("centres each of six rows", () => {
    expect(rulerCenter(0, 6)).toBeCloseTo(8.33, 1);
    expect(rulerCenter(5, 6)).toBeCloseTo(91.67, 1);
  });
});

describe("spreadLabels", () => {
  it("leaves well-spaced labels alone", () => {
    expect(spreadLabels([10, 50, 90], 18)).toEqual([10, 50, 90]);
  });

  it("pushes crowded labels apart and keeps their order", () => {
    const out = spreadLabels([100, 102, 104], 18);
    expect(out).toEqual([100, 118, 136]);
  });

  it("returns positions in the original order for unsorted input", () => {
    const out = spreadLabels([104, 100, 102], 18);
    expect(out).toEqual([136, 100, 118]);
  });
});

describe("axisMax", () => {
  it("adds headroom and rounds up to a tenth", () => {
    expect(axisMax([0.8, 0.45])).toBeCloseTo(0.9);
  });
  it("never goes above 1", () => {
    expect(axisMax([0.99])).toBe(1);
  });
  it("copes with an empty or non-finite list", () => {
    expect(axisMax([])).toBeGreaterThan(0);
    expect(axisMax([Number.NaN])).toBeGreaterThan(0);
  });
});
