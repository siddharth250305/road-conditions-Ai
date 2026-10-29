import type { Health, PredictResponse } from "./types";

export const MOCK_BANNER = "DEMO MODEL BUILT FROM MOCK DATA: its answers mean nothing.";

// The one-line version, shown under every state. The full text of a result comes from the API
// (limits_note); this is what the page says when there is no result yet.
export const LIMITS_LINE =
  "Trained only on RSCD, road-surface patches from vehicle cameras in China. Other countries, " +
  "night, rain on the lens and other surfaces have not been tested. Confidence is calibrated " +
  "on held-out RSCD data only. A student project, not a safety system.";

export const NOT_SEEN = [
  "Roads outside China, including Nordic roads",
  "Night, low sun, rain or dirt on the lens",
  "Anything that is not a road surface: the model has six classes and no 'not a road' class",
  "Surfaces and snow types outside the six friction classes",
];

export function isMock(health: Health | null, result: PredictResponse | null): boolean {
  return Boolean(result?.model.mock ?? health?.mock);
}
