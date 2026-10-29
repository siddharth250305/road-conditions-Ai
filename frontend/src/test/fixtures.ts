// Real responses captured from the running API (mock bundle), so these tests fail if the page's
// types ever drift from what the service actually sends.
import health from "./fixtures/health.json";
import predictSure from "./fixtures/predict_sure.json";
import predictUnsure from "./fixtures/predict_unsure.json";
import study from "./fixtures/study.json";
import type { Health, PredictResponse, Study } from "../types";

export const SURE = predictSure as unknown as PredictResponse;
export const UNSURE = predictUnsure as unknown as PredictResponse;
export const HEALTH = health as unknown as Health;
export const STUDY = study as unknown as Study;

type Route = (init?: RequestInit) => Response | Promise<Response>;

export function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

/** A fetch stand-in: routes are matched on the exact path; anything else is a network error. */
export function mockFetch(routes: Record<string, Route>) {
  const calls: { url: string; init?: RequestInit }[] = [];
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    calls.push({ url, init });
    const route = routes[url];
    if (!route) throw new TypeError("Failed to fetch");
    return route(init);
  });
  vi.stubGlobal("fetch", fn);
  return { fn, calls };
}

export const jpeg = () => new File([new Uint8Array([255, 216, 255, 217])], "road.jpg", { type: "image/jpeg" });
