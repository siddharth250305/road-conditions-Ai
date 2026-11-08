import type { Health, PredictResponse, ReviewItem, ReviewQueue, Study } from "./types";

export type ErrorKind = "not_image" | "too_large" | "no_model" | "offline" | "server";

// Fixed wording: server details are never shown to the person using the page.
export const ERROR_MESSAGES: Record<ErrorKind, string> = {
  not_image: "That file could not be read as a picture. Try a JPEG, PNG or WebP road photo.",
  too_large: "That picture is too large. Try one under 10 MB.",
  no_model: "The service is running but has no model loaded, so it cannot read photos yet.",
  offline: "Cannot reach the service. Check your connection, then try again.",
  server: "The service had a problem reading that photo. Try again in a moment.",
};

export class ApiError extends Error {
  kind: ErrorKind;
  status?: number;
  constructor(kind: ErrorKind, status?: number) {
    super(ERROR_MESSAGES[kind]);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
  }
}

export function errorFromStatus(status: number): ApiError {
  if (status === 400 || status === 415 || status === 422) return new ApiError("not_image", status);
  if (status === 413) return new ApiError("too_large", status);
  if (status === 503) return new ApiError("no_model", status);
  return new ApiError("server", status);
}

type Fetch = typeof fetch;

async function request(fetchImpl: Fetch, input: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetchImpl(input, init);
  } catch {
    throw new ApiError("offline");
  }
}

export async function predict(
  file: Blob,
  name = "photo",
  fetchImpl: Fetch = fetch,
): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", file, name);
  const response = await request(fetchImpl, "/predict", { method: "POST", body: form });
  if (!response.ok) throw errorFromStatus(response.status);
  return (await response.json()) as PredictResponse;
}

export async function fetchHealth(fetchImpl: Fetch = fetch): Promise<Health> {
  const response = await request(fetchImpl, "/health");
  if (response.status === 503) {
    return { status: "unavailable", version: "", ...(await response.json().catch(() => ({}))) };
  }
  if (!response.ok) throw errorFromStatus(response.status);
  return (await response.json()) as Health;
}

export type StudyResult = { kind: "ok"; study: Study } | { kind: "absent" };

export async function fetchStudy(fetchImpl: Fetch = fetch): Promise<StudyResult> {
  const response = await request(fetchImpl, "/study");
  if (response.status === 404) return { kind: "absent" };
  if (!response.ok) throw errorFromStatus(response.status);
  return { kind: "ok", study: (await response.json()) as Study };
}

export async function fetchSample(url: string, fetchImpl: Fetch = fetch): Promise<Blob> {
  const response = await request(fetchImpl, url);
  if (!response.ok) throw new ApiError("server", response.status);
  return await response.blob();
}

// --- review queue (reviewers only: every call carries the reviewer's token) -----------------

export type ReviewErrorKind = "token" | "disabled" | "taken" | "gone" | "offline" | "server";

export const REVIEW_ERROR_MESSAGES: Record<ReviewErrorKind, string> = {
  token: "That reviewer token was not accepted.",
  disabled: "Reviewing is not switched on for this deployment.",
  taken: "This photo no longer needs your review: it is finished, or you have already reviewed it.",
  gone: "That review no longer exists.",
  offline: "Cannot reach the service. Check your connection, then try again.",
  server: "The service had a problem. Try again in a moment.",
};

export class ReviewError extends Error {
  kind: ReviewErrorKind;
  constructor(kind: ReviewErrorKind) {
    super(REVIEW_ERROR_MESSAGES[kind]);
    this.name = "ReviewError";
    this.kind = kind;
  }
}

function reviewErrorFrom(status: number): ReviewError {
  if (status === 401) return new ReviewError("token");
  if (status === 503) return new ReviewError("disabled");
  if (status === 409) return new ReviewError("taken");
  if (status === 404) return new ReviewError("gone");
  return new ReviewError("server");
}

async function reviewRequest(fetchImpl: Fetch, url: string, token: string, init: RequestInit = {}) {
  let response: Response;
  try {
    response = await fetchImpl(url, {
      ...init,
      headers: { ...(init.headers as Record<string, string>), Authorization: `Bearer ${token}` },
    });
  } catch {
    throw new ReviewError("offline");
  }
  if (!response.ok) throw reviewErrorFrom(response.status);
  return response;
}

export async function fetchQueue(token: string, fetchImpl: Fetch = fetch): Promise<ReviewQueue> {
  return (await (await reviewRequest(fetchImpl, "/review/queue", token)).json()) as ReviewQueue;
}

export async function submitReview(
  token: string,
  decisionId: string,
  label: string,
  fetchImpl: Fetch = fetch,
): Promise<ReviewItem> {
  const response = await reviewRequest(fetchImpl, `/review/${encodeURIComponent(decisionId)}`, token, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label }),
  });
  return (await response.json()) as ReviewItem;
}

export async function fetchReviewPhoto(token: string, decisionId: string, fetchImpl: Fetch = fetch): Promise<Blob> {
  return await (await reviewRequest(fetchImpl, `/review/${encodeURIComponent(decisionId)}/image`, token)).blob();
}
