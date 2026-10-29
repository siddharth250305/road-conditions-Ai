import type { Health, PredictResponse, Study } from "./types";

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
