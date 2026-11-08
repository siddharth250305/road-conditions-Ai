import { prettyClass } from "./classes";
import type { ReviewMode } from "./types";

// Plain wording for the review router's modes and rule ids (roadcond.routing.policy).

export const MODES: Record<ReviewMode, { name: string; what: string; people: string }> = {
  light_review: { name: "Light review", what: "A person glances at the label.", people: "1 person" },
  full_review: { name: "Full review", what: "A person checks the photo carefully.", people: "1 person" },
  second_review: { name: "Second review", what: "Two people check it independently.", people: "2 people" },
};

const share = (value: string) => `${Math.round(Number(value) * 100)}%`;

export function reasonText(reason: string): string {
  if (reason.startsWith("safety_class:")) {
    return `The answer is ${prettyClass(reason.slice("safety_class:".length))}, a safety-critical class.`;
  }
  if (reason.startsWith("conf>=")) return `Review confidence is ${share(reason.slice(6))} or more.`;
  if (reason.startsWith("conf<")) return `Review confidence is below ${share(reason.slice(5))}.`;
  if (reason === "models_disagree") return "The models disagree.";
  if (reason.startsWith("unusable_prediction:")) return "The model's output could not be used.";
  if (reason.startsWith("missing_signal:") || reason === "fallback:provider_error") {
    return "A confidence signal was unavailable, so the photo is checked more strictly.";
  }
  if (reason === "fallback:policy_error") return "The review rules failed, so the photo gets a full review.";
  if (reason === "fallback:log_unavailable") {
    return "The decision could not be recorded, so the photo gets a full review.";
  }
  return reason;
}

// The reviewer token lives for this browser tab only; storage can be blocked, so it may not.
const TOKEN_KEY = "roadcond.reviewerToken";

export function loadToken(): string {
  try {
    return window.sessionStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

export function saveToken(token: string): void {
  try {
    if (token) window.sessionStorage.setItem(TOKEN_KEY, token);
    else window.sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    // the token then lasts until the page reloads
  }
}
