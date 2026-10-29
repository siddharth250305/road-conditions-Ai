import type { ApiError, ErrorKind } from "./api";
import type { PredictResponse } from "./types";

export type ViewState =
  | { status: "idle" }
  | { status: "loading"; name: string; previewUrl: string }
  | { status: "result"; name: string; previewUrl: string; result: PredictResponse }
  | { status: "unsure"; name: string; previewUrl: string; result: PredictResponse }
  | { status: "error"; kind: ErrorKind; message: string; name: string; previewUrl: string }
  | { status: "offline"; kind: ErrorKind; message: string; name: string; previewUrl: string };

export type Action =
  | { type: "start"; name: string; previewUrl: string }
  | { type: "success"; result: PredictResponse }
  | { type: "failure"; error: ApiError }
  | { type: "reset" };

export const initialState: ViewState = { status: "idle" };

// A low-confidence answer is its own state, so the page can never show a forced label for it.
export function reducer(state: ViewState, action: Action): ViewState {
  switch (action.type) {
    case "start":
      return { status: "loading", name: action.name, previewUrl: action.previewUrl };
    case "success":
      if (state.status !== "loading") return state;
      return {
        status: action.result.is_uncertain ? "unsure" : "result",
        name: state.name,
        previewUrl: state.previewUrl,
        result: action.result,
      };
    case "failure": {
      if (state.status !== "loading") return state;
      const { name, previewUrl } = state;
      const kind = action.error.kind;
      return {
        status: kind === "offline" ? "offline" : "error",
        kind,
        message: action.error.message,
        name,
        previewUrl,
      };
    }
    case "reset":
      return initialState;
  }
}

export function previewOf(state: ViewState): string | null {
  return state.status === "idle" ? null : state.previewUrl;
}
