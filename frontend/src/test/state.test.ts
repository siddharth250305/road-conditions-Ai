import { ApiError } from "../api";
import { initialState, previewOf, reducer, type ViewState } from "../state";
import { SURE, UNSURE } from "./fixtures";

const loading = reducer(initialState, { type: "start", name: "a.jpg", previewUrl: "blob:1" });

describe("view state machine", () => {
  it("starts idle and moves to loading with the picture", () => {
    expect(initialState.status).toBe("idle");
    expect(loading).toMatchObject({ status: "loading", name: "a.jpg", previewUrl: "blob:1" });
  });

  it("a confident answer is a result", () => {
    expect(reducer(loading, { type: "success", result: SURE }).status).toBe("result");
  });

  it("an uncertain answer is its own 'unsure' state, never a result", () => {
    const next = reducer(loading, { type: "success", result: UNSURE });
    expect(next.status).toBe("unsure");
  });

  it("failures become error, or offline when the network is gone", () => {
    expect(reducer(loading, { type: "failure", error: new ApiError("not_image") })).toMatchObject({
      status: "error",
      kind: "not_image",
    });
    expect(reducer(loading, { type: "failure", error: new ApiError("offline") }).status).toBe(
      "offline",
    );
  });

  it("ignores answers that arrive when nothing is loading", () => {
    expect(reducer(initialState, { type: "success", result: SURE })).toBe(initialState);
    const error: ViewState = reducer(loading, { type: "failure", error: new ApiError("server") });
    expect(reducer(error, { type: "success", result: SURE })).toBe(error);
  });

  it("reset returns to idle and the preview goes away", () => {
    expect(reducer(loading, { type: "reset" })).toBe(initialState);
    expect(previewOf(initialState)).toBeNull();
    expect(previewOf(loading)).toBe("blob:1");
  });
});
