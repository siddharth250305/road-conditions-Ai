import { ApiError, errorFromStatus, fetchHealth, fetchStudy, predict } from "../api";
import { HEALTH, json, mockFetch, STUDY, SURE } from "./fixtures";

describe("errorFromStatus", () => {
  it.each([
    [400, "not_image"],
    [415, "not_image"],
    [422, "not_image"],
    [413, "too_large"],
    [503, "no_model"],
    [500, "server"],
    [502, "server"],
  ])("maps %i to %s", (status, kind) => {
    expect(errorFromStatus(status).kind).toBe(kind);
  });

  it("uses fixed wording, never the server's text", () => {
    const error = errorFromStatus(500);
    expect(error.message).not.toMatch(/traceback|exception|onnx/i);
  });
});

describe("predict", () => {
  it("posts the file as multipart form data and returns the answer", async () => {
    const { calls } = mockFetch({ "/predict": () => json(SURE) });
    const result = await predict(new Blob(["x"]), "road.jpg");
    expect(result.prediction).toBe(SURE.prediction);
    expect(calls[0]!.init!.method).toBe("POST");
    expect((calls[0]!.init!.body as FormData).get("file")).toBeInstanceOf(Blob);
  });

  it("turns a failed request into an offline error", async () => {
    mockFetch({});
    await expect(predict(new Blob(["x"]))).rejects.toMatchObject({ kind: "offline" });
  });

  it.each([
    [400, "not_image"],
    [413, "too_large"],
    [503, "no_model"],
    [500, "server"],
  ])("maps HTTP %i to a %s error", async (status, kind) => {
    mockFetch({ "/predict": () => json({ detail: "secret internals" }, status) });
    const error = await predict(new Blob(["x"])).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).kind).toBe(kind);
    expect((error as ApiError).message).not.toContain("secret");
  });
});

describe("fetchHealth", () => {
  it("returns the service's own description", async () => {
    mockFetch({ "/health": () => json(HEALTH) });
    expect((await fetchHealth()).model).toBe("small_cnn");
  });

  it("reads a 503 as 'unavailable' instead of throwing", async () => {
    mockFetch({ "/health": () => json({ status: "unavailable", detail: "no bundle" }, 503) });
    expect((await fetchHealth()).status).toBe("unavailable");
  });

  it("throws offline when the service cannot be reached", async () => {
    mockFetch({});
    await expect(fetchHealth()).rejects.toMatchObject({ kind: "offline" });
  });
});

describe("fetchStudy", () => {
  it("returns the benchmark", async () => {
    mockFetch({ "/study": () => json(STUDY) });
    const result = await fetchStudy();
    expect(result.kind).toBe("ok");
  });

  it("treats a 404 as 'absent', not as an error", async () => {
    mockFetch({ "/study": () => json({ detail: "none" }, 404) });
    expect(await fetchStudy()).toEqual({ kind: "absent" });
  });
});
