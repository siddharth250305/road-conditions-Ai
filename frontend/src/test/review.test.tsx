import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "../App";
import { fetchQueue, submitReview } from "../api";
import { reasonText } from "../review";
import type { Health, Review, ReviewItem, ReviewQueue } from "../types";
import { HEALTH, json, jpeg, mockFetch, STUDY, SURE } from "./fixtures";

const ROUTED: Health = { ...HEALTH, routing: { enabled: true, reviewing: true, stores_images: false } };
const REVIEW: Review = {
  decision_id: "d-1",
  mode: "second_review",
  reasons: ["safety_class:ice", "conf>=0.85"],
  confidence: 0.91,
  fallback: false,
  reviews_needed: 2,
  recorded: true,
};
const ITEM: ReviewItem = {
  decision_id: "d-1",
  routed_at: "2026-11-08T10:00:00+00:00",
  prediction: "ice",
  mode: "second_review",
  reasons: ["safety_class:ice"],
  confidence: 0.91,
  status: "pending",
  agreed_label: null,
  reviews_needed: 2,
  reviewers: ["sam"],
  has_image: false,
};
const counts = (open: number) => ({
  light_review: { open: 0, done: 0 },
  full_review: { open: 0, done: 0 },
  second_review: { open, done: 0 },
});
const QUEUE: ReviewQueue = { reviewer: "alex", counts: counts(1), items: [ITEM] };

afterEach(() => {
  window.location.hash = "";
  window.sessionStorage.clear();
});

describe("review API", () => {
  it("sends the reviewer token and the label", async () => {
    const { calls } = mockFetch({
      "/review/queue": () => json(QUEUE),
      "/review/d-1": () => json({ ...ITEM, status: "done" }),
    });
    expect((await fetchQueue("tok")).items).toHaveLength(1);
    await submitReview("tok", "d-1", "wet");
    expect((calls[0]!.init!.headers as Record<string, string>).Authorization).toBe("Bearer tok");
    expect(calls[1]!.init!.method).toBe("POST");
    expect(JSON.parse(calls[1]!.init!.body as string)).toEqual({ label: "wet" });
  });

  it.each([
    [401, "token"],
    [503, "disabled"],
    [409, "taken"],
    [404, "gone"],
    [500, "server"],
  ])("maps %i to %s", async (status, kind) => {
    mockFetch({ "/review/queue": () => json({ detail: "internal detail" }, status) });
    await expect(fetchQueue("tok")).rejects.toMatchObject({ kind });
  });
});

describe("reason wording", () => {
  it.each([
    ["safety_class:ice", "The answer is ice, a safety-critical class."],
    ["conf<0.85", "Review confidence is below 85%."],
    ["conf<0.5", "Review confidence is below 50%."],
    ["conf>=0.85", "Review confidence is 85% or more."],
    ["missing_signal:calibrated_confidence", "A confidence signal was unavailable, so the photo is checked more strictly."],
    ["fallback:log_unavailable", "The decision could not be recorded, so the photo gets a full review."],
  ])("%s", (reason, text) => {
    expect(reasonText(reason)).toBe(text);
  });
});

describe("the report", () => {
  beforeEach(() => {
    window.location.hash = "#/report";
  });

  it("says how much human review the photo gets, and why", async () => {
    mockFetch({ "/health": () => json(ROUTED), "/study": () => json(STUDY), "/predict": () => json({ ...SURE, review: REVIEW }) });
    const { container } = render(<App />);
    await userEvent.upload(container.querySelector<HTMLInputElement>('input[type="file"]')!, jpeg());
    const note = await screen.findByTestId("review-note");
    expect(note).toHaveTextContent("Second review");
    expect(note).toHaveTextContent("Two people check it independently.");
    expect(note).toHaveTextContent("The answer is ice, a safety-critical class.");
    expect(note).toHaveTextContent("91%");
  });

  it("shows no review note when the server does not route photos", async () => {
    mockFetch({ "/health": () => json(HEALTH), "/study": () => json(STUDY), "/predict": () => json(SURE) });
    const { container } = render(<App />);
    await userEvent.upload(container.querySelector<HTMLInputElement>('input[type="file"]')!, jpeg());
    await screen.findByTestId("stamp");
    expect(screen.queryByTestId("review-note")).toBeNull();
    const nav = screen.getByRole("navigation", { name: /sections/i });
    expect(within(nav).queryByRole("link", { name: "Review" })).toBeNull();
  });
});

describe("the review queue", () => {
  beforeEach(() => {
    window.location.hash = "#/review";
  });

  it("signs in with a token, shows the queue and records a label", async () => {
    let open = [ITEM];
    const { calls } = mockFetch({
      "/health": () => json(ROUTED),
      "/study": () => json(STUDY),
      "/review/queue": () => json({ ...QUEUE, counts: counts(open.length), items: open }),
      "/review/d-1": () => {
        open = [];
        return json({ ...ITEM, status: "done", agreed_label: "ice", reviewers: ["sam", "alex"] });
      },
    });
    render(<App />);
    const nav = screen.getByRole("navigation", { name: /sections/i });
    expect(within(nav).getByRole("link", { name: "Review" })).toHaveAttribute("aria-current", "page");
    await userEvent.type(screen.getByLabelText("Reviewer token"), "alex-token");
    await userEvent.click(screen.getByRole("button", { name: "Open queue" }));

    const card = await screen.findByTestId("review-card");
    expect(screen.getByText("alex")).toBeInTheDocument();
    expect(card).toHaveTextContent("1 of 2 reviews");
    expect(card).toHaveTextContent("Model says");
    expect(card).toHaveTextContent("does not keep photos");
    expect(screen.getByTestId("queue-counts")).toHaveTextContent("Second review 1 open");
    expect(window.sessionStorage.getItem("roadcond.reviewerToken")).toBe("alex-token");

    const choices = within(card).getByRole("group", { name: "What is on the road?" });
    expect(within(choices).getAllByRole("button")).toHaveLength(6);
    await userEvent.click(within(choices).getByRole("button", { name: /^ice/i }));
    await screen.findByText("Nothing is waiting for you.");
    const post = calls.find((c) => c.url === "/review/d-1")!;
    expect(JSON.parse(post.init!.body as string)).toEqual({ label: "ice" });
  });

  it("forgets a token the server does not accept", async () => {
    mockFetch({
      "/health": () => json(ROUTED),
      "/study": () => json(STUDY),
      "/review/queue": () => json({ detail: "no" }, 401),
    });
    render(<App />);
    await userEvent.type(screen.getByLabelText("Reviewer token"), "wrong");
    await userEvent.click(screen.getByRole("button", { name: "Open queue" }));
    expect(await screen.findByText("That reviewer token was not accepted.")).toBeInTheDocument();
    expect(screen.getByLabelText("Reviewer token")).toBeInTheDocument();
    expect(window.sessionStorage.getItem("roadcond.reviewerToken")).toBeNull();
  });

  it("says so when review routing is off", async () => {
    mockFetch({ "/health": () => json(HEALTH), "/study": () => json(STUDY) });
    render(<App />);
    await waitFor(() => expect(screen.getByText(/review routing is off on this deployment/i)).toBeInTheDocument());
    expect(screen.queryByLabelText("Reviewer token")).toBeNull();
  });
});
