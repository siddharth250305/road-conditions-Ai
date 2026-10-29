import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "../App";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { HEALTH, json, jpeg, mockFetch, STUDY, SURE, UNSURE } from "./fixtures";

const live = () => screen.getByRole("status");
const fileInput = (container: HTMLElement) =>
  container.querySelector<HTMLInputElement>('input[type="file"]')!;

async function upload(container: HTMLElement) {
  await userEvent.upload(fileInput(container), jpeg());
}

function healthy(extra: Record<string, () => Response | Promise<Response>> = {}) {
  return mockFetch({ "/health": () => json(HEALTH), "/study": () => json(STUDY), ...extra });
}

describe("empty state", () => {
  it("shows the blank specimen frame, three samples and the limits line", async () => {
    healthy();
    render(<App />);
    expect(screen.getByRole("button", { name: /drop a road photo/i })).toBeInTheDocument();
    const samples = screen.getByRole("region", { name: /try a sample/i });
    expect(within(samples).getAllByRole("button")).toHaveLength(3);
    expect(screen.getByTestId("limits-line")).toHaveTextContent(/only on RSCD/i);
    expect(screen.getByText(/awaiting photo/i)).toBeInTheDocument();
    await screen.findByTestId("mock-banner"); // the health check says the model is a mock
  });

  it("has no mock banner for a real model", async () => {
    healthy({ "/health": () => json({ ...HEALTH, mock: false }) });
    render(<App />);
    await waitFor(() => expect(fetch).toHaveBeenCalledWith("/health", undefined));
    expect(screen.queryByTestId("mock-banner")).not.toBeInTheDocument();
  });
});

describe("a confident result", () => {
  it("shows the verdict stamp, calibrated confidence and announces it", async () => {
    healthy({ "/predict": () => json(SURE) });
    const { container } = render(<App />);
    await upload(container);
    const stamp = await screen.findByTestId("stamp");
    expect(stamp).toHaveTextContent(/wet/i);
    expect(screen.getByText("87%", { selector: "p" })).toBeInTheDocument();
    expect(screen.getByText(/calibrated on held-out data \(RSCD\)/i)).toBeInTheDocument();
    expect(live()).toHaveTextContent(/Result: wet, confidence 87%, calibrated on held-out data/);
    expect(screen.getByRole("img", { name: /predicted wet, confidence 87%/i })).toBeInTheDocument();
  });

  it("shows the limits line on every result", async () => {
    healthy({ "/predict": () => json(SURE) });
    const { container } = render(<App />);
    await upload(container);
    await screen.findByTestId("stamp");
    expect(screen.getByTestId("limits-line")).toBeVisible();
  });

  it("runs a sample picture through the real endpoint", async () => {
    const { calls } = healthy({
      // jsdom's FormData only accepts jsdom Blobs, while Response.blob() returns Node's.
      "/samples/sample-1.jpg": () =>
        ({ ok: true, status: 200, blob: async () => new Blob(["jpg"], { type: "image/jpeg" }) }) as Response,
      "/predict": () => json(SURE),
    });
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /sample 1/i }));
    await screen.findByTestId("stamp");
    expect(calls.map((c) => c.url)).toEqual(["/health", "/samples/sample-1.jpg", "/predict"]);
  });

  it("toggles the heatmap with aria-pressed and a crossfade layer", async () => {
    healthy({ "/predict": () => json({ ...SURE, heatmap_base64: "AAAA", heatmap_media_type: "image/png" }) });
    const { container } = render(<App />);
    await upload(container);
    const layer = await screen.findByTestId("heatmap-layer");
    const toggle = screen.getByRole("button", { name: /heatmap on/i });
    expect(toggle).toHaveAttribute("aria-pressed", "true");
    expect(layer).toHaveStyle({ opacity: "1" });
    expect(layer.getAttribute("src")).toBe("data:image/png;base64,AAAA");
    await userEvent.click(toggle);
    expect(screen.getByRole("button", { name: /heatmap off/i })).toHaveAttribute("aria-pressed", "false");
    expect(layer).toHaveStyle({ opacity: "0" });
  });

  it("clear returns to the empty state", async () => {
    healthy({ "/predict": () => json(SURE) });
    const { container } = render(<App />);
    await upload(container);
    await screen.findByTestId("stamp");
    await userEvent.click(screen.getByRole("button", { name: /clear/i }));
    expect(screen.getByRole("button", { name: /drop a road photo/i })).toBeInTheDocument();
  });
});

describe("low confidence", () => {
  it("says 'Not sure' with the top two classes and never picks a label", async () => {
    healthy({ "/predict": () => json(UNSURE) });
    const { container } = render(<App />);
    await upload(container);
    const stamp = await screen.findByTestId("stamp");
    expect(stamp).toHaveTextContent(/not sure/i);
    for (const name of ["dry", "wet", "water", "fresh snow", "melted snow", "ice"]) {
      expect(stamp).not.toHaveTextContent(new RegExp(`\\b${name}\\b`, "i"));
    }
    for (const entry of UNSURE.top2) {
      expect(screen.getAllByText(new RegExp(entry.class, "i")).length).toBeGreaterThan(0);
    }
    expect(live()).toHaveTextContent(/^Not sure\. Closest classes: ice 17%, then wet 17%\.$/);
    expect(screen.getByText(/no way to say so/i)).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /not sure/i })).toBeInTheDocument();
  });
});

describe("errors", () => {
  it("explains an unreadable file and offers no pointless retry", async () => {
    healthy({ "/predict": () => json({ detail: "boom" }, 400) });
    const { container } = render(<App />);
    await upload(container);
    const box = await screen.findByTestId("state-error");
    expect(box).toHaveTextContent(/could not be read as a picture/i);
    expect(within(box).queryByRole("button")).not.toBeInTheDocument();
    expect(live()).toHaveTextContent(/could not be read/i);
  });

  it("says the model is missing on a 503 and can retry", async () => {
    let calls = 0;
    healthy({ "/predict": () => (++calls === 1 ? json({ detail: "x" }, 503) : json(SURE)) });
    const { container } = render(<App />);
    await upload(container);
    const box = await screen.findByTestId("state-error");
    expect(box).toHaveTextContent(/no model loaded/i);
    await userEvent.click(within(box).getByRole("button", { name: /try again/i }));
    expect(await screen.findByTestId("stamp")).toHaveTextContent(/wet/i);
  });

  it("shows the offline state when the service cannot be reached, then recovers", async () => {
    let up = false;
    healthy({ "/predict": () => (up ? json(SURE) : Promise.reject(new TypeError("Failed to fetch"))) });
    const { container } = render(<App />);
    await upload(container);
    const box = await screen.findByTestId("state-offline");
    expect(box).toHaveTextContent(/cannot reach the service/i);
    up = true;
    await userEvent.click(within(box).getByRole("button", { name: /try again/i }));
    expect(await screen.findByTestId("stamp")).toBeInTheDocument();
  });

  it("warns when the browser reports it is offline", async () => {
    healthy();
    render(<App />);
    vi.spyOn(navigator, "onLine", "get").mockReturnValue(false);
    window.dispatchEvent(new Event("offline"));
    expect(await screen.findByText(/you are offline/i)).toBeInTheDocument();
  });

  it("tells the visitor when the service has no model", async () => {
    healthy({ "/health": () => json({ status: "unavailable", detail: "no bundle" }, 503) });
    render(<App />);
    expect(await screen.findByText(/no model is loaded/i)).toBeInTheDocument();
  });

  it("only the newest upload may change the page", async () => {
    let release!: () => void;
    const slow = new Promise<Response>((resolve) => {
      release = () => resolve(json({ ...SURE, prediction: "ice" }));
    });
    let n = 0;
    healthy({ "/predict": () => (++n === 1 ? slow : json(SURE)) });
    const { container } = render(<App />);
    await upload(container);
    await upload(container);
    expect(await screen.findByTestId("stamp")).toHaveTextContent(/wet/i);
    release();
    await new Promise((r) => setTimeout(r, 20));
    expect(screen.getByTestId("stamp")).toHaveTextContent(/wet/i);
  });
});

describe("limits footer", () => {
  it("expands and collapses with aria-expanded and shows calibration and provenance", async () => {
    healthy({ "/predict": () => json(SURE) });
    const { container } = render(<App />);
    await upload(container);
    await screen.findByTestId("stamp");
    const button = screen.getByRole("button", { name: /full limits/i });
    expect(button).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(button);
    expect(button).toHaveAttribute("aria-expanded", "true");
    const panel = document.getElementById(button.getAttribute("aria-controls")!)!;
    expect(panel).toBeVisible();
    expect(panel).toHaveTextContent(SURE.limits_note.slice(0, 40));
    expect(panel).toHaveTextContent(/China/);
    expect(panel).toHaveTextContent(/T = 1\.70/);
    expect(panel).toHaveTextContent(/checkpoint sha256/);
    await userEvent.click(button);
    expect(panel).not.toBeVisible();
  });
});

describe("study tab", () => {
  it("draws both charts and the numbers behind them", async () => {
    healthy();
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "Study" }));
    expect(await screen.findByRole("img", { name: /detection accuracy at home and abroad/i })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /accuracy as target labels are added/i })).toBeInTheDocument();
    const table = screen.getByRole("table");
    for (const country of STUDY.slope) {
      expect(within(table).getByRole("rowheader", { name: country.country.replace(/_/g, " ") })).toBeInTheDocument();
    }
    expect(screen.getAllByTestId("error-bar").length).toBeGreaterThan(0);
    expect(screen.getAllByRole("note").some((n) => /synthetic numbers/i.test(n.textContent ?? ""))).toBe(true);
  });

  it("says so, and draws nothing, when no study has been added", async () => {
    healthy({ "/study": () => json({ detail: "none" }, 404) });
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "Study" }));
    expect(await screen.findByTestId("study-absent")).toHaveTextContent(/nothing to draw/i);
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });
});

describe("error boundary", () => {
  it("shows a plain message instead of a blank page", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const Broken = () => {
      throw new Error("bug");
    };
    render(
      <ErrorBoundary>
        <Broken />
      </ErrorBoundary>,
    );
    expect(screen.getByRole("alert")).toHaveTextContent(/something broke/i);
  });
});
