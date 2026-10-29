import { render, screen, within } from "@testing-library/react";
import { FrictionRuler, readingOf } from "../components/FrictionRuler";
import { CLASS_ORDER } from "../classes";
import { initialState, type ViewState } from "../state";
import { SURE, UNSURE } from "./fixtures";

const sure: ViewState = { status: "result", name: "a.jpg", previewUrl: "blob:1", result: SURE };
const unsure: ViewState = { status: "unsure", name: "a.jpg", previewUrl: "blob:1", result: UNSURE };

describe("friction ruler", () => {
  it("has a tick for every class, dry first and ice last", () => {
    render(<FrictionRuler state={initialState} />);
    const rows = within(screen.getByRole("list", { name: /friction ruler/i })).getAllByRole("listitem");
    expect(rows.map((r) => r.getAttribute("data-class"))).toEqual([...CLASS_ORDER]);
    expect(CLASS_ORDER[0]).toBe("dry");
    expect(CLASS_ORDER[5]).toBe("ice");
  });

  it("shows no marker before there is a result", () => {
    render(<FrictionRuler state={initialState} />);
    expect(screen.getByTestId("ruler-marker")).toHaveStyle({ opacity: "0" });
  });

  it("puts the marker on the predicted class and prints every probability", () => {
    render(<FrictionRuler state={sure} />);
    const index = CLASS_ORDER.indexOf(SURE.prediction as (typeof CLASS_ORDER)[number]);
    const marker = screen.getByTestId("ruler-marker");
    expect(marker).toHaveStyle({ opacity: "1" });
    expect(marker.getAttribute("data-index")).toBe(String(index));
    expect(marker.parentElement!.style.getPropertyValue("--i")).toBe(String(index));
    expect(screen.queryAllByTestId("ruler-ghost")).toHaveLength(0);
    const rows = screen.getAllByRole("listitem");
    expect(rows[index]).toHaveTextContent(/87%/);
  });

  it("when unsure, hides the marker and shows the top two as hollow ghosts", () => {
    render(<FrictionRuler state={unsure} />);
    expect(screen.getByTestId("ruler-marker")).toHaveStyle({ opacity: "0" });
    expect(screen.getAllByTestId("ruler-ghost")).toHaveLength(2);
  });

  it("readingOf never reports a single class for an unsure answer", () => {
    expect(readingOf(unsure).marker).toBeNull();
    expect(readingOf(unsure).ghosts).toHaveLength(2);
  });

  it("keeps the marker where it last stood, so the next answer travels from there", () => {
    const { rerender } = render(<FrictionRuler state={sure} />);
    const index = CLASS_ORDER.indexOf(SURE.prediction as (typeof CLASS_ORDER)[number]);
    rerender(<FrictionRuler state={initialState} />);
    expect(screen.getByTestId("ruler-marker").parentElement!.style.getPropertyValue("--i")).toBe(
      String(index),
    );
  });
});
