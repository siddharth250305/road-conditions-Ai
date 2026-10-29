import { useRef, type CSSProperties } from "react";
import { CLASS_ORDER, classIndex, prettyClass, tintClass } from "../classes";
import { percent } from "../format";
import type { ViewState } from "../state";

const ROW = 56; // px per class row

export interface RulerReading {
  marker: number | null; // the one predicted class, only when the model is sure
  ghosts: number[]; // the top two, drawn hollow when it is not sure
  probabilities: Record<string, number> | null;
}

export function readingOf(state: ViewState): RulerReading {
  if (state.status === "result") {
    return {
      marker: classIndex(state.result.prediction),
      ghosts: [],
      probabilities: state.result.probabilities,
    };
  }
  if (state.status === "unsure") {
    return {
      marker: null,
      ghosts: state.result.top2.map((t) => classIndex(t.class)).filter((i) => i >= 0),
      probabilities: state.result.probabilities,
    };
  }
  return { marker: null, ghosts: [], probabilities: null };
}

export function FrictionRuler({ state }: { state: ViewState }) {
  const { marker, ghosts, probabilities } = readingOf(state);
  // Remember where the marker last stood, so a new result travels from there.
  const resting = useRef(0);
  if (marker !== null) resting.current = marker;
  const style = { "--row": `${ROW}px`, "--i": marker ?? resting.current } as CSSProperties;

  return (
    <div className="relative" style={style}>
      <ol aria-label="Friction ruler, dry to ice" className="relative m-0 list-none p-0">
        {CLASS_ORDER.map((name, i) => {
          const p = probabilities?.[name];
          const active = marker === i;
          return (
            <li
              key={name}
              data-class={name}
              className="flex h-[var(--row)] items-center border-t border-rule last:border-b"
            >
              <span className="w-3 shrink-0" aria-hidden="true" />
              <span
                className={`h-full w-9 shrink-0 border-x border-rule ${tintClass(name)}`}
                aria-hidden="true"
              />
              <span className="mx-3 h-px w-5 bg-rule" aria-hidden="true" />
              <span className={`data flex-1 uppercase ${active ? "font-medium" : ""}`}>
                {prettyClass(name)}
              </span>
              <span className="data tabular-nums">{p === undefined ? "–" : percent(p)}</span>
            </li>
          );
        })}
      </ol>

      <div
        data-testid="ruler-marker"
        data-index={marker ?? ""}
        aria-hidden="true"
        className="ruler-marker pointer-events-none absolute top-0 left-0 h-[var(--row)] w-3 border-2 border-ink bg-lane"
        style={{ opacity: marker === null ? 0 : 1 }}
      />
      {ghosts.map((index) => (
        <div
          key={index}
          data-testid="ruler-ghost"
          aria-hidden="true"
          className="pointer-events-none absolute left-0 h-[var(--row)] w-3 border-2 border-dashed border-text"
          style={{ top: index * ROW }}
        />
      ))}
    </div>
  );
}
