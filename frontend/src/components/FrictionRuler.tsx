import { useRef, type CSSProperties } from "react";
import { CLASS_ORDER, classIndex, prettyClass, tintClass } from "../classes";
import { percent } from "../format";
import type { ViewState } from "../state";

const ROW = 48; // px per class row

export interface RulerReading {
  marker: number | null; // the one predicted class, only when the model is sure
  ghosts: number[]; // the top two, drawn dashed when it is not sure
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

// Six lanes, dry to ice. The committed class carries a solid marking; candidates are dashed;
// every class is always drawn, so an empty lane is visible as empty rather than missing.
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
              className="flex h-[var(--row)] items-center border-t border-a10/10 last:border-b"
            >
              <span className="w-4 shrink-0" aria-hidden="true" />
              <span className={`h-7 w-7 shrink-0 rounded-md border border-a0/40 ${tintClass(name)}`} aria-hidden="true" />
              <span className={`ml-3 w-28 shrink-0 text-[15px] first-letter:uppercase ${active ? "font-semibold text-a10" : "text-muted"}`}>
                {prettyClass(name)}
              </span>
              <span className="relative mr-4 h-1.5 flex-1 overflow-hidden rounded-full bg-a10/8" aria-hidden="true">
                <span
                  className="absolute inset-y-0 left-0 rounded-full bg-a10"
                  style={{ width: `${Math.round((p ?? 0) * 100)}%`, opacity: active ? 1 : 0.4 }}
                />
              </span>
              <span className={`data w-12 text-right ${active ? "font-semibold text-a10" : "text-muted"}`}>
                {p === undefined ? "–" : percent(p)}
              </span>
            </li>
          );
        })}
      </ol>

      <div
        data-testid="ruler-marker"
        data-index={marker ?? ""}
        aria-hidden="true"
        className="ruler-marker pointer-events-none absolute top-1.5 left-0 h-[calc(var(--row)-12px)] w-[5px] rounded-full bg-a10"
        style={{ opacity: marker === null ? 0 : 1 }}
      />
      {ghosts.map((index) => (
        <div
          key={index}
          data-testid="ruler-ghost"
          aria-hidden="true"
          className="pointer-events-none absolute left-0 w-[5px] border-l-[3px] border-dashed border-a9"
          style={{ top: index * ROW + 6, height: ROW - 12 }}
        />
      ))}
    </div>
  );
}
