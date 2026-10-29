import { prettyClass, tintClass } from "../classes";
import { percent } from "../format";
import type { ViewState } from "../state";

interface Props {
  state: ViewState;
  onRetry: () => void;
}

export function liveMessage(state: ViewState): string {
  switch (state.status) {
    case "loading":
      return "Reading the photo.";
    case "result":
      return `Result: ${prettyClass(state.result.prediction)}, confidence ${percent(state.result.confidence)}, calibrated on held-out data.`;
    case "unsure": {
      const [a, b] = state.result.top2;
      const two = [a, b]
        .filter((t) => t !== undefined)
        .map((t) => `${prettyClass(t.class)} ${percent(t.probability)}`)
        .join(", then ");
      return `Not sure. Closest classes: ${two}.`;
    }
    case "error":
    case "offline":
      return state.message;
    default:
      return "";
  }
}

export function VerdictPanel({ state, onRetry }: Props) {
  return (
    <div className="mt-6 min-h-[16rem]">
      <p className="caption mb-2">Verdict</p>
      {state.status === "idle" && (
        <p className="display border-2 border-dashed border-rule p-3 text-[clamp(3.5rem,8vw,5rem)] text-muted">
          Awaiting photo
        </p>
      )}
      {state.status === "loading" && (
        <p className="display p-3 text-[clamp(3.5rem,8vw,5rem)] text-muted">Reading…</p>
      )}
      {state.status === "result" && (
        <>
          <div
            key={`${state.name}-${state.result.prediction}`}
            data-testid="stamp"
            className="stamp inline-flex items-center gap-4 border-4 border-double border-rule px-4 py-2"
          >
            <span className={`h-10 w-10 border-2 border-rule ${tintClass(state.result.prediction)}`} aria-hidden="true" />
            <span className="display text-[clamp(3.5rem,8vw,6rem)]">
              {prettyClass(state.result.prediction)}
            </span>
          </div>
          <p className="display mt-4 text-5xl">{percent(state.result.confidence)}</p>
          <p className="data">calibrated on held-out data (RSCD)</p>
        </>
      )}
      {state.status === "unsure" && (
        <>
          <div
            key={state.name}
            data-testid="stamp"
            className="stamp inline-block border-4 border-double border-rule px-4 py-2"
          >
            <span className="display text-[clamp(3.5rem,8vw,6rem)]">
              {state.result.display_label}
            </span>
          </div>
          <p className="caption mt-4">Closest two classes</p>
          <ul className="data m-0 list-none p-0">
            {state.result.top2.map((t) => (
              <li key={t.class} className="flex items-center gap-3 py-1">
                <span className={`h-4 w-4 border border-rule ${tintClass(t.class)}`} aria-hidden="true" />
                <span className="uppercase">{prettyClass(t.class)}</span>
                <span className="tabular-nums">{percent(t.probability)}</span>
              </li>
            ))}
          </ul>
          <p className="mt-3 max-w-prose">
            The best guess is below {percent(state.result.uncertainty_threshold)}, so the page does
            not pick a label. If this is not a road photo, the model has no way to say so: it only
            knows six road classes.
          </p>
        </>
      )}
      {(state.status === "error" || state.status === "offline") && (
        <div className="border-2 border-rule p-4" data-testid={`state-${state.status}`}>
          <p className="display text-4xl">
            {state.status === "offline" ? "Offline" : "Could not read"}
          </p>
          <p className="mt-2 max-w-prose">{state.message}</p>
          {state.kind !== "not_image" && state.kind !== "too_large" && (
            <button type="button" className="btn-primary mt-3" onClick={onRetry}>
              Try again
            </button>
          )}
        </div>
      )}
    </div>
  );
}
