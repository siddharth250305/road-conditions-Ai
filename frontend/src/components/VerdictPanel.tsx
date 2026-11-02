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
    <div className="min-h-[13rem]">
      {state.status === "idle" && (
        <>
          <p className="display m-0 text-[clamp(2.25rem,3.6vw,3rem)] text-a10/40">Awaiting photo</p>
          <p className="mt-3 max-w-[40ch] text-[15px] text-muted">
            Drop a photo or pick a sample. The reading appears here, with how sure the model is.
          </p>
        </>
      )}
      {state.status === "loading" && (
        <p className="display m-0 text-[clamp(2.25rem,3.6vw,3rem)] text-a10/60">Reading…</p>
      )}
      {state.status === "result" && (
        <>
          <div
            key={`${state.name}-${state.result.prediction}`}
            data-testid="stamp"
            className="stamp inline-flex items-center gap-4 rounded-[var(--r-panel)] border-2 border-solid border-a10 bg-a10/6 py-3 pr-6 pl-3"
          >
            <span className={`h-12 w-12 shrink-0 rounded-lg border border-a0/40 ${tintClass(state.result.prediction)}`} aria-hidden="true" />
            <span className="display text-[clamp(2.75rem,5vw,4.5rem)] leading-none first-letter:uppercase">
              {prettyClass(state.result.prediction)}
            </span>
          </div>
          <div className="mt-6 flex items-baseline gap-3">
            <p className="data m-0 !text-[44px] font-semibold !tracking-[-0.02em] leading-none">{percent(state.result.confidence)}</p>
            <span className="text-[14px] text-muted">calibrated on held-out data (RSCD)</span>
          </div>
        </>
      )}
      {state.status === "unsure" && (
        <>
          <div key={state.name} data-testid="stamp" className="stamp dashed inline-block rounded-[var(--r-panel)] px-6 py-3">
            <span className="display text-[clamp(2.75rem,5vw,4.5rem)] leading-none">{state.result.display_label}</span>
          </div>
          <h3 className="heading mt-6 mb-1 text-muted">Closest two classes</h3>
          <ul className="m-0 list-none p-0 text-[15px]">
            {state.result.top2.map((t) => (
              <li key={t.class} className="flex items-center gap-3 py-1.5 font-semibold">
                <span className={`h-5 w-5 rounded border border-a0/40 ${tintClass(t.class)}`} aria-hidden="true" />
                <span className="first-letter:uppercase">{prettyClass(t.class)}</span>
                <span className="data text-muted">{percent(t.probability)}</span>
              </li>
            ))}
          </ul>
          <p className="mt-3 max-w-[48ch] text-[14px] leading-relaxed text-muted">
            The best guess is below {percent(state.result.uncertainty_threshold)}, so the page does
            not pick a label. If this is not a road photo, the model has no way to say so: it only
            knows six road classes.
          </p>
        </>
      )}
      {(state.status === "error" || state.status === "offline") && (
        <div className="wash p-5" data-testid={`state-${state.status}`}>
          <p className="display m-0 text-[2rem]">{state.status === "offline" ? "Offline" : "Could not read"}</p>
          <p className="mt-2 mb-0 max-w-[48ch] text-[15px] text-a10/85">{state.message}</p>
          {state.kind !== "not_image" && state.kind !== "too_large" && (
            <button type="button" className="btn-solid mt-4" onClick={onRetry}>
              Try again
            </button>
          )}
        </div>
      )}
    </div>
  );
}
