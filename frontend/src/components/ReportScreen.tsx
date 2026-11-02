import { useRef, useState } from "react";
import { usePrediction } from "../hooks";
import type { Health } from "../types";
import { bytes } from "../format";
import { LIMITS_LINE } from "../limits";
import { FrictionRuler } from "./FrictionRuler";
import { SampleStrip } from "./SampleStrip";
import { SpecimenFrame } from "./SpecimenFrame";
import { liveMessage, VerdictPanel } from "./VerdictPanel";

interface Props {
  prediction: ReturnType<typeof usePrediction>;
  health: Health | null;
  reachable: boolean | null;
  online: boolean;
}

function Notice({ children }: { children: string }) {
  return (
    <p role="alert" className="wash mx-3 mt-0 mb-4 flex items-center gap-3 px-4 py-3 text-[15px]">
      <span className="h-2 w-2 shrink-0 rounded-full bg-a10" aria-hidden="true" />
      {children}
    </p>
  );
}

export function ReportScreen({ prediction, health, reachable, online }: Props) {
  const { state, submitFile, submitSample, retry, reset } = prediction;
  const [showHeatmap, setShowHeatmap] = useState(true);
  const input = useRef<HTMLInputElement>(null);
  const [fileSize, setFileSize] = useState<number | null>(null);

  const pick = () => input.current?.click();
  const take = (file: File) => {
    setFileSize(file.size);
    submitFile(file);
  };

  const hasHeatmap =
    (state.status === "result" || state.status === "unsure") && state.result.heatmap_base64 != null;
  const noModel = health?.status === "unavailable";

  return (
    <>
      <div className="mb-6 flex flex-wrap items-end justify-between gap-x-10 gap-y-3 px-3">
        <h1 className="display m-0 text-[clamp(2rem,3.4vw,3rem)]">Analyse a road photo</h1>
        <p className="m-0 max-w-[46ch] text-[15px] text-muted">
          One photo of the road surface in, one of six friction classes out, or an honest “not sure”.
        </p>
      </div>
      <div className="grid gap-8 lg:grid-cols-[minmax(0,1.55fr)_minmax(0,1fr)] lg:gap-10">
        <section aria-label="Photo">
          <input
            ref={input}
            type="file"
            accept="image/*"
            className="sr-only"
            tabIndex={-1}
            aria-hidden="true"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) take(file);
              e.target.value = "";
            }}
          />
          {!online && <Notice>You are offline. Photos cannot be read until the connection is back.</Notice>}
          {online && reachable === false && state.status === "idle" && (
            <Notice>The service cannot be reached right now.</Notice>
          )}
          {noModel && <Notice>The service is up but no model is loaded, so photos cannot be read yet.</Notice>}

          <SpecimenFrame state={state} showHeatmap={showHeatmap} onPick={pick} onDropFile={take} />

          <div className="mt-3 flex flex-wrap items-center gap-2 px-3">
            {hasHeatmap && (
              <button type="button" className="btn-quiet" aria-pressed={showHeatmap} onClick={() => setShowHeatmap((v) => !v)}>
                Heatmap {showHeatmap ? "on" : "off"}
              </button>
            )}
            {state.status !== "idle" && (
              <button type="button" className="btn-quiet" onClick={pick}>
                Another photo
              </button>
            )}
            {state.status !== "idle" && (
              <button type="button" className="btn-quiet" onClick={reset}>
                Clear
              </button>
            )}
            {state.status !== "idle" && (
              <span className="data ml-1 text-muted">
                {state.name}
                {fileSize !== null && ` · ${bytes(fileSize)}`}
              </span>
            )}
          </div>
          {hasHeatmap && (
            <p className="mt-3 max-w-[60ch] px-3 text-[14px] text-muted">
              The heatmap shows where the model looked, not proof that it is right.
            </p>
          )}

          {state.status === "idle" && (
            <SampleStrip
              onPick={(sample) => {
                setFileSize(null);
                submitSample(sample.url, sample.caption);
              }}
            />
          )}
        </section>

        <section aria-label="Reading" className="rounded-[var(--r-panel)] border border-a10/10 bg-a2/80 p-5 sm:p-7 lg:self-start">
          <VerdictPanel state={state} onRetry={retry} />
          <h2 className="heading mt-8 mb-3 text-muted">Friction ruler, dry to ice</h2>
          <FrictionRuler state={state} />
          <p className="mt-6 mb-0 border-t border-a10/10 pt-4 text-[13px] leading-snug text-muted">
            <strong className="font-semibold text-a10">Limits:</strong> {LIMITS_LINE}
          </p>
        </section>

        <div role="status" aria-live="polite" className="sr-only">
          {liveMessage(state)}
        </div>
      </div>
    </>
  );
}
