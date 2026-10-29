import { useRef, useState } from "react";
import { usePrediction } from "../hooks";
import type { Health } from "../types";
import { bytes } from "../format";
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
    <div className="grid gap-10 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
      <section aria-label="Specimen">
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
        {!online && (
          <p role="alert" className="data mb-3 border-2 border-rule p-2">
            You are offline. Photos cannot be read until the connection is back.
          </p>
        )}
        {online && reachable === false && state.status === "idle" && (
          <p role="alert" className="data mb-3 border-2 border-rule p-2">
            The service cannot be reached right now.
          </p>
        )}
        {noModel && (
          <p role="alert" className="data mb-3 border-2 border-rule p-2">
            The service is up but no model is loaded, so photos cannot be read yet.
          </p>
        )}

        <SpecimenFrame state={state} showHeatmap={showHeatmap} onPick={pick} onDropFile={take} />

        <div className="flex flex-wrap items-center gap-3">
          {hasHeatmap && (
            <button
              type="button"
              className="btn-quiet"
              aria-pressed={showHeatmap}
              onClick={() => setShowHeatmap((v) => !v)}
            >
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
            <span className="data text-muted">
              {state.name}
              {fileSize !== null && ` · ${bytes(fileSize)}`}
            </span>
          )}
        </div>
        {hasHeatmap && (
          <p className="data mt-2 max-w-prose text-muted">
            The heatmap shows where the model looked, not proof that it is right.
          </p>
        )}

        {state.status === "idle" && (
          <SampleStrip onPick={(sample) => {
              setFileSize(null);
              submitSample(sample.url, sample.caption);
            }} />
        )}
      </section>

      <section aria-label="Reading">
        <p className="caption mb-2">Friction ruler · dry to ice</p>
        <FrictionRuler state={state} />
        <VerdictPanel state={state} onRetry={retry} />
      </section>

      <div role="status" aria-live="polite" className="sr-only">
        {liveMessage(state)}
      </div>
    </div>
  );
}
