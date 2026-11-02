import { useState, type DragEvent } from "react";
import { prettyClass } from "../classes";
import { percent } from "../format";
import { previewOf, type ViewState } from "../state";
import { CropMarks } from "./CropMarks";
import { UploadIcon } from "./icons";

interface Props {
  state: ViewState;
  showHeatmap: boolean;
  onPick: () => void;
  onDropFile: (file: File) => void;
}

export function altText(state: ViewState): string {
  switch (state.status) {
    case "result":
      return `Road photo ${state.name}. Predicted ${prettyClass(state.result.prediction)}, confidence ${percent(state.result.confidence)}.`;
    case "unsure": {
      const [a, b] = state.result.top2;
      return `Road photo ${state.name}. Not sure. Closest: ${a ? prettyClass(a.class) : "none"} ${a ? percent(a.probability) : ""}, then ${b ? prettyClass(b.class) : "none"} ${b ? percent(b.probability) : ""}.`;
    }
    case "loading":
      return `Road photo ${state.name}, being read.`;
    case "error":
    case "offline":
      return `Road photo ${state.name}. It could not be read.`;
    default:
      return "";
  }
}

export function SpecimenFrame({ state, showHeatmap, onPick, onDropFile }: Props) {
  const [dragging, setDragging] = useState(false);
  const preview = previewOf(state);
  const result = state.status === "result" || state.status === "unsure" ? state.result : null;
  const overlay =
    result?.heatmap_base64 != null
      ? `data:${result.heatmap_media_type};base64,${result.heatmap_base64}`
      : null;

  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) onDropFile(file);
  };

  return (
    <div className="relative m-3">
      <CropMarks />
      <div
        className="relative aspect-[3/2] w-full overflow-hidden rounded-[var(--r-panel)] border border-a10/14 bg-a2"
        aria-busy={state.status === "loading"}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        {preview === null ? (
          <button
            type="button"
            onClick={onPick}
            className={`group absolute inset-3 flex cursor-pointer flex-col items-center justify-center gap-4 rounded-[8px] border-2 p-6 text-center transition-colors ${
              dragging ? "border-solid border-a10 bg-a10/10" : "border-dashed border-a10/30 hover:border-a10/60 hover:bg-a10/5"
            }`}
          >
            <span className="grid h-14 w-14 place-items-center rounded-full border border-a10/25 bg-a10/6 text-a10" aria-hidden="true">
              <UploadIcon width={26} height={26} />
            </span>
            <span className="display text-[clamp(1.75rem,3.2vw,2.5rem)]">Drop a road photo</span>
            <span className="text-[14px] text-muted">
              or <span className="font-semibold text-a10 underline underline-offset-4">choose a file</span>. JPEG, PNG or WebP, up to 10 MB.
            </span>
          </button>
        ) : (
          <>
            {/* The picture is stretched to 3:2, the shape the model sees, so the heatmap lines up. */}
            <img src={preview} alt={altText(state)} className="absolute inset-0 h-full w-full object-fill" />
            {overlay && (
              <img
                data-testid="heatmap-layer"
                src={overlay}
                alt=""
                aria-hidden="true"
                className="fade-layer absolute inset-0 h-full w-full object-fill mix-blend-screen"
                style={{ opacity: showHeatmap ? 1 : 0 }}
              />
            )}
            {state.status === "loading" && (
              <div className="absolute inset-0 overflow-hidden bg-a0/30" aria-hidden="true">
                <div className="scan-bar absolute inset-x-0 h-1/3 bg-gradient-to-b from-transparent via-a10/25 to-transparent" />
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
