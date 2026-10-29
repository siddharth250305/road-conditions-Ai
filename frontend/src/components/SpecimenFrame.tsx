import { useState, type DragEvent } from "react";
import { prettyClass } from "../classes";
import { percent } from "../format";
import { previewOf, type ViewState } from "../state";
import { CropMarks } from "./CropMarks";

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
    <div className="relative mx-6 my-6">
      <CropMarks />
      <div
        className="relative aspect-[3/2] w-full overflow-hidden border-2 border-rule bg-surface"
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
            className={`absolute inset-0 flex w-full cursor-pointer flex-col items-start justify-end gap-2 p-5 text-left ${
              dragging ? "bg-lane text-ink" : "bg-transparent text-text"
            }`}
          >
            <span className="caption">Specimen 01 · empty</span>
            <span className="display text-[clamp(3.5rem,9vw,6rem)]">Drop a road photo</span>
            <span className="data">or choose a file · JPEG, PNG or WebP · up to 10 MB</span>
          </button>
        ) : (
          <>
            {/* The picture is stretched to 3:2, the shape the model sees, so the heatmap lines up. */}
            <img
              src={preview}
              alt={altText(state)}
              className="absolute inset-0 h-full w-full object-fill"
            />
            {overlay && (
              <img
                data-testid="heatmap-layer"
                src={overlay}
                alt=""
                aria-hidden="true"
                className="fade-layer absolute inset-0 h-full w-full object-fill"
                style={{ opacity: showHeatmap ? 1 : 0 }}
              />
            )}
            {state.status === "loading" && (
              <div className="absolute inset-0 overflow-hidden" aria-hidden="true">
                <div className="scan-bar absolute inset-x-0 h-1/3 bg-lane/60" />
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
