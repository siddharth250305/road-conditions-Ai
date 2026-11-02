import { useStudy } from "../hooks";
import { map, percent } from "../format";
import { MOCK_BANNER } from "../limits";
import type { Study } from "../types";
import { LearningCurve } from "./LearningCurve";
import { SlopeChart } from "./SlopeChart";

function NumbersTable({ study }: { study: Study }) {
  const budgets = [0, 10, 50, 200];
  return (
    <table className="w-full max-w-[52rem] border-collapse text-left text-[14px] tabular-nums">
      <caption className="heading mb-3 text-left text-muted">The numbers behind both charts (mAP@0.5)</caption>
      <thead>
        <tr className="border-b border-a10/25">
          <th scope="col" className="py-2 pr-4 font-semibold">Country</th>
          <th scope="col" className="py-2 pr-4 font-semibold">Source</th>
          <th scope="col" className="py-2 pr-4 font-semibold">Gap</th>
          {budgets.map((b) => (
            <th key={b} scope="col" className="py-2 pr-4 font-semibold whitespace-nowrap">{b} labels</th>
          ))}
        </tr>
      </thead>
      <tbody className="data">
        {study.slope.map((row) => {
          const curve = study.learning_curve[row.country] ?? [];
          return (
            <tr key={row.country} className="border-b border-a10/10">
              <th scope="row" className="py-2 pr-4 font-sans text-[14px] font-semibold tracking-normal capitalize">{row.country.replace(/_/g, " ")}</th>
              <td className="py-2 pr-4">{map(row.source_map50)}</td>
              <td className="py-2 pr-4">
                {row.relative_gap_map50 == null ? "n/a" : percent(row.relative_gap_map50)}
              </td>
              {budgets.map((b) => {
                const cell = curve.find((p) => p.labels === b)?.map50;
                return (
                  <td key={b} className="py-2 pr-4 whitespace-nowrap">
                    {cell?.mean == null ? "–" : map(cell.mean)}
                    {cell?.std != null && <span className="text-muted">{` ± ${cell.std.toFixed(3)}`}</span>}
                  </td>
                );
              })}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/** The two charts' frames, drawn empty and marked pending, so a missing study still shows its shape. */
function PendingFrames() {
  const frames = [
    {
      title: "Detection accuracy at home and abroad",
      body: "mAP@0.5 of a detector trained on Japan, scored on Japan and on Norway, the Czech Republic and the USA.",
    },
    {
      title: "Accuracy as target labels are added",
      body: "The same detector given 0, 10, 50 and 200 labelled images from each target country.",
    },
  ];
  return (
    <div className="mt-6 grid gap-6 xl:grid-cols-2">
      {frames.map((f) => (
        <div key={f.title} className="rounded-[var(--r-panel)] border-2 border-dashed border-a10/15 p-6 sm:p-8">
          <h2 className="m-0 text-[17px] font-semibold">{f.title}</h2>
          <p className="mt-2 mb-6 max-w-[52ch] text-[14px] leading-relaxed text-muted">{f.body}</p>
          <p className="heading m-0 text-a8">Pending</p>
        </div>
      ))}
    </div>
  );
}

export function StudyScreen({ active }: { active: boolean }) {
  const state = useStudy(active);
  return (
    <div className="px-3">
      <div className="flex flex-wrap items-end justify-between gap-x-10 gap-y-3">
        <h1 className="display m-0 text-[clamp(2rem,3.4vw,3rem)]">Cross-country study</h1>
        <p className="m-0 max-w-[58ch] text-[15px] leading-relaxed text-muted">
          A road-damage detector trained on one country and scored on others, then given a few
          labelled images from the target country. These are measured numbers from this project's own
          held-out splits, not from the original RDD2022 competition.
        </p>
      </div>

      {state.status === "loading" && <p className="mt-8 text-[15px] text-muted">Loading the study…</p>}
      {state.status === "error" && (
        <>
          <div role="alert" className="wash mt-8 flex max-w-[60ch] flex-wrap items-center justify-between gap-4 p-5">
            <p className="m-0 text-[15px] leading-relaxed">
              The study results could not be loaded from the service. Check that it is running, then
              try again.
            </p>
            <button type="button" className="btn-quiet" onClick={() => window.location.reload()}>
              Try again
            </button>
          </div>
          <PendingFrames />
        </>
      )}
      {state.status === "ok" && state.result.kind === "absent" && (
        <p data-testid="study-absent" className="wash mt-8 max-w-[60ch] p-5 text-[15px] leading-relaxed">
          No study results have been added to this deployment yet. The page shows measured numbers
          only, so there is nothing to draw until the detection experiments have been run and
          exported.
        </p>
      )}
      {state.status === "ok" && state.result.kind === "absent" && <PendingFrames />}
      {state.status === "ok" && state.result.kind === "ok" && (
        <>
          {state.result.study.mock && (
            <p role="note" className="mt-8 mb-0 inline-flex rounded-[var(--r-plate)] bg-works px-4 py-1.5 text-[13px] font-semibold text-a0">
              {MOCK_BANNER} These charts show synthetic numbers.
            </p>
          )}
          <div className="mt-8 grid gap-6 xl:grid-cols-2">
            <div className="rounded-[var(--r-panel)] border border-a10/10 bg-a2/80 p-5 sm:p-7">
              <SlopeChart study={state.result.study} />
            </div>
            <div className="rounded-[var(--r-panel)] border border-a10/10 bg-a2/80 p-5 sm:p-7">
              <LearningCurve study={state.result.study} />
            </div>
          </div>
          <div className="mt-10 overflow-x-auto">
            <NumbersTable study={state.result.study} />
          </div>
          <ul className="mt-8 max-w-[70ch] list-disc pl-5 text-[14px] leading-relaxed text-muted marker:text-a8">
            {state.result.study.notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
