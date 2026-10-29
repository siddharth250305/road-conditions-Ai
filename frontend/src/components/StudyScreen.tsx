import { useStudy } from "../hooks";
import { map, percent } from "../format";
import { MOCK_BANNER } from "../limits";
import type { Study } from "../types";
import { LearningCurve } from "./LearningCurve";
import { SlopeChart } from "./SlopeChart";

function NumbersTable({ study }: { study: Study }) {
  const budgets = [0, 10, 50, 200];
  return (
    <table className="data w-full max-w-[40rem] border-collapse text-left">
      <caption className="caption mb-2 text-left">The numbers behind both charts (mAP@0.5)</caption>
      <thead>
        <tr className="border-b-2 border-rule">
          <th scope="col" className="py-1 pr-3 font-medium">Country</th>
          <th scope="col" className="py-1 pr-3 font-medium">Source</th>
          <th scope="col" className="py-1 pr-3 font-medium">Gap</th>
          {budgets.map((b) => (
            <th key={b} scope="col" className="py-1 pr-3 font-medium">{b} labels</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {study.slope.map((row) => {
          const curve = study.learning_curve[row.country] ?? [];
          return (
            <tr key={row.country} className="border-b border-rule">
              <th scope="row" className="py-1 pr-3 font-medium">{row.country.replace(/_/g, " ")}</th>
              <td className="py-1 pr-3">{map(row.source_map50)}</td>
              <td className="py-1 pr-3">
                {row.relative_gap_map50 == null ? "n/a" : percent(row.relative_gap_map50)}
              </td>
              {budgets.map((b) => {
                const cell = curve.find((p) => p.labels === b)?.map50;
                return (
                  <td key={b} className="py-1 pr-3 whitespace-nowrap">
                    {cell?.mean == null ? "–" : map(cell.mean)}
                    {cell?.std != null && ` ± ${cell.std.toFixed(3)}`}
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

export function StudyScreen({ active }: { active: boolean }) {
  const state = useStudy(active);
  return (
    <div>
      <h2 className="display text-[clamp(3.5rem,8vw,6rem)]">Cross-country study</h2>
      <p className="mt-3 max-w-prose">
        A road-damage detector trained on one country and scored on others, then given a few
        labelled images from the target country. These are measured numbers from this project's own
        held-out splits, not from the original RDD2022 competition.
      </p>

      {state.status === "loading" && <p className="data mt-6">Loading the study…</p>}
      {state.status === "error" && (
        <p role="alert" className="mt-6 max-w-prose border-2 border-rule p-3">
          {state.message}
        </p>
      )}
      {state.status === "ok" && state.result.kind === "absent" && (
        <p data-testid="study-absent" className="mt-6 max-w-prose border-2 border-rule p-3">
          No study results have been added to this deployment yet. The page shows measured numbers
          only, so there is nothing to draw until the detection experiments have been run and
          exported.
        </p>
      )}
      {state.status === "ok" && state.result.kind === "ok" && (
        <>
          {state.result.study.mock && (
            <p role="note" className="data mt-6 border-2 border-rule p-2">
              {MOCK_BANNER} These charts show synthetic numbers.
            </p>
          )}
          <div className="mt-8 grid gap-10 xl:grid-cols-2">
            <SlopeChart study={state.result.study} />
            <LearningCurve study={state.result.study} />
          </div>
          <div className="mt-8 overflow-x-auto">
            <NumbersTable study={state.result.study} />
          </div>
          <ul className="data mt-6 max-w-prose list-disc pl-5 text-muted">
            {state.result.study.notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
