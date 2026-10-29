import { scaleLinear, scalePoint } from "d3-scale";
import { line } from "d3-shape";
import { axisMax, spreadLabels } from "../geometry";
import { map } from "../format";
import type { Study } from "../types";

const W = 640;
const H = 380;
const M = { top: 24, right: 170, bottom: 56, left: 64 };
const BUDGETS = [0, 10, 50, 200];
const DASHES = ["", "7 4", "2 4", "10 3 2 3"];

export function LearningCurve({ study }: { study: Study }) {
  const countries = Object.keys(study.learning_curve);
  const all = countries.flatMap((c) =>
    study.learning_curve[c]!.flatMap((p) =>
      p.map50.mean === null ? [] : [p.map50.mean + (p.map50.std ?? 0)],
    ),
  );
  const top = axisMax([study.source.map50, ...all]);
  const x = scalePoint<number>().domain(BUDGETS).range([M.left, W - M.right]).padding(0.15);
  const y = scaleLinear().domain([0, top]).range([H - M.bottom, M.top]);
  const path = line<[number, number]>();
  const series = countries.map((country) => ({
    country,
    points: study.learning_curve[country]!.filter(
      (p) => p.map50.mean !== null && BUDGETS.includes(p.labels),
    ),
  }));
  // The source reference label and every country label share one column, so spread them together.
  const column = spreadLabels(
    [
      y(study.source.map50),
      ...series.map((s) => {
        const last = s.points[s.points.length - 1];
        return last ? y(last.map50.mean!) : 0;
      }),
    ],
    18,
  );
  const [sourceLabelY, ...endY] = column as [number, ...number[]];
  const sourceName = study.source_country.replace(/_/g, " ");

  return (
    <figure className="m-0">
      <svg
        role="img"
        aria-labelledby="curve-title curve-desc"
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full max-w-[40rem] text-text"
      >
        <title id="curve-title">Accuracy as target labels are added</title>
        <desc id="curve-desc">
          {`mAP at 0.5 on each country with 0, 10, 50 and 200 labelled target images. ` +
            series
              .map(
                (s) =>
                  `${s.country.replace(/_/g, " ")}: ` +
                  s.points.map((p) => `${p.labels} labels ${map(p.map50.mean)}`).join(", "),
              )
              .join(". ")}
        </desc>

        {y.ticks(5).map((t) => (
          <g key={t}>
            <line x1={M.left - 4} x2={W - M.right} y1={y(t)} y2={y(t)} stroke="currentColor" strokeOpacity={t === 0 ? 1 : 0.15} />
            <text x={M.left - 8} y={y(t) + 4} textAnchor="end" className="data" fill="currentColor">
              {t.toFixed(1)}
            </text>
          </g>
        ))}
        {BUDGETS.map((b) => (
          <g key={b}>
            <line x1={x(b)} x2={x(b)} y1={y(0)} y2={y(0) + 5} stroke="currentColor" />
            <text x={x(b)} y={y(0) + 20} textAnchor="middle" className="data" fill="currentColor">
              {b}
            </text>
          </g>
        ))}
        <text x={(M.left + W - M.right) / 2} y={H - 8} textAnchor="middle" className="caption" fill="currentColor">
          Labelled target images
        </text>
        <text x={14} y={M.top - 8} className="data" fill="currentColor">
          mAP@0.5
        </text>

        <line
          x1={M.left}
          x2={W - M.right}
          y1={y(study.source.map50)}
          y2={y(study.source.map50)}
          stroke="currentColor"
          strokeDasharray="1 5"
          strokeWidth={2}
        />
        <text x={W - M.right + 8} y={sourceLabelY + 4} className="data" fill="currentColor">
          {sourceName} {map(study.source.map50)}
        </text>

        {series.map((s, i) => (
          <g key={s.country} data-country={s.country}>
            <path
              d={path(s.points.map((p) => [x(p.labels)!, y(p.map50.mean!)]))!}
              fill="none"
              stroke="currentColor"
              strokeWidth={2}
              strokeDasharray={DASHES[i % DASHES.length]}
            />
            {s.points.map((p) => (
              <g key={p.labels}>
                {p.map50.std != null && (
                  <line
                    data-testid="error-bar"
                    x1={x(p.labels)}
                    x2={x(p.labels)}
                    y1={y(p.map50.mean! - p.map50.std)}
                    y2={y(p.map50.mean! + p.map50.std)}
                    stroke="currentColor"
                    strokeWidth={1.5}
                  />
                )}
                <circle cx={x(p.labels)} cy={y(p.map50.mean!)} r={4} fill="currentColor" />
              </g>
            ))}
            {s.points.length > 0 && (
              <text x={W - M.right + 8} y={endY[i]! + 4} className="data" fill="currentColor">
                {s.country.replace(/_/g, " ")}
              </text>
            )}
          </g>
        ))}
      </svg>
      <figcaption className="data mt-2 max-w-prose text-muted">
        Mean over seeds with ± one standard deviation for 10, 50 and 200 labels. The 0-label point is
        the source detector with no target images, and is a single run.
      </figcaption>
    </figure>
  );
}
