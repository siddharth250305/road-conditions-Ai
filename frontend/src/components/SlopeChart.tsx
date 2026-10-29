import { scaleLinear } from "d3-scale";
import { line } from "d3-shape";
import { axisMax, spreadLabels } from "../geometry";
import { map, percent } from "../format";
import type { Study } from "../types";

const W = 700;
const H = 400;
const M = { top: 56, right: 0, bottom: 36, left: 64 };
const XS = 150; // source axis
const XT = 390; // target axis
const DASHES = ["", "7 4", "2 4", "10 3 2 3"];

export function SlopeChart({ study }: { study: Study }) {
  const rows = study.slope;
  const top = axisMax([study.source.map50, ...rows.map((r) => r.target_map50)]);
  const y = scaleLinear().domain([0, top]).range([H - M.bottom, M.top]);
  const ticks = y.ticks(5);
  const path = line<[number, number]>();
  const labelY = spreadLabels(
    rows.map((r) => y(r.target_map50)),
    18,
  );
  const sourceName = study.source_country.replace(/_/g, " ");

  return (
    <figure className="m-0">
      <svg
        role="img"
        aria-labelledby="slope-title slope-desc"
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full max-w-[40rem] text-text"
      >
        <title id="slope-title">Detection accuracy at home and abroad</title>
        <desc id="slope-desc">
          {`mAP at 0.5 for a detector trained on ${sourceName} (${map(study.source.map50)}): ` +
            rows.map((r) => `${r.country.replace(/_/g, " ")} ${map(r.target_map50)}`).join(", ")}
        </desc>

        <text x={XS} y={24} textAnchor="middle" className="caption" fill="currentColor">
          {sourceName} (source)
        </text>
        <text x={XT} y={24} textAnchor="middle" className="caption" fill="currentColor">
          Other countries
        </text>
        <text x={XS} y={42} textAnchor="middle" className="data" fill="currentColor">
          mAP@0.5
        </text>

        {ticks.map((t) => (
          <g key={t}>
            <line x1={M.left - 4} x2={M.left} y1={y(t)} y2={y(t)} stroke="currentColor" />
            <text x={M.left - 8} y={y(t) + 4} textAnchor="end" className="data" fill="currentColor">
              {t.toFixed(1)}
            </text>
          </g>
        ))}
        <line x1={M.left} x2={M.left} y1={y(0)} y2={y(top)} stroke="currentColor" />
        <line x1={XS} x2={XS} y1={y(0)} y2={y(top)} stroke="currentColor" strokeOpacity={0.35} />
        <line x1={XT} x2={XT} y1={y(0)} y2={y(top)} stroke="currentColor" strokeOpacity={0.35} />

        {rows.map((r, i) => (
          <g key={r.country} data-country={r.country}>
            <path
              d={path([
                [XS, y(r.source_map50)],
                [XT, y(r.target_map50)],
              ])!}
              fill="none"
              stroke="currentColor"
              strokeWidth={2}
              strokeDasharray={DASHES[i % DASHES.length]}
            />
            <circle cx={XT} cy={y(r.target_map50)} r={4.5} fill="currentColor" />
            <text
              x={XT + 14}
              y={labelY[i]! + 4}
              className="data"
              fill="currentColor"
            >
              {r.country.replace(/_/g, " ")} {map(r.target_map50)}
              {r.relative_gap_map50 != null && ` (${percent(r.relative_gap_map50)} gap)`}
            </text>
          </g>
        ))}
        <circle cx={XS} cy={y(study.source.map50)} r={5.5} fill="currentColor" />
        <text x={XS - 12} y={y(study.source.map50) + 4} textAnchor="end" className="data" fill="currentColor">
          {map(study.source.map50)}
        </text>
      </svg>
      <figcaption className="data mt-2 max-w-prose text-muted">
        Each line is one target country: the same detector, scored at home and on that country's
        own held-out images. A steeper fall is a bigger domain shift.
      </figcaption>
    </figure>
  );
}
