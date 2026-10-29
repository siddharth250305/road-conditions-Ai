import { useId, useState } from "react";
import { LIMITS_LINE, MOCK_BANNER, NOT_SEEN } from "../limits";
import { map, percent } from "../format";
import type { Health, PredictResponse } from "../types";

interface Props {
  health: Health | null;
  result: PredictResponse | null;
  mock: boolean;
}

export function LimitsFooter({ health, result, mock }: Props) {
  const [open, setOpen] = useState(false);
  const panel = useId();
  const cal = health?.calibration ?? null;
  const training = health?.training ?? null;
  const sha = health?.provenance?.checkpoint_sha256;
  const temperature = result?.temperature ?? cal?.temperature;
  const threshold = result?.uncertainty_threshold ?? health?.uncertainty_threshold;

  return (
    <footer className="border-t-2 border-rule">
      <div className="mx-auto flex max-w-6xl flex-wrap items-start gap-x-6 gap-y-2 px-4 py-3 sm:px-6">
        <p data-testid="limits-line" className="data min-w-[16rem] flex-1">
          <strong className="font-medium">LIMITS</strong> · {mock ? `${MOCK_BANNER} ` : ""}
          {LIMITS_LINE}
        </p>
        <button
          type="button"
          className="btn-quiet"
          aria-expanded={open}
          aria-controls={panel}
          onClick={() => setOpen((v) => !v)}
        >
          {open ? "Hide limits" : "Full limits"}
        </button>
      </div>

      <div
        id={panel}
        hidden={!open}
        className="mx-auto max-w-6xl border-t border-rule px-4 py-6 sm:px-6"
      >
        {result && <p className="mb-6 max-w-prose">{result.limits_note}</p>}
        <dl className="grid gap-x-10 gap-y-5 md:grid-cols-2">
          <div>
            <dt className="caption">Training data</dt>
            <dd className="m-0 mt-1">
              RSCD road-surface images, six friction classes (dry, wet, water, fresh snow, melted
              snow, ice). The model is a classifier of road surface patches, not of whole scenes.
            </dd>
          </div>
          <div>
            <dt className="caption">Country</dt>
            <dd className="m-0 mt-1">
              The training images come from China. The cross-country study on the Study tab is a
              separate experiment on road damage detection, not on this classifier.
            </dd>
          </div>
          <div>
            <dt className="caption">What the model has not seen</dt>
            <dd className="m-0 mt-1">
              <ul className="m-0 list-disc pl-5">
                {NOT_SEEN.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </dd>
          </div>
          <div>
            <dt className="caption">Calibration</dt>
            <dd className="m-0 mt-1">
              Confidence is the model's probability after temperature scaling
              {temperature !== undefined && ` (T = ${temperature.toFixed(2)})`}, fitted on held-out
              RSCD validation images. It is not a guarantee on other roads.
              {threshold != null && ` Below ${percent(threshold)} the page says "Not sure".`}
              {cal?.ece_before != null && cal?.ece_after != null && (
                <>
                  {" "}
                  Expected calibration error {map(cal.ece_before)} before, {map(cal.ece_after)}{" "}
                  after.
                </>
              )}
            </dd>
          </div>
          <div className="md:col-span-2">
            <dt className="caption">Provenance</dt>
            <dd className="data m-0 mt-1">
              model {health?.model ?? "unknown"}
              {training?.seed != null && ` · seed ${training.seed}`}
              {training?.best_epoch != null && ` · best epoch ${training.best_epoch}`}
              {training?.val_macro_f1 != null && ` · validation macro-F1 ${map(training.val_macro_f1)}`}
              {training?.test_macro_f1 != null && ` · test macro-F1 ${map(training.test_macro_f1)}`}
              {sha && ` · checkpoint sha256 ${sha.slice(0, 12)}…`}
              {mock && " · built from mock data"}
            </dd>
          </div>
        </dl>
      </div>
    </footer>
  );
}
