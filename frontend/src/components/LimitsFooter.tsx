import { LIMITS_LINE, MOCK_BANNER, NOT_SEEN } from "../limits";
import { map, percent } from "../format";
import type { Health, PredictResponse } from "../types";

interface Props {
  health: Health | null;
  result: PredictResponse | null;
  mock: boolean;
  open: boolean;
  onToggle: () => void;
  /** On the home screen the footer floats as glass over the 3D scene. */
  overlay: boolean;
}

export function LimitsFooter({ health, result, mock, open, onToggle, overlay }: Props) {
  const cal = health?.calibration ?? null;
  const training = health?.training ?? null;
  const sha = health?.provenance?.checkpoint_sha256;
  const temperature = result?.temperature ?? cal?.temperature;
  const threshold = result?.uncertainty_threshold ?? health?.uncertainty_threshold;

  return (
    <footer
      className={
        overlay
          ? "relative z-20 px-3 pb-3 sm:px-5 sm:pb-4 lg:absolute lg:inset-x-0 lg:bottom-auto lg:top-[calc(100svh-5.25rem)]"
          : "border-t border-a4 bg-a0/60 px-3 py-3 sm:px-5"
      }
    >
      <div className={`mx-auto max-w-[1400px] ${overlay ? "glass rounded-[var(--r-panel)]" : ""}`}>
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 px-3 py-2.5 sm:px-4">
          <p data-testid="limits-line" className="m-0 min-w-[16rem] flex-1 text-[13px] leading-snug text-muted">
            <strong className="font-semibold text-a10">Limits:</strong> {mock ? `${MOCK_BANNER} ` : ""}
            {LIMITS_LINE}
          </p>
          <button type="button" className="btn-quiet" aria-expanded={open} aria-controls="limits" onClick={onToggle}>
            {open ? "Hide limits" : "Full limits"}
          </button>
        </div>

        <div
          id="limits"
          tabIndex={-1}
          hidden={!open}
          className={`border-t border-a10/12 px-3 py-6 outline-none sm:px-4 ${overlay ? "lg:max-h-[52vh] lg:overflow-y-auto" : ""}`}
        >
          {result && <p className="mt-0 mb-6 max-w-[70ch] text-[15px] leading-relaxed">{result.limits_note}</p>}
          <dl className="m-0 grid gap-x-12 gap-y-6 text-[15px] leading-relaxed md:grid-cols-2">
            <div>
              <dt className="heading text-muted">Training data</dt>
              <dd className="m-0 mt-1.5 max-w-[60ch]">
                RSCD road-surface images, six friction classes (dry, wet, water, fresh snow, melted
                snow, ice). The model is a classifier of road surface patches, not of whole scenes.
              </dd>
            </div>
            <div>
              <dt className="heading text-muted">Country</dt>
              <dd className="m-0 mt-1.5 max-w-[60ch]">
                The training images come from China. The cross-country study is a separate experiment
                on road damage detection, not on this classifier.
              </dd>
            </div>
            <div>
              <dt className="heading text-muted">What the model has not seen</dt>
              <dd className="m-0 mt-1.5 max-w-[60ch]">
                <ul className="m-0 list-disc pl-5 marker:text-a8">
                  {NOT_SEEN.map((line) => (
                    <li key={line}>{line}</li>
                  ))}
                </ul>
              </dd>
            </div>
            <div>
              <dt className="heading text-muted">Calibration</dt>
              <dd className="m-0 mt-1.5 max-w-[60ch]">
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
              <dt className="heading text-muted">Provenance</dt>
              <dd className="data m-0 mt-1.5 text-a9">
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
      </div>
    </footer>
  );
}
