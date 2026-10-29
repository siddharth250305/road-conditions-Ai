import { useState } from "react";
import { LimitsFooter } from "./components/LimitsFooter";
import { ReportScreen } from "./components/ReportScreen";
import { StudyScreen } from "./components/StudyScreen";
import { useHealth, useOnline, usePrediction } from "./hooks";
import { isMock, MOCK_BANNER } from "./limits";

type Tab = "report" | "study";

export default function App() {
  const [tab, setTab] = useState<Tab>("report");
  const prediction = usePrediction();
  const { health, reachable } = useHealth();
  const online = useOnline();
  const { state } = prediction;
  const result = state.status === "result" || state.status === "unsure" ? state.result : null;
  const mock = isMock(health, result);

  return (
    <div className="flex min-h-screen flex-col bg-surface text-text">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:bg-lane focus:p-2 focus:text-ink"
      >
        Skip to the survey sheet
      </a>
      {mock && (
        <p
          role="note"
          data-testid="mock-banner"
          className="data border-b-2 border-rule bg-lane px-4 py-1 text-ink sm:px-6"
        >
          {MOCK_BANNER}
        </p>
      )}
      <header className="mx-auto w-full max-w-6xl px-4 pt-8 sm:px-6">
        <p className="caption">Road survey sheet · form 01 · specimen / friction / verdict</p>
        <h1 className="display mt-2 text-[clamp(3.5rem,9vw,6rem)]">Road condition</h1>
        <nav aria-label="Sections" className="mt-5 flex gap-3 border-b-2 border-rule pb-3">
          {(["report", "study"] as const).map((t) => (
            <button
              key={t}
              type="button"
              className="btn-quiet"
              aria-pressed={tab === t}
              aria-current={tab === t ? "page" : undefined}
              onClick={() => setTab(t)}
            >
              {t === "report" ? "Report" : "Study"}
            </button>
          ))}
        </nav>
      </header>

      <main id="main" className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:px-6">
        <div hidden={tab !== "report"}>
          <ReportScreen prediction={prediction} health={health} reachable={reachable} online={online} />
        </div>
        <div hidden={tab !== "study"}>
          <StudyScreen active={tab === "study"} />
        </div>
      </main>

      <LimitsFooter health={health} result={result} mock={mock} />
    </div>
  );
}
