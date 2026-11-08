import { useCallback, useState } from "react";
import { GlassNav } from "./components/GlassNav";
import { HomeScreen, scrollToHowItWorks } from "./components/HomeScreen";
import { LimitsFooter } from "./components/LimitsFooter";
import { ReportScreen } from "./components/ReportScreen";
import { ReviewScreen } from "./components/ReviewScreen";
import { StudyScreen } from "./components/StudyScreen";
import { useHealth, useOnline, usePrediction } from "./hooks";
import { isMock, MOCK_BANNER } from "./limits";
import { useRoute } from "./route";

export default function App() {
  const [route, go] = useRoute();
  const [limitsOpen, setLimitsOpen] = useState(false);
  const prediction = usePrediction();
  const { health, reachable } = useHealth();
  const online = useOnline();
  const { state, submitFile } = prediction;
  const result = state.status === "result" || state.status === "unsure" ? state.result : null;
  const mock = isMock(health, result);
  const home = route === "home";

  const openLimits = useCallback(() => {
    setLimitsOpen(true);
    requestAnimationFrame(() => {
      const panel = document.getElementById("limits");
      panel?.scrollIntoView?.({ behavior: "smooth", block: "end" });
      panel?.focus({ preventScroll: true });
    });
  }, []);
  const toggleLimits = useCallback(() => {
    if (limitsOpen) setLimitsOpen(false);
    else openLimits();
  }, [limitsOpen, openLimits]);
  const openReport = useCallback(() => go("report"), [go]);
  // How it works lives on the home screen: go there first when another screen is open.
  const howItWorks = useCallback(() => {
    if (route === "home") scrollToHowItWorks();
    else {
      go("home");
      window.setTimeout(scrollToHowItWorks, 450);
    }
  }, [route, go]);

  return (
    <div className={`flex min-h-screen w-full flex-col text-text ${home ? "bg-a0" : "asphalt"}`}>
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-50 focus:rounded-lg focus:bg-a10 focus:px-3 focus:py-2 focus:text-a0"
      >
        Skip to content
      </a>
      <header className={`site-header site-bar w-full ${home ? "site-bar-overlay absolute inset-x-0 top-0 z-30" : "sticky top-0 z-30"}`}>
        {mock && (
          <p
            role="note"
            data-testid="mock-banner"
            className="m-0 bg-works px-4 py-1.5 text-center text-[13px] font-semibold tracking-[0.01em] text-a0"
          >
            {MOCK_BANNER}
          </p>
        )}
        <GlassNav
          route={route}
          limitsOpen={limitsOpen}
          onLimits={toggleLimits}
          onHowItWorks={howItWorks}
          showReview={health?.routing?.enabled === true || route === "review"}
        />
      </header>

      <main id="main" tabIndex={-1} className="w-full flex-1 outline-none">
        {home && <HomeScreen onPhoto={submitFile} onOpenReport={openReport} onLimits={openLimits} />}
        <div hidden={route !== "report"} className="mx-auto w-full max-w-[1400px] px-4 pt-8 pb-14 sm:px-6 lg:px-10">
          <ReportScreen prediction={prediction} health={health} reachable={reachable} online={online} />
        </div>
        <div hidden={route !== "study"} className="mx-auto w-full max-w-[1400px] px-4 pt-8 pb-14 sm:px-6 lg:px-10">
          <StudyScreen active={route === "study"} />
        </div>
        <div hidden={route !== "review"} className="mx-auto w-full max-w-[1400px] px-4 pt-8 pb-14 sm:px-6 lg:px-10">
          <ReviewScreen active={route === "review"} health={health} />
        </div>
      </main>

      <LimitsFooter
        health={health}
        result={result}
        mock={mock}
        open={limitsOpen}
        onToggle={toggleLimits}
        overlay={false}
      />
    </div>
  );
}
