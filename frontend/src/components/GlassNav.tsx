import { hrefOf, type Route } from "../route";
import { REPO_URL } from "./HomeScreen";
import { GitHubIcon, MarkIcon } from "./icons";

interface Props {
  route: Route;
  limitsOpen: boolean;
  onLimits: () => void;
  onHowItWorks: () => void;
}

// The site bar. Routes are text over a lane marking that glides to the current page; How it
// works scrolls to its section and Model opens the limits and provenance panel.
export function GlassNav({ route, limitsOpen, onLimits, onHowItWorks }: Props) {
  const link = (r: Route, name: string) => (
    <a href={hrefOf(r)} className="nav-link" aria-current={route === r ? "page" : undefined}>
      {name}
      <span className="lane-marker" aria-hidden="true" />
    </a>
  );
  return (
    <div className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-x-6 px-5 sm:px-10">
      <a href={hrefOf("home")} className="flex min-h-[64px] items-center gap-2.5 no-underline">
        <MarkIcon />
        <span className="text-[20px] font-semibold tracking-[-0.01em] max-[479px]:sr-only">RoadCondition AI</span>
      </a>
      <nav aria-label="Sections" className="order-last -mx-2 flex w-[calc(100%+1rem)] overflow-x-auto md:order-none md:mx-0 md:w-auto">
        {link("home", "Home")}
        {link("report", "Demo")}
        <button type="button" className="nav-link" onClick={onHowItWorks}>
          How it works
          <span className="lane-marker" aria-hidden="true" />
        </button>
        {link("study", "Study")}
        <button type="button" className="nav-link" aria-expanded={limitsOpen} aria-controls="limits" onClick={onLimits}>
          Model
          <span className="lane-marker" aria-hidden="true" />
        </button>
      </nav>
      <div className="flex items-center gap-2">
        <a href={REPO_URL} target="_blank" rel="noreferrer" className="grid h-11 w-11 place-items-center rounded-full text-a10/85 hover:text-a10">
          <GitHubIcon width={22} height={22} />
          <span className="sr-only">RoadCondition AI on GitHub</span>
        </a>
        <a href={hrefOf("report")} className="cta cta-small">
          Get started
        </a>
      </div>
    </div>
  );
}
