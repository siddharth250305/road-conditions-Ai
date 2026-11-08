import { useCallback, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";

// Three screens addressed by the URL hash, so the back button and shared links work without a
// router dependency: #/ (home), #/report, #/study, #/review (reviewers).
export type Route = "home" | "report" | "study" | "review";

const ROUTES: Record<string, Route> = { "": "home", "#": "home", "#/": "home", "#/report": "report", "#/study": "study", "#/review": "review" };

// The screens run left to right like lanes; a route change slides the way you are travelling.
const ORDER: Record<Route, number> = { home: 0, report: 1, study: 2, review: 3 };

export function routeOf(hash: string): Route {
  return ROUTES[hash] ?? "home";
}

export function hrefOf(route: Route): string {
  return route === "home" ? "#/" : `#/${route}`;
}

type TransitionDocument = Document & { startViewTransition?: (update: () => void) => unknown };

export function useRoute(): [Route, (next: Route) => void] {
  const [route, setRoute] = useState<Route>(() => routeOf(window.location.hash));
  const current = useRef(route);

  useEffect(() => {
    const onHash = () => {
      const next = routeOf(window.location.hash);
      if (next === current.current) return;
      const forward = ORDER[next] > ORDER[current.current];
      current.current = next;
      const apply = () => {
        flushSync(() => setRoute(next));
        window.scrollTo({ top: 0 });
      };
      const doc = document as TransitionDocument;
      if (typeof doc.startViewTransition === "function") {
        document.documentElement.dataset.nav = forward ? "forward" : "back";
        doc.startViewTransition(apply);
      } else {
        apply();
      }
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const go = useCallback((next: Route) => {
    if (routeOf(window.location.hash) !== next) window.location.hash = hrefOf(next);
  }, []);
  return [route, go];
}
