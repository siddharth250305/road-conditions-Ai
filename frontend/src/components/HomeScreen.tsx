import { Component, lazy, Suspense, useEffect, useRef, useState, type DragEvent, type ReactNode } from "react";
import { prettyClass, type ClassName } from "../classes";
import { hrefOf } from "../route";
import { ChevronIcon, GitHubIcon, IceIcon, PlayIcon, SnowflakeIcon, SunIcon, WavesIcon } from "./icons";

const AerialScene = lazy(() => import("./scene/AerialScene"));

export const REPO_URL = "https://github.com/siddharth250305/road-conditions-Ai";

interface Props {
  onPhoto: (file: File) => void;
  onOpenReport: () => void;
  onLimits: () => void;
}

const CHIPS: { cls: ClassName; label: string; Icon: typeof SunIcon }[] = [
  { cls: "dry", label: "Dry", Icon: SunIcon },
  { cls: "wet", label: "Wet", Icon: WavesIcon },
  { cls: "fresh_snow", label: "Snow", Icon: SnowflakeIcon },
  { cls: "ice", label: "Ice", Icon: IceIcon },
];

function media(query: string): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function" && window.matchMedia(query).matches;
}

function useMedia(query: string): boolean {
  const [match, setMatch] = useState(() => media(query));
  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const mq = window.matchMedia(query);
    const on = () => setMatch(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, [query]);
  return match;
}

/** True when the browser can draw the 3D layer. jsdom (the test runner) cannot. */
function canDraw3D(): boolean {
  if (typeof navigator !== "undefined" && /jsdom/i.test(navigator.userAgent)) return false;
  try {
    const canvas = document.createElement("canvas");
    return Boolean(canvas.getContext("webgl2") ?? canvas.getContext("webgl"));
  } catch {
    return false;
  }
}

/** If the 3D layer fails, the photo stays and the page keeps working. */
class SceneBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? null : this.props.children;
  }
}

export function scrollToHowItWorks() {
  document.getElementById("how-it-works")?.scrollIntoView({ behavior: media("(prefers-reduced-motion: reduce)") ? "auto" : "smooth" });
}

export function HomeScreen({ onPhoto, onOpenReport, onLimits }: Props) {
  const reduced = useMedia("(prefers-reduced-motion: reduce)");
  const [webgl] = useState(canDraw3D);
  const [sceneReady, setSceneReady] = useState(false);
  const [hovered, setHovered] = useState<ClassName | null>(null);
  const [pinned, setPinned] = useState<ClassName | null>(null);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const preview = hovered ?? pinned;

  // A photo dropped anywhere on the hero is read straight away on the demo screen.
  const take = (file: File) => {
    onPhoto(file);
    onOpenReport();
  };
  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) take(file);
  };

  return (
    <>
      <section
        aria-labelledby="home-title"
        className="relative isolate flex min-h-[100svh] w-full flex-col overflow-hidden bg-[#050808]"
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={(e) => {
          if (e.currentTarget === e.target) setDragging(false);
        }}
        onDrop={onDrop}
      >
        <input
          ref={input}
          type="file"
          accept="image/*"
          className="sr-only"
          tabIndex={-1}
          aria-hidden="true"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) take(file);
            e.target.value = "";
          }}
        />

        {/* Background: the photograph, then the 3D road layer over it once it is drawn. */}
        <div className="absolute inset-0 -z-10">
          <img
            src="/home/aerial.jpg"
            alt=""
            aria-hidden="true"
            className={`h-full w-full object-cover object-[60%_48%] transition-opacity duration-700 ${webgl && sceneReady ? "opacity-0" : ""}`}
          />
          {webgl && (
            <SceneBoundary>
              <Suspense fallback={null}>
                <div className={`absolute inset-0 transition-opacity duration-700 ${sceneReady ? "opacity-100" : "opacity-0"}`}>
                  <AerialScene surface={preview} reducedMotion={reduced} onReady={() => setSceneReady(true)} />
                </div>
              </Suspense>
            </SceneBoundary>
          )}
          <div
            aria-hidden="true"
            className="absolute inset-0 bg-[linear-gradient(90deg,rgb(5_8_8/0.94)_0%,rgb(5_8_8/0.8)_30%,rgb(5_8_8/0.2)_58%,transparent_75%),linear-gradient(180deg,rgb(5_8_8/0.7)_0%,transparent_18%,transparent_78%,rgb(5_8_8/0.9)_100%)] max-lg:bg-[linear-gradient(180deg,rgb(5_8_8/0.6)_0%,rgb(5_8_8/0.55)_40%,rgb(5_8_8/0.92)_100%)]"
          />
        </div>

        {dragging && (
          <div aria-hidden="true" className="pointer-events-none absolute inset-4 z-20 grid place-items-center rounded-[var(--r-panel)] border-2 border-dashed border-a10/70 bg-a0/40">
            <p className="hero-sub m-0 text-a10">Drop to read this photo</p>
          </div>
        )}

        <div className="relative mx-auto flex w-full max-w-[1400px] flex-1 items-center px-5 pt-48 pb-24 sm:px-10 md:pt-40 lg:pt-36 lg:pb-20">
          <div className="max-w-[36rem]">
            <h1 id="home-title" className="hero-title m-0">
              RoadCondition AI
            </h1>
            <p className="hero-sub mt-4 mb-0">
              Detects whether a road is <br className="max-sm:hidden" />
              <span className="text-lavender">Dry, Wet, Snowy or Icy</span>
            </p>
            <p className="mt-6 mb-0 max-w-[30rem] text-[17px] leading-relaxed text-a10/80">
              A computer vision model that reads a road photo and names its surface condition, with a
              calibrated confidence and a heatmap of where it looked. A student project trained on
              RSCD, not a safety system.
            </p>

            <div className="mt-8 flex flex-wrap gap-3">
              <a href={hrefOf("report")} className="cta">
                <PlayIcon />
                Try the Demo
              </a>
              <a href={REPO_URL} target="_blank" rel="noreferrer" className="cta cta-outline">
                <GitHubIcon />
                View on GitHub
              </a>
            </div>

            <div role="group" aria-label="Preview a surface on the roads" className="mt-10 grid grid-cols-2 gap-3 sm:flex sm:flex-wrap sm:items-center">
              {CHIPS.map(({ cls, label, Icon }, i) => (
                <div key={cls} className="flex items-center gap-3 max-sm:[&>.surface-chip]:w-full">
                  {i > 0 && <span className="h-6 w-px bg-a10/20 max-sm:hidden" aria-hidden="true" />}
                  <button
                    type="button"
                    className="surface-chip"
                    aria-pressed={pinned === cls}
                    data-preview={hovered === cls && pinned !== cls ? "" : undefined}
                    onMouseEnter={() => setHovered(cls)}
                    onMouseLeave={() => setHovered(null)}
                    onFocus={() => setHovered(cls)}
                    onBlur={() => setHovered(null)}
                    onClick={() => setPinned(pinned === cls ? null : cls)}
                  >
                    <Icon width={24} height={24} className={`chip-icon chip-icon-${cls}`} />
                    <span className="flex flex-col items-start leading-tight">
                      <span className="text-[15px] font-semibold">{label}</span>
                      <span className="text-[13px] text-a10/70">Road</span>
                    </span>
                  </button>
                </div>
              ))}
            </div>
            <p className="sr-only" aria-live="polite">
              {preview === null ? "The roads in the photo are shown as taken." : `Every road in the photo now shows ${prettyClass(preview)}.`}
            </p>
          </div>
        </div>

        <button
          type="button"
          className="absolute bottom-6 left-1/2 grid h-11 w-11 -translate-x-1/2 place-items-center rounded-full text-a10/80 transition-colors hover:text-a10"
          onClick={scrollToHowItWorks}
        >
          <span className="sr-only">How it works</span>
          <ChevronIcon width={28} height={28} />
        </button>
      </section>

      <section id="how-it-works" aria-labelledby="how-title" className="scroll-mt-20 border-t border-a10/10 bg-a0">
        <div className="mx-auto max-w-[1400px] px-5 py-20 sm:px-10 lg:py-24">
          <h2 id="how-title" className="hero-h2 m-0">
            How it works
          </h2>
          <ol className="steps mt-12 grid list-none gap-10 p-0 md:grid-cols-3 md:gap-12">
            <li>
              <h3 className="m-0 text-[20px] font-semibold">Drop a road photo</h3>
              <p className="mt-2 mb-0 max-w-[34ch] text-[16px] leading-relaxed text-muted">
                A JPEG, PNG or WebP of the road surface, up to 10 MB. A view like a vehicle camera's works best.
              </p>
            </li>
            <li>
              <h3 className="m-0 text-[20px] font-semibold">A network trained on RSCD reads it</h3>
              <p className="mt-2 mb-0 max-w-[34ch] text-[16px] leading-relaxed text-muted">
                Six friction classes: dry, wet, water, fresh snow, melted snow and ice. Confidence is
                temperature-scaled on held-out RSCD images.
              </p>
            </li>
            <li>
              <h3 className="m-0 text-[20px] font-semibold">You get a reading, or “not sure”</h3>
              <p className="mt-2 mb-0 max-w-[34ch] text-[16px] leading-relaxed text-muted">
                Below the confidence threshold the page shows the closest two classes instead of a
                label, with a heatmap of where the model looked.
              </p>
            </li>
          </ol>
          <div className="mt-12 flex flex-wrap items-center gap-x-6 gap-y-3">
            <a href={hrefOf("report")} className="cta">
              <PlayIcon />
              Try the Demo
            </a>
            <a href={hrefOf("study")} className="text-[15px] font-semibold">
              Cross-country study
            </a>
            <button type="button" className="text-[15px] font-semibold text-a10/80 underline underline-offset-4" onClick={onLimits}>
              What the model has not seen
            </button>
          </div>
        </div>
      </section>
    </>
  );
}
