// Checks the road-weather design rules (PRODUCT.md, DESIGN.md) against the stylesheet itself.
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

const SRC = join(__dirname, "..");
const css = readFileSync(join(SRC, "index.css"), "utf8");

function block(re: RegExp): Record<string, string> {
  const match = re.exec(css);
  if (!match) throw new Error(`stylesheet block not found: ${re}`);
  return Object.fromEntries([...match[1]!.matchAll(/--([\w-]+):\s*([^;]+);/g)].map((m) => [m[1]!, m[2]!.trim()]));
}

const root = block(/:root\s*\{([^}]*)\}/);
const theme = block(/@theme\s*\{([^}]*)\}/);

/** Follow var(--x) references in :root down to a hex value. */
function hex(name: string): string {
  let value = root[name] ?? theme[name];
  for (let i = 0; value && i < 5; i++) {
    const ref = /^var\(--([\w-]+)\)$/.exec(value);
    if (!ref) break;
    value = root[ref[1]!] ?? theme[ref[1]!];
  }
  if (!value || !/^#[0-9a-f]{6}$/i.test(value)) throw new Error(`--${name} is not a hex colour: ${value}`);
  return value;
}

function luminance(color: string): number {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(color.slice(i, i + 2), 16) / 255) as [number, number, number];
  const lin = (c: number) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

export function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [number, number];
  return (hi + 0.05) / (lo + 0.05);
}

function sourceFiles(dir = SRC): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const path = join(dir, e.name);
    if (e.isDirectory()) return e.name === "test" ? [] : sourceFiles(path);
    return /\.(tsx?|css)$/.test(e.name) ? [path] : [];
  });
}

describe("palette", () => {
  const ramp = Array.from({ length: 11 }, (_, i) => `a${i}`);

  it("every neutral comes from one eleven-step asphalt ramp, darkest to marking white", () => {
    const lum = ramp.map((step) => luminance(hex(step)));
    for (let i = 1; i < lum.length; i++) expect(lum[i]!, ramp[i]).toBeGreaterThan(lum[i - 1]!);
    expect(root["surface"]).toBe("var(--a1)");
    expect(root["text"]).toBe("var(--a10)");
    expect(root["muted"]).toMatch(/^var\(--a\d+\)$/);
  });

  it("temporary-works yellow is only a fill for provisional notices, never text", () => {
    for (const file of sourceFiles()) {
      const text = readFileSync(file, "utf8");
      expect(text, file).not.toMatch(/\btext-works\b/);
      expect(text, file).not.toMatch(/(^|[^-])color:\s*var\(--works\)/);
      if (/\bbg-works\b/.test(text)) expect(text, file).toMatch(/MOCK_BANNER/);
    }
  });

  it("there is a single dark theme: no light variant to fall out of step", () => {
    expect(css).not.toMatch(/prefers-color-scheme/);
    expect(css).toMatch(/color-scheme:\s*dark/);
  });
});

describe("contrast of every text pair", () => {
  it.each([
    ["marking white on asphalt", () => contrast(hex("text"), hex("a1")), 7],
    ["marking white on the raised panel", () => contrast(hex("text"), hex("a2")), 7],
    ["muted text on asphalt", () => contrast(hex("muted"), hex("a1")), 4.5],
    ["muted text on the raised panel", () => contrast(hex("muted"), hex("a3")), 4.5],
    ["asphalt lettering on the solid white button", () => contrast(hex("a0"), hex("a10")), 7],
    ["asphalt lettering on the mock banner", () => contrast(hex("on-works"), hex("works")), 7],
    ["focus ring on asphalt", () => contrast(hex("a10"), hex("a0")), 3],
    ["lavender class words on the hero's darkest ground", () => contrast(hex("lavender"), "#050808"), 7],
  ])("%s", (_name, ratio, min) => {
    expect(ratio()).toBeGreaterThanOrEqual(min);
  });
});

describe("states", () => {
  it("the current route carries a solid marking-white lane line; hovering another shows it dashed", () => {
    expect(css).toMatch(/\.site-bar\s*\{[^}]*backdrop-filter:\s*blur/s);
    expect(css).toMatch(/\.nav-link\[aria-current="page"\] \.lane-marker\s*\{[^}]*background:\s*var\(--a10\)/s);
    expect(css).toMatch(/\.nav-link:hover \.lane-marker\s*\{[^}]*repeating-linear-gradient/s);
  });

  it("the hero follows the reference: a real headline, class words in lavender, pill calls to action", () => {
    const home = readFileSync(join(SRC, "components/HomeScreen.tsx"), "utf8");
    expect(home).toMatch(/<h1 id="home-title" className="hero-title m-0">/);
    expect(home).toMatch(/className="text-lavender"/);
    expect(css).toMatch(/\.cta\s*\{[^}]*border-radius:\s*999px/s);
    expect(css).toMatch(/\.surface-chip\[aria-pressed="true"\]/);
  });

  it("a candidate is dashed: the 'not sure' verdict and ruler ghosts use dashed lines", () => {
    expect(css).toMatch(/\.dashed\s*\{[^}]*border:\s*2px dashed/s);
    const verdict = readFileSync(join(SRC, "components/VerdictPanel.tsx"), "utf8");
    expect(verdict).toMatch(/stamp dashed/);
    const ruler = readFileSync(join(SRC, "components/FrictionRuler.tsx"), "utf8");
    expect(ruler).toMatch(/border-dashed/);
  });

  it("no thick coloured side stripes on notices or cards", () => {
    for (const file of sourceFiles()) {
      expect(readFileSync(file, "utf8"), file).not.toMatch(/\bborder-[lr]-[2-9]\b/);
    }
  });
});

describe("class tints", () => {
  const classes = ["dry", "wet", "water", "fresh_snow", "melted_snow", "ice"];

  it("defines a tint and a pattern rule for every class", () => {
    for (const c of classes) {
      expect(theme[`color-tint-${c}`], c).toMatch(/^#[0-9a-f]{6}$/);
      expect(css).toContain(`.tint-${c} {`);
    }
  });

  it("every class except dry also has its own pattern, so colour is never the only signal", () => {
    const patterns = classes
      .filter((c) => c !== "dry")
      .map((c) => new RegExp(`\\.tint-${c} \\{([^}]*)\\}`).exec(css)![1]!)
      .map((rule) => /background-image:([^;]+);/.exec(rule)?.[1]?.trim());
    expect(patterns.every(Boolean)).toBe(true);
    expect(new Set(patterns).size).toBe(patterns.length);
  });
});

describe("motion", () => {
  it("uses the agreed durations", () => {
    expect(root["dur-marker"]).toBe("300ms");
    expect(root["dur-heatmap"]).toBe("180ms");
    expect(root["dur-stamp"]).toBe("160ms");
  });

  it("the marker eases out, the heatmap crossfades, the verdict is painted on", () => {
    expect(css).toMatch(/\.ruler-marker\s*\{[^}]*var\(--dur-marker\) cubic-bezier\(0\.2, 0\.7, 0\.3, 1\)/s);
    expect(css).toMatch(/\.fade-layer\s*\{[^}]*opacity var\(--dur-heatmap\)/s);
    expect(css).toMatch(/\.stamp\s*\{[^}]*animation: stamp-settle var\(--dur-stamp\)/s);
  });

  it("all motion switches off under reduced motion", () => {
    const reduced = /@media \(prefers-reduced-motion: reduce\)\s*\{([\s\S]*)\}\s*$/.exec(css)?.[1] ?? "";
    expect(reduced).toMatch(/animation-duration:\s*0\.01ms !important/);
    expect(reduced).toMatch(/transition-duration:\s*0\.01ms !important/);
  });

  it("route changes slide with the direction of travel and fall back to a short crossfade", () => {
    expect(css).toMatch(/:root\[data-nav="back"\]::view-transition-new\(root\)/);
    expect(css).toMatch(/view-transition-name:\s*lane-marker/);
    const reduced = /@media \(prefers-reduced-motion: reduce\)\s*\{([\s\S]*)\}\s*$/.exec(css)?.[1] ?? "";
    expect(reduced).toMatch(/route-fade-in/);
  });

  it("the 3D scene is told about reduced motion", () => {
    const home = readFileSync(join(SRC, "components/HomeScreen.tsx"), "utf8");
    expect(home).toMatch(/prefers-reduced-motion: reduce/);
    expect(home).toMatch(/reducedMotion=\{reduced\}/);
  });
});

describe("type", () => {
  it("nothing is set below 11 px", () => {
    for (const file of sourceFiles()) {
      const text = readFileSync(file, "utf8");
      for (const m of text.matchAll(/font-size:\s*(\d+(?:\.\d+)?)px/g)) {
        expect(Number(m[1]), `${file}: ${m[0]}`).toBeGreaterThanOrEqual(11);
      }
      for (const m of text.matchAll(/text-\[(\d+(?:\.\d+)?)px\]/g)) {
        expect(Number(m[1]), `${file}: ${m[0]}`).toBeGreaterThanOrEqual(11);
      }
    }
  });

  it("data is 13 px with 4% letter spacing and tabular figures", () => {
    expect(css).toMatch(/\.data\s*\{[^}]*font-size:\s*13px[^}]*letter-spacing:\s*0\.04em[^}]*tabular-nums/s);
  });

  it("Barlow, from highway signage: the condensed cut for screen titles and headings, the regular cut for UI and the hero", () => {
    expect(theme["font-display"]).toMatch(/^"Barlow Condensed"/);
    expect(theme["font-sans"]).toMatch(/^"Barlow"/);
    expect(css).toMatch(/\.display\s*\{[^}]*font-family:\s*var\(--font-display\)[^}]*font-weight:\s*700/s);
    expect(css).toMatch(/\.heading\s*\{[^}]*font-family:\s*var\(--font-display\)/s);
    const main = readFileSync(join(SRC, "main.tsx"), "utf8");
    expect([...main.matchAll(/@fontsource\/barlow\/latin-(\d+)\.css/g)].map((m) => m[1]).sort()).toEqual(["400", "600", "700"]);
    expect([...main.matchAll(/@fontsource\/barlow-condensed\/latin-(\d+)\.css/g)].map((m) => m[1]).sort()).toEqual(["600", "700"]);
  });

  it("no tracked-out uppercase eyebrow labels and no synthetic bold", () => {
    expect(css).not.toMatch(/\.label\s*\{/);
    for (const file of sourceFiles()) {
      const text = readFileSync(file, "utf8");
      expect(text, file).not.toMatch(/className="[^"]*\blabel\b/);
      expect(text, file).not.toMatch(/\bfont-bold\b/);
    }
  });

  it("working-screen controls use sign-plate corners; only the hero's calls to action are pills", () => {
    for (const cls of ["btn-solid", "btn-quiet"]) {
      expect(css, cls).toMatch(new RegExp(`\\.${cls}\\s*\\{[^}]*border-radius:\\s*var\\(--r-plate\\)`, "s"));
    }
  });
});
