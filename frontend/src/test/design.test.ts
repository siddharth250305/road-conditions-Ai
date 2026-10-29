// Checks the survey-sheet design rules (PRD pages 10-11) against the stylesheet itself.
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

const SRC = join(__dirname, "..");
const css = readFileSync(join(SRC, "index.css"), "utf8");

function block(re: RegExp): Record<string, string> {
  const match = re.exec(css);
  if (!match) throw new Error(`stylesheet block not found: ${re}`);
  return Object.fromEntries([...match[1]!.matchAll(/--([\w-]+):\s*([^;]+);/g)].map((m) => [m[1]!, m[2]!.trim()]));
}

const light = block(/:root\s*\{([^}]*)\}/);
const dark = block(/prefers-color-scheme: dark\)\s*\{\s*:root:not\(\[data-theme="light"\]\)\s*\{([^}]*)\}/);
const darkForced = block(/:root\[data-theme="dark"\]\s*\{([^}]*)\}/);
const theme = block(/@theme\s*\{([^}]*)\}/);

function luminance(hex: string): number {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255) as [number, number, number];
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
  it("uses the PRD colours", () => {
    expect(light["surface"]).toBe("#ede8dc");
    expect(light["text"]).toBe("#16181a");
    expect(theme["color-lane"]).toBe("#f2b705");
  });

  it("asphalt ink on paper is about 14.5:1", () => {
    expect(contrast(light["text"]!, light["surface"]!)).toBeGreaterThan(14);
  });

  it("dark variant swaps asphalt and paper", () => {
    expect(dark["surface"]).toBe(light["text"]);
    expect(dark["text"]).toBe(light["surface"]);
    expect(darkForced).toEqual(dark);
  });
});

describe("contrast of every text pair", () => {
  it.each([
    ["light text on paper", () => contrast(light["text"]!, light["surface"]!), 7],
    ["light muted text on paper", () => contrast(light["muted"]!, light["surface"]!), 4.5],
    ["dark text on asphalt", () => contrast(dark["text"]!, dark["surface"]!), 7],
    ["dark muted text on asphalt", () => contrast(dark["muted"]!, dark["surface"]!), 4.5],
    ["ink text on the yellow fill", () => contrast(light["ink"]!, theme["color-lane"]!), 7],
    ["yellow focus ring on asphalt", () => contrast(theme["color-lane"]!, dark["surface"]!), 3],
    ["ink focus ring on paper", () => contrast(light["ink"]!, light["surface"]!), 3],
  ])("%s", (_name, ratio, min) => {
    expect(ratio()).toBeGreaterThanOrEqual(min);
  });

  it("yellow is never used as a text colour (about 1.5:1 on paper)", () => {
    expect(contrast(theme["color-lane"]!, light["surface"]!)).toBeLessThan(2);
    for (const file of sourceFiles()) {
      const text = readFileSync(file, "utf8");
      expect(text, file).not.toMatch(/\btext-lane\b/);
      expect(text, file).not.toMatch(/(^|[^-])color:\s*var\(--color-lane\)/);
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
  it("uses the PRD durations", () => {
    expect(light["dur-marker"]).toBe("300ms");
    expect(light["dur-heatmap"]).toBe("180ms");
    expect(light["dur-stamp"]).toBe("120ms");
  });

  it("the marker eases out, the heatmap crossfades, the stamp settles", () => {
    expect(css).toMatch(/\.ruler-marker\s*\{[^}]*var\(--dur-marker\) cubic-bezier\(0\.2, 0\.7, 0\.3, 1\)/s);
    expect(css).toMatch(/\.fade-layer\s*\{[^}]*opacity var\(--dur-heatmap\)/s);
    expect(css).toMatch(/\.stamp\s*\{[^}]*animation: stamp-settle var\(--dur-stamp\)/s);
  });

  it("all motion switches off under reduced motion", () => {
    const reduced = /@media \(prefers-reduced-motion: reduce\)\s*\{([\s\S]*)\}\s*$/.exec(css)?.[1] ?? "";
    expect(reduced).toMatch(/animation-duration:\s*0\.01ms !important/);
    expect(reduced).toMatch(/transition-duration:\s*0\.01ms !important/);
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

  it("mono data is 13 px with 4% letter spacing", () => {
    expect(css).toMatch(/\.data\s*\{[^}]*font-size:\s*13px[^}]*letter-spacing:\s*0\.04em/s);
  });

  it("display type is tight, uppercase Big Shoulders", () => {
    expect(css).toMatch(/\.display\s*\{[^}]*font-weight:\s*800[^}]*text-transform:\s*uppercase[^}]*line-height:\s*0\.9/s);
    expect(theme["font-display"]).toContain("Big Shoulders Display");
    expect(theme["font-mono"]).toContain("IBM Plex Mono");
    expect(theme["font-sans"]).toContain("Public Sans");
  });
});
