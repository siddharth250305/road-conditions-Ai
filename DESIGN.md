---
name: Road Condition AI
description: Night asphalt and marking white; a drone photograph of a real junction whose roads repaint with the friction class, and working screens cut as sign plates.
colors:
  asphalt-0: "#0a0d0c"
  asphalt-1: "#0f1311"
  asphalt-2: "#151a17"
  asphalt-3: "#1c221f"
  asphalt-4: "#252c28"
  asphalt-5: "#333b36"
  asphalt-6: "#4a534d"
  asphalt-7: "#6c756f"
  asphalt-8: "#9aa29c"
  asphalt-9: "#c3c9c4"
  marking-white: "#eceee9"
  night-black: "#050808"
  site-bar: "rgb(5 8 8 / 0.82)"
  lavender: "#bdb8ff"
  works-yellow: "#e8b23a"
  glass-fill: "rgb(21 26 23 / 0.52)"
  glass-fill-hover: "rgb(43 51 46 / 0.62)"
  glass-edge: "rgb(236 238 233 / 0.16)"
  glass-edge-hover: "rgb(236 238 233 / 0.34)"
  chip-tile: "rgb(10 14 14 / 0.62)"
  chip-tile-hover: "rgb(24 30 30 / 0.75)"
  glyph-dry: "#f0c27a"
  glyph-wet: "#8fb4ff"
  glyph-snow: "#cfe4ff"
  glyph-ice: "#9fd6e8"
  tint-dry: "#8d918b"
  tint-wet: "#4f6f80"
  tint-water: "#2f6d99"
  tint-fresh-snow: "#eef2f4"
  tint-melted-snow: "#9aa7ad"
  tint-ice: "#9fd6e8"
typography:
  hero-title:
    fontFamily: "Barlow, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "clamp(2.9rem, 5.6vw, 4.75rem)"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "-0.025em"
  hero-sub:
    fontFamily: "Barlow, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "clamp(1.75rem, 3.2vw, 2.75rem)"
    fontWeight: 700
    lineHeight: 1.12
    letterSpacing: "-0.02em"
  hero-h2:
    fontFamily: "Barlow, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "clamp(2rem, 3.4vw, 2.75rem)"
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: "-0.02em"
  display:
    fontFamily: "Barlow Condensed, Arial Narrow, system-ui, sans-serif"
    fontSize: "clamp(2.75rem, 5vw, 4.5rem)"
    fontWeight: 700
    lineHeight: 0.94
    letterSpacing: "0.005em"
  headline:
    fontFamily: "Barlow Condensed, Arial Narrow, system-ui, sans-serif"
    fontSize: "clamp(2rem, 3.4vw, 3rem)"
    fontWeight: 700
    lineHeight: 0.94
    letterSpacing: "0.005em"
  heading:
    fontFamily: "Barlow Condensed, Arial Narrow, system-ui, sans-serif"
    fontSize: "19px"
    fontWeight: 600
    lineHeight: 1.15
    letterSpacing: "0.005em"
  wordmark:
    fontFamily: "Barlow, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "20px"
    fontWeight: 600
    letterSpacing: "-0.01em"
  nav:
    fontFamily: "Barlow, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1
  body:
    fontFamily: "Barlow, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.5
  control:
    fontFamily: "Barlow, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "16px"
    fontWeight: 600
    lineHeight: 1
  data:
    fontFamily: "Barlow, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    letterSpacing: "0.04em"
    fontFeature: "tnum"
  caption:
    fontFamily: "Barlow Condensed, Arial Narrow, system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 600
    letterSpacing: "0.01em"
rounded:
  plate: "6px"
  inset: "8px"
  panel: "12px"
  tile: "16px"
  pill: "999px"
spacing:
  gutter-sm: "16px"
  gutter-md: "24px"
  gutter-lg: "40px"
  bar-x-sm: "20px"
  panel-sm: "20px"
  panel-lg: "28px"
  container: "1400px"
components:
  cta:
    backgroundColor: "{colors.marking-white}"
    textColor: "{colors.asphalt-0}"
    typography: "{typography.control}"
    rounded: "{rounded.pill}"
    padding: "0 1.6rem"
    height: "50px"
  cta-hover:
    backgroundColor: "{colors.asphalt-9}"
  cta-outline:
    textColor: "{colors.marking-white}"
    typography: "{typography.control}"
    rounded: "{rounded.pill}"
    padding: "0 1.6rem"
    height: "50px"
  cta-small:
    backgroundColor: "{colors.marking-white}"
    textColor: "{colors.asphalt-0}"
    rounded: "{rounded.pill}"
    padding: "0 1.2rem"
    height: "42px"
  surface-chip:
    backgroundColor: "{colors.chip-tile}"
    textColor: "{colors.marking-white}"
    rounded: "{rounded.tile}"
    padding: "0.55rem 1.1rem 0.55rem 0.9rem"
    height: "60px"
  surface-chip-hover:
    backgroundColor: "{colors.chip-tile-hover}"
  site-bar:
    backgroundColor: "{colors.site-bar}"
    textColor: "{colors.marking-white}"
    height: "64px"
  nav-link:
    textColor: "{colors.marking-white}"
    typography: "{typography.nav}"
    padding: "0 0.85rem"
    height: "44px"
  button-solid:
    backgroundColor: "{colors.marking-white}"
    textColor: "{colors.asphalt-0}"
    typography: "{typography.control}"
    rounded: "{rounded.plate}"
    padding: "0.6rem 1.15rem"
    height: "44px"
  button-solid-hover:
    backgroundColor: "{colors.asphalt-9}"
  button-quiet:
    textColor: "{colors.marking-white}"
    rounded: "{rounded.plate}"
    padding: "0.4rem 0.85rem"
    height: "36px"
  glass-panel:
    backgroundColor: "{colors.glass-fill}"
    rounded: "{rounded.panel}"
    padding: "6px"
  reading-panel:
    backgroundColor: "{colors.asphalt-2}"
    rounded: "{rounded.panel}"
    padding: "{spacing.panel-lg}"
  wash:
    backgroundColor: "{colors.asphalt-2}"
    rounded: "{rounded.panel}"
    padding: "{spacing.panel-sm}"
  provisional-banner:
    backgroundColor: "{colors.works-yellow}"
    textColor: "{colors.asphalt-0}"
    padding: "6px 16px"
  provisional-note:
    backgroundColor: "{colors.works-yellow}"
    textColor: "{colors.asphalt-0}"
    rounded: "{rounded.plate}"
    padding: "6px 16px"
---

# Design System: Road Condition AI

## Overview

**Creative North Star: "Road-Weather Night Shift"**

A gritter crew's view of the network after dark. The home screen is a hero over a real drone photograph of a junction at dusk: the photo is the ground plane of a 3D scene seen almost straight down, and one surface layer fitted to its roads repaints them with the chosen friction class (wet sheen catching the lamp pools, standing water, snow, slush, ice) under the scene's own light. A left copy column sits on dark gradients over the photo: a large bold headline, a subline whose class words are lavender, honest copy, two pill calls to action and four surface tiles that drive the road layer. Below it a plain "How it works" section numbers the three steps.

The working screens (Analyse, Study) keep the older, stricter grammar: asphalt panels, marking-white text, condensed sign lettering for titles and headings, 6px control plates and 12px panel plates, give-way corner marks around photographs. State is drawn the way roads draw it everywhere: a solid line is committed, a dashed line is a candidate. Temporary-works yellow appears only to say "provisional".

Motion is weighted in the scene and quick in the HTML: the camera settles from slightly high to its framing over 1.8s and leans a little with the pointer; the road layer eases between looks; route changes slide in the direction of travel while the site bar holds still; controls answer in 160ms; reduced motion keeps only a short crossfade.

**Key Characteristics:**
- The home background is the drone photograph itself; only its roads change, through one soft road-shaped layer, never a painted illustration.
- One eleven-step green-grey asphalt ramp supplies every working-screen neutral; the hero and site bar sit on a deeper night black.
- Hue is spent in the photograph, the class tints, the chip glyphs, the lavender class words and the provisional yellow; chrome stays asphalt, white and glass.
- Solid edge or marking = committed / selected; dashed = candidate / preview / pending.
- Pills only for the hero and site-bar calls to action; working-screen controls are 6px plates.
- Barlow bold, tight and large for the hero; Barlow Condensed for working-screen titles and headings.
- Class tints always carry a pattern; hero chips always carry a glyph and a name, so colour is never the only signal.

## Colors

A near-monochrome green-grey night with marking white as the voice; hue lives in the photograph and in a few small, named places.

### Primary
- **Marking White** (`marking-white`): body text, headlines, the solid call-to-action fill, committed borders and lane markers, step numerals and the focus ring. A road-paint white, slightly warm-green.

### Secondary
- **Lavender** (`lavender`): the class words in the hero subline ("Dry, Wet, Snowy or Icy"), as a flat colour, never a gradient. Contrast against night black is held at 7:1 or better. Nothing else.
- **Temporary-Works Yellow** (`works-yellow`): the mock-model banner across the top of the app and the "synthetic numbers" note on Study, with asphalt-0 lettering; also the text-selection highlight. Never text colour, never decoration.

### Tertiary: class colour
- **Chip glyphs** (`glyph-dry` warm sun, `glyph-wet` rain blue, `glyph-snow` frost blue, `glyph-ice` glacier): the stroke colour of the surface glyph on each hero tile. Only on those glyphs.
- **Class tints** (Dry grey stone; Wet slate blue with fine horizontal hatch; Water deep blue with bold white bars; Fresh Snow near-white with an asphalt dot grid; Melted Snow grey-blue with a 45° hatch; Ice pale glacier with a cross-hatch): class swatches on the working screens (verdict, ruler). Pattern ink is asphalt, white on water.

### Neutral
- **Night Black** (`night-black`): the hero ground, the 3D scene background, the hero's copy gradients and the base of the site bar (`site-bar`, 82% with a 14px blur; on home a 92%→55% gradient so the photo shows through).
- **Asphalt 0** (a0): page base, the How it works section, lettering on white and yellow.
- **Asphalt 1** (a1): working-screen surface, with a faint fractal grain and a soft top-right light.
- **Asphalt 2** (a2): recessed panels (reading panel, washes, photo frame bed); the glass fill is this step at 52%.
- **Asphalt 4–5** (a4, a5): wash borders and rules, scrollbar thumb.
- **Asphalt 8** (a8): muted text for secondary copy and panel headings.
- **Asphalt 9** (a9): call-to-action and solid-button hover, dashed candidate outlines and the dashed hover lane marker.
- Marking white at an alpha carries hero secondary text (80% paragraph, 70% chip sub-line and resting nav link) and hairlines (8% under the site bar, 10–12% on tiles and panels).

### Named Rules
**The One Ramp Rule.** Every neutral comes from a0–a10, night black, or one of those at an alpha. No other greys, no blue-blacks.

**The Works Yellow Rule.** Temporary-works yellow means provisional (mock model, synthetic data). It never decorates, never marks a hover, never becomes an accent or a text colour.

**The Named Hue Rule.** Outside the photograph and scene, hue appears only as class tints, the four chip glyphs, the lavender class words and the provisional yellow. Every other piece of chrome stays asphalt, white and glass.

## Typography

**Hero Font:** Barlow 700 (with system-ui fallback)
**Working-Screen Display and Heading Font:** Barlow Condensed 700 / 600 (with Arial Narrow, system-ui fallback)
**Body Font:** Barlow 400 and 600

**Character:** One family drawn from highway signs and licence plates. The hero speaks in the regular cut at its boldest, large and tightly tracked; the working screens set titles in condensed capitals like road lettering and name their parts in condensed sentence case; running text, controls and figures are the regular cut. Five cuts are loaded (Barlow 400/600/700, Barlow Condensed 600/700) and nothing else; bold copy renders at 600.

### Hierarchy
- **Hero title** (Barlow 700, clamp(2.9rem, 5.6vw, 4.75rem), 1, -0.025em, balanced): the product name, home only.
- **Hero subline** (Barlow 700, clamp(1.75rem, 3.2vw, 2.75rem), 1.12, -0.02em): the one-line promise under the title; also the drag-over prompt.
- **Hero H2** (Barlow 700, clamp(2rem, 3.4vw, 2.75rem), 1.1, -0.02em): home section titles ("How it works"). Step titles under it are Barlow 600 20px.
- **Display** (Condensed 700, capitals, 0.94, balanced): the verdict class name at clamp(2.75rem, 5vw, 4.5rem); verdict empty and reading states and the Analyse drop prompt smaller.
- **Headline** (Condensed 700, capitals, clamp(2rem, 3.4vw, 3rem)): screen titles on Analyse and Study.
- **Heading** (Condensed 600, 19px, 1.15, sentence case): panel and section headings, limits terms, the "Or try a sample" title. Muted (a8).
- **Wordmark** (Barlow 600, 20px, -0.01em): "RoadCondition AI" beside the crossroads mark; hidden visually under 480px.
- **Nav** (Barlow 400, 15px): site-bar links at 70% white, full white on hover, 600 on the current page.
- **Body** (Barlow 400, 16px/1.5; hero paragraph 17px relaxed at 80% white, max 30rem; step copy 16px relaxed muted, max 34ch; secondary copy 13–15px muted).
- **Control** (Barlow 600, 16px, line-height 1): calls to action and the solid button; small call to action 15px; quiet buttons 14px; chip names 15px over a 13px sub-line.
- **Data** (Barlow 400, 13px, 0.04em, tabular figures): percentages, measurements, chart ticks, tables. The headline confidence figure is 44px, 600, tracked -0.02em.
- **Caption** (Condensed 600, 13px, sentence case): chart axis captions.

### Named Rules
**The Two Voices Rule.** The hero and home sections speak in Barlow 700, sentence case, tight negative tracking; the working screens speak in Barlow Condensed (700 capitals for titles, 600 sentence case for headings). Do not mix them on one screen.

**The No Synthetic Bold Rule.** Never request a weight that is not loaded; 600 is the strong weight for UI and copy.

**The Tabular Figures Rule.** Any number that is a measurement or probability is set in Barlow with tabular figures.

**The Twelve Pixel Floor.** No HTML text is set below 12px; 12px is reserved for sample captions and notes.

## Layout

**Site bar.** One row inside the 1400px container, 64px tall, 20px gutters (40px from 640px): mark and wordmark left, the route links centred, a 44px GitHub icon link and the small Get started pill right. On home it lies over the hero with the gradient fill; on working screens it sticks to the top. Below 768px the links drop to a full-width second row that scrolls sideways. It holds still across route changes.

**Home hero.** At least 640px tall, exactly 100svh from 1024px. The 3D photo layer covers the section; on wide screens the camera frames the photo's full width with the crossing about 58% across, and on narrow screens it fits the photo and keeps the crossing in view. Two dark gradients sit over it: horizontal from the left (94% → 80% at 30% → 20% at 58% → clear at 75%) plus a top and bottom fade; below 1024px a single vertical gradient darkens the whole photo towards the bottom. The copy column is left-aligned inside the container, max 36rem, vertically centred, with top padding that clears the bar (11rem / 9rem / 6rem at base / md / lg). Order: title, subline (16px gap), paragraph (24px), two calls to action (32px, 12px gap), surface tiles (40px; a two-column grid on phones, a wrapping row with 1px dividers from 640px). A chevron centred 24px from the bottom scrolls to How it works. Dropping a photo anywhere on the hero reads it.

**How it works.** a0, a 10% white top hairline, 80px vertical padding (96px from 1024px): the H2, then a three-column numbered list from 768px (48px gap), then a call-to-action row with two text links.

**Working screens.** Analyse and Study sit inside the 1400px container with 16 / 24 / 40px gutters (base / sm / lg): a title row, then a two-column grid (photo 1.55fr : reading 1fr on Analyse; paired charts at 1280px on Study). Panels pad 20px, 28px from 640px. Interactive targets keep 44px (36px for in-panel quiet buttons).

## Elevation & Depth

Depth comes from the photograph and the light on its roads. HTML over the hero is translucent: the site bar and the surface tiles are dark, blurred glass with hairline edges and no drop shadow; the calls to action are flat. Working screens are flatter still: recessed asphalt panels with a 10% white hairline and no shadow; glass panels (sample cards) keep a soft inner highlight and a long drop shadow.

### Shadow Vocabulary
- **Glass panel** (`inset 0 1px 0 rgb(255 255 255 / 0.07), 0 18px 40px -18px rgb(0 0 0 / 0.75)`, with an 18px blur and 140% saturation): sample cards on Analyse, and the limits bar when it overlays.

### Named Rules
**The Photo Carries Depth Rule.** Over the hero nothing casts a shadow; layers separate by blur and darkness, and the only lit material is the road layer in the scene.

## Shapes

Two shape languages, kept apart by screen. The hero and site bar use pills (999px) for calls to action, round 40px step numerals and 44px round icon targets, and 16px tiles for the surface chips. The working screens use sign-plate corners: controls are 6px plates (solid and quiet buttons, the provisional note), panels 12px plates (reading panels, washes, glass sample cards, the photo frame, the verdict stamp, the drag-over frame), and inset drop zones and the verdict swatch 8px. Borders are 1px hairlines at rest; call-to-action borders are 1.5px; state borders on working screens are 2px.

Signature silhouettes: the **give-way L** (four 24 × 3px marking-white corners at 70%, 12px outside the photo frame on Analyse) and the **lane marker** (a 3px bar under a nav link).

### Named Rules
**The Lane Line Rule.** A solid line is committed (pinned surface tile, current page's lane marker, confident verdict, active drop target). A dashed line is a candidate (previewed surface tile, hovered lane marker, uncertain verdict, idle drop zone, drag-over frame, pending card, step connectors). Never swap them.

**The Pill Boundary Rule.** Pills belong only to the hero and site-bar calls to action. Working-screen controls stay 6px plates and panels 12px plates.

## Components

### Calls to action (hero and site bar)
- **Shape:** pill (999px), 50px tall, 1.5px border.
- **Solid:** marking white with asphalt-0 lettering, Barlow 600 16px, a 16px solid play glyph before "Try the Demo". Hover drops fill and border to a9; press nudges down 1px.
- **Outline:** transparent with a 72% white border and white lettering, the GitHub mark before "View on GitHub". Hover adds a 10% white fill and a full white border.
- **Small:** 42px tall, 15px, the site bar's "Get started".
- **Timing:** 160ms on cubic-bezier(0.16, 1, 0.3, 1).

### Buttons (working screens)
- **Shape:** sign plate (6px).
- **Solid:** marking white with asphalt-0 lettering, Barlow 600 16px, 44px tall; Retry and similar single actions. Hover a9; press nudges 1px.
- **Quiet:** 6% white fill, glass edge, Barlow 600 14px, 36px tall; in-panel actions and the limits toggle. Hover lifts to 12% fill and 34% edge; pressed takes a solid white edge on 14%.
- **Focus (all interactive elements):** 2px marking-white outline, offset 3px.

### Chips (surface tiles)
- **Style:** 60px dark glass tiles (16px radius, 62% night fill, 12px blur, 12% white hairline): a 24px stroked glyph in its class colour (sun, waves, snowflake, crystal), the class name in Barlow 600 15px and "Road" in 13px at 70% white.
- **State:** hover deepens the fill; preview (hover or focus while not pinned) draws a dashed 70% white edge and repaints the roads; pressed is a solid marking-white edge on a 12% fill and keeps the roads repainted. A polite live region announces the change.

### Navigation
- **Site bar:** night black at 82% with a 14px blur and an 8% white bottom hairline; on home a top-down gradient (92% → 55%).
- **Links:** Home, Demo, How it works, Study, Model; Barlow 400 15px at 70% white, white on hover, 600 and white on the current page. Each sits over a 3px lane marker: solid marking white under the current page, dashed a9 on hover. Model toggles the limits panel and reads as current while open.
- **Motion:** on a route change the lane marker morphs to the new link (380ms, ease-out) while the bar holds still.

### Steps (How it works)
A numbered list: each item opens with a 40px round numeral (1.5px white border, Barlow 600 16px), a 20px 600 title and muted copy. From 768px a 2px dashed 25% white connector runs from each numeral towards the next.

### Cards / Containers
- **Glass panel:** 12px radius, 6px padding, glass panel shadow.
- **Reading panel:** a2 at 80%, 10% white hairline, 12px radius, 20/28px padding.
- **Wash:** a2 fill, a4 border, 12px radius; quiet notices, errors, absent data.
- **Pending card:** 2px dashed 15% white border, 12px radius.

### Inputs / Fields
- **Analyse photo drop:** inside the 3:2 photo frame (12px radius on a2, give-way corners), an inset 8px zone with a 2px dashed 30% white border (60% on hover); drag-over turns it solid white on a 10% fill. While reading, a soft scan bar sweeps it in 1.2s.
- **Hero drop:** dragging a file over the hero shows a 12px dashed 70% white frame inset 16px on a 40% asphalt veil with "Drop to read this photo" in the hero subline.

### Verdict stamp (signature)
A confident verdict is a 2px solid white 12px plate with the class swatch and the class name in display lettering; an uncertain one is the same plate with a dashed a9 border. Both wipe in left to right like fresh paint (clip-path, 160ms).

### Aerial scene (signature)
A three.js scene whose ground is the drone photograph at its own colours (not tone-mapped), seen from almost straight above (fov 30). One surface layer covers the photo, masked to its roads by a soft road-shaped mask (both carriageways, 70px kerb fillets where the pavements round into the junction, 7px blur), so the layer has no seams. Lit by a dusk sky environment with warm lamps, a hemisphere and a low key light, and warm point lights at the photo's own street lamps. Per-class looks: dry shows the photo as taken; wet darkens with a clearcoat sheen that picks up the lamp pools; water is a near-black mirror with drifting ripples; fresh snow is pale, grained and self-lit so it reads white at dusk; melted snow is grey slush with partial sheen and ripples; ice is a pale glaze. Looks blend with an exponential ease (about 4/s). Until the scene is ready the still photo holds the space and crossfades out (700ms); if WebGL is missing or the scene fails, the photo stays and the page works.

### Motion
- **Camera:** settles over 1.8s (cubic ease-out) from 18% higher to the framing; leans up to 0.18 units with the pointer.
- **Route change:** View Transitions. Home, Analyse and Study run left to right like lanes; forward slides the old page out 28px left (220ms, accelerating) and the new page in from 28px right (380ms, ease-out); back mirrors it.
- **Small motion:** ruler marker 300ms; heatmap crossfade 180ms; verdict stamp 160ms; scan bar 1.2s.
- **Reduced motion:** route changes become a 140ms crossfade; every other animation collapses to near zero; the scene renders on demand with no settle, lean or ripple drift; How it works scrolls without smoothing.

## Do's and Don'ts

### Do:
- **Do** keep the home background a real photograph and change only its roads, through one soft road-shaped layer.
- **Do** keep hero copy on the left over dark gradients, with lavender reserved for the class words.
- **Do** take every neutral from the a0–a10 ramp, night black, or an alpha of them.
- **Do** use a solid marking-white edge or marker for committed state and a dashed one for candidates.
- **Do** give every class a second signal: a pattern on tints, a glyph and a name on hero tiles.
- **Do** use pills only for the hero and site-bar calls to action; cut working-screen controls as 6px plates and panels as 12px plates.
- **Do** set the hero in Barlow 700 with tight negative tracking, and working-screen titles and headings in Barlow Condensed.
- **Do** set measurements in Barlow with tabular figures, never below 12px.
- **Do** frame photographs on the working screens with give-way L corner marks.
- **Do** keep the hero working as a still photograph when WebGL is missing or fails.
- **Do** honour reduced motion: route changes crossfade briefly, everything else stills.

### Don't:
- **Don't** use temporary-works yellow for anything other than provisional / mock notices.
- **Don't** introduce hue into the chrome beyond the class tints, chip glyphs, lavender class words and provisional yellow.
- **Don't** set the lavender class words as a gradient.
- **Don't** add a kicker or eyebrow line above the hero title.
- **Don't** request a weight that is not loaded (Barlow 400/600/700, Barlow Condensed 600/700).
- **Don't** set HTML headings, captions or links in tracked capitals.
- **Don't** round working-screen controls into pills.
- **Don't** show a class by colour alone.
- **Don't** use pure white (#fff) or pure black for text or surfaces.
