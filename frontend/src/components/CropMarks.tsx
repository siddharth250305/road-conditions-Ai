// Printer's crop marks at the corners and registration ticks along the edges of a specimen.
// Drawn with positioned boxes so they stay crisp at any frame size.
const CORNERS = [
  { v: "top", h: "left" },
  { v: "top", h: "right" },
  { v: "bottom", h: "left" },
  { v: "bottom", h: "right" },
] as const;

const TICKS_X =
  "repeating-linear-gradient(90deg, currentColor 0 1px, transparent 1px calc(10% - 0.1px))";
const TICKS_Y =
  "repeating-linear-gradient(180deg, currentColor 0 1px, transparent 1px calc(10% - 0.1px))";

export function CropMarks() {
  return (
    <div aria-hidden="true" className="pointer-events-none absolute inset-0 text-text">
      {CORNERS.map(({ v, h }) => (
        <span key={`${v}-${h}`}>
          <span
            className="absolute h-0.5 w-4 bg-current"
            style={{ [v]: -6, [h]: -22 }}
          />
          <span
            className="absolute h-4 w-0.5 bg-current"
            style={{ [v]: -22, [h]: -6 }}
          />
        </span>
      ))}
      <span className="absolute -top-2.5 right-0 left-0 h-1.5" style={{ backgroundImage: TICKS_X }} />
      <span className="absolute right-0 -bottom-2.5 left-0 h-1.5" style={{ backgroundImage: TICKS_X }} />
      <span className="absolute top-0 bottom-0 -left-2.5 w-1.5" style={{ backgroundImage: TICKS_Y }} />
      <span className="absolute top-0 -right-2.5 bottom-0 w-1.5" style={{ backgroundImage: TICKS_Y }} />
    </div>
  );
}
