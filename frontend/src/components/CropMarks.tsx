// Give-way L marks at the corners of the photo frame, as painted at the junction in the
// reference photo. Drawn with positioned boxes so they stay crisp at any frame size.
const CORNERS = [
  { v: "top", h: "left" },
  { v: "top", h: "right" },
  { v: "bottom", h: "left" },
  { v: "bottom", h: "right" },
] as const;

export function CropMarks() {
  return (
    <div aria-hidden="true" className="pointer-events-none absolute -inset-3 text-a10/70">
      {CORNERS.map(({ v, h }) => (
        <span key={`${v}-${h}`}>
          <span className="absolute h-[3px] w-6 rounded-[1px] bg-current" style={{ [v]: 0, [h]: 0 }} />
          <span className="absolute h-6 w-[3px] rounded-[1px] bg-current" style={{ [v]: 0, [h]: 0 }} />
        </span>
      ))}
    </div>
  );
}
