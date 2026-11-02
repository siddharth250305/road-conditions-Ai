// A small authored icon set: 20 px grid, 1.6 stroke, round caps. Decorative only; every control
// that carries one also carries its name as text.
import type { SVGProps } from "react";

function Icon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...props}
    />
  );
}

/** The mark: the crossroads from above, two roads with a zebra crossing at each mouth. */
export function MarkIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg width="30" height="30" viewBox="0 0 30 30" aria-hidden="true" focusable="false" {...props}>
      <rect x="0.5" y="0.5" width="29" height="29" rx="6" fill="#151a17" stroke="rgb(236 238 233 / 0.28)" />
      <rect x="11" y="1" width="8" height="28" fill="#252c28" />
      <rect x="1" y="11" width="28" height="8" fill="#252c28" />
      <g fill="#eceee9">
        {[12.2, 14.3, 16.4].map((x) => (
          <rect key={`n${x}`} x={x} y="6.6" width="1.4" height="3.2" rx="0.3" />
        ))}
        {[12.2, 14.3, 16.4].map((x) => (
          <rect key={`s${x}`} x={x} y="20.2" width="1.4" height="3.2" rx="0.3" />
        ))}
        {[12.2, 14.3, 16.4].map((y) => (
          <rect key={`w${y}`} x="6.6" y={y} width="3.2" height="1.4" rx="0.3" />
        ))}
        {[12.2, 14.3, 16.4].map((y) => (
          <rect key={`e${y}`} x="20.2" y={y} width="3.2" height="1.4" rx="0.3" />
        ))}
      </g>
    </svg>
  );
}

export function ChevronIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Icon {...props}>
      <path d="M5.5 8l4.5 4.5L14.5 8" />
    </Icon>
  );
}

export function UploadIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Icon {...props}>
      <path d="M10 13V3.5" />
      <path d="M6 7.5l4-4 4 4" />
      <path d="M3.5 13v2.5a1.5 1.5 0 001.5 1.5h10a1.5 1.5 0 001.5-1.5V13" />
    </Icon>
  );
}

export function PlayIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" focusable="false" {...props}>
      <path d="M4 2.6v10.8a.6.6 0 00.9.5l8.4-5.4a.6.6 0 000-1L4.9 2.1a.6.6 0 00-.9.5z" fill="currentColor" />
    </svg>
  );
}

/** The GitHub mark, used only to label links to the project's repository. */
export function GitHubIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg width="20" height="20" viewBox="0 0 16 16" aria-hidden="true" focusable="false" {...props}>
      <path
        fill="currentColor"
        d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"
      />
    </svg>
  );
}

/** Surface glyphs for the hero chips: one consistent 1.6 stroke on a 20 px grid. */
export function SunIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Icon {...props}>
      <circle cx="10" cy="10" r="3.4" />
      <path d="M10 2.5v1.8M10 15.7v1.8M2.5 10h1.8M15.7 10h1.8M4.7 4.7l1.3 1.3M14 14l1.3 1.3M4.7 15.3L6 14M14 6l1.3-1.3" />
    </Icon>
  );
}

export function WavesIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Icon {...props}>
      <path d="M2.5 7c1.25-1.2 2.5-1.2 3.75 0s2.5 1.2 3.75 0 2.5-1.2 3.75 0 2.5 1.2 3.75 0" />
      <path d="M2.5 11c1.25-1.2 2.5-1.2 3.75 0s2.5 1.2 3.75 0 2.5-1.2 3.75 0 2.5 1.2 3.75 0" />
      <path d="M2.5 15c1.25-1.2 2.5-1.2 3.75 0s2.5 1.2 3.75 0 2.5-1.2 3.75 0 2.5 1.2 3.75 0" />
    </Icon>
  );
}

export function SnowflakeIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Icon {...props}>
      <path d="M10 2v16M3.1 6l13.8 8M3.1 14l13.8-8" />
      <path d="M8 3.6L10 5l2-1.4M8 16.4L10 15l2 1.4M3.6 8.6l2.3.4.6-2.3M16.4 11.4l-2.3-.4-.6 2.3M3.6 11.4l2.3-.4.6 2.3M16.4 8.6l-2.3.4-.6-2.3" />
    </Icon>
  );
}

export function IceIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Icon {...props}>
      <path d="M10 2.5l6 4.2v6.6l-6 4.2-6-4.2V6.7z" />
      <path d="M10 2.5v15M4 6.7l12 6.6M16 6.7L4 13.3" strokeOpacity={0.55} />
    </Icon>
  );
}
