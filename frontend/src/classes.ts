// Class order follows the PRD list (and configs/default.yaml), which is how the ruler reads
// from dry to ice. It is a display order, not a measured friction ranking.

export const CLASS_ORDER = ["dry", "wet", "water", "fresh_snow", "melted_snow", "ice"] as const;
export type ClassName = (typeof CLASS_ORDER)[number];

export function classIndex(name: string): number {
  return CLASS_ORDER.indexOf(name as ClassName);
}

export function prettyClass(name: string): string {
  return name.replace(/_/g, " ");
}

export function tintClass(name: string): string {
  return classIndex(name) >= 0 ? `tint-${name}` : "";
}
