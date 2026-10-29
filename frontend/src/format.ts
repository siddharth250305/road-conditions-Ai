export function percent(p: number, digits = 0): string {
  return `${(p * 100).toFixed(digits)}%`;
}

export function map(value: number | null | undefined): string {
  return value == null ? "n/a" : value.toFixed(3);
}

export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
