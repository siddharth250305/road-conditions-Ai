import { SAMPLES, type Sample } from "../samples";

export function SampleStrip({ onPick }: { onPick: (sample: Sample) => void }) {
  return (
    <section aria-labelledby="samples-heading" className="mt-6 px-3">
      <h2 id="samples-heading" className="heading m-0 mb-3 text-muted">
        Or try a sample
      </h2>
      <ul className="m-0 grid list-none grid-cols-3 gap-3 p-0">
        {SAMPLES.map((sample) => (
          <li key={sample.id}>
            <button
              type="button"
              onClick={() => onPick(sample)}
              className="glass block w-full cursor-pointer overflow-hidden rounded-[var(--r-panel)] p-1.5 text-left text-text transition-colors hover:border-a10/40"
            >
              <img src={sample.url} alt="" className="aspect-[5/2] w-full rounded-[4px] object-cover" loading="lazy" />
              <span className="block px-1.5 pt-2 pb-1 text-[12px] text-muted">{sample.caption}</span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
