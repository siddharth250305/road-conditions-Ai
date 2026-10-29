import { SAMPLES, type Sample } from "../samples";

export function SampleStrip({ onPick }: { onPick: (sample: Sample) => void }) {
  return (
    <section aria-labelledby="samples-heading" className="mt-6">
      <h2 id="samples-heading" className="caption mb-2">
        Or try a sample
      </h2>
      <ul className="m-0 grid list-none grid-cols-3 gap-3 p-0">
        {SAMPLES.map((sample) => (
          <li key={sample.id}>
            <button
              type="button"
              onClick={() => onPick(sample)}
              className="block w-full cursor-pointer border-2 border-rule bg-transparent p-1 text-left text-text"
            >
              <img
                src={sample.url}
                alt=""
                className="aspect-[3/2] w-full object-cover"
                loading="lazy"
              />
              <span className="data mt-1 block">{sample.caption}</span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
