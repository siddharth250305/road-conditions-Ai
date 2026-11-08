import { percent } from "../format";
import { MODES, reasonText } from "../review";
import type { Review } from "../types";

// Under the reading: how much human review this photo gets, and why. Only shown when the
// server routes photos (it has a routing.json checked on the served model).
export function ReviewNote({ review }: { review: Review }) {
  const mode = MODES[review.mode];
  return (
    <div data-testid="review-note">
      <h2 className="heading mt-8 mb-3 text-muted">Human review</h2>
      <p className="m-0 flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="display text-[1.75rem]">{mode.name}</span>
        <span className="text-[15px]">{mode.what}</span>
      </p>
      <ul className="mt-2 mb-0 list-none space-y-1 p-0 text-[14px] leading-snug text-muted">
        {review.reasons.map((r) => (
          <li key={r}>{reasonText(r)}</li>
        ))}
      </ul>
      {review.confidence !== null && (
        <p className="mt-2 mb-0 text-[13px] text-muted">
          Review confidence <span className="data">{percent(review.confidence)}</span>, scaled with the
          temperature fitted for routing on the validation set.
        </p>
      )}
      {!review.recorded && (
        <p role="alert" className="wash mt-3 mb-0 px-4 py-3 text-[14px]">
          This decision could not be recorded, so it is not in the review queue.
        </p>
      )}
    </div>
  );
}
