import { useCallback, useEffect, useState, type FormEvent } from "react";
import { fetchQueue, fetchReviewPhoto, ReviewError, submitReview } from "../api";
import { CLASS_ORDER, prettyClass, tintClass } from "../classes";
import { percent } from "../format";
import { loadToken, MODES, reasonText, saveToken } from "../review";
import type { Health, ReviewItem, ReviewMode, ReviewQueue } from "../types";

interface Props {
  active: boolean;
  health: Health | null;
}

const message = (err: unknown) => (err instanceof ReviewError ? err.message : "Something went wrong.");

function ReviewPhoto({ token, id }: { token: string; id: string }) {
  const [url, setUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let live = true;
    let made: string | null = null;
    fetchReviewPhoto(token, id)
      .then((blob) => {
        if (!live) return;
        made = URL.createObjectURL(blob);
        setUrl(made);
      })
      .catch(() => live && setFailed(true));
    return () => {
      live = false;
      if (made) URL.revokeObjectURL(made);
    };
  }, [token, id]);
  if (failed) return <p className="wash m-0 px-4 py-3 text-[14px]">The photo could not be loaded.</p>;
  return (
    <div className="aspect-[3/2] overflow-hidden rounded-[8px] bg-a1">
      {url && <img src={url} alt="The uploaded road photo" className="h-full w-full object-cover" />}
    </div>
  );
}

function ReviewCard({ item, token, onDone }: { item: ReviewItem; token: string; onDone: () => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mode = MODES[item.mode];
  const label = async (cls: string) => {
    setBusy(true);
    setError(null);
    try {
      await submitReview(token, item.decision_id, cls);
      onDone();
    } catch (err) {
      setError(message(err));
      if (err instanceof ReviewError && (err.kind === "taken" || err.kind === "gone")) onDone();
    } finally {
      setBusy(false);
    }
  };
  const progress =
    item.reviews_needed > 1
      ? `${item.reviewers.length} of ${item.reviews_needed} reviews${item.status === "disputed" ? ", reviewers disagree" : ""}`
      : null;
  return (
    <li
      data-testid="review-card"
      className="rounded-[var(--r-panel)] border border-a10/10 bg-a2/80 p-5"
      aria-label={`${mode.name}: model says ${item.prediction ? prettyClass(item.prediction) : "nothing usable"}`}
    >
      {item.has_image ? (
        <ReviewPhoto token={token} id={item.decision_id} />
      ) : (
        <p className="wash m-0 px-4 py-3 text-[14px]">This deployment does not keep photos, so only the uploader saw it.</p>
      )}
      <p className="mt-4 mb-0 flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="display text-[1.5rem]">{mode.name}</span>
        <span className="text-[14px] text-muted">{mode.what}</span>
      </p>
      {progress && <p className="mt-1 mb-0 text-[14px] font-semibold">{progress}</p>}
      <p className="mt-2 mb-0 text-[15px]">
        Model says{" "}
        <strong className="font-semibold first-letter:uppercase">
          {item.prediction ? prettyClass(item.prediction) : "nothing usable"}
        </strong>
        {item.confidence !== null && <span className="data ml-2 text-muted">{percent(item.confidence)}</span>}
      </p>
      <ul className="mt-1 mb-0 list-none space-y-0.5 p-0 text-[13px] text-muted">
        {item.reasons.map((r) => (
          <li key={r}>{reasonText(r)}</li>
        ))}
      </ul>
      <div role="group" aria-label="What is on the road?" className="mt-4 flex flex-wrap gap-2">
        {CLASS_ORDER.map((cls) => (
          <button key={cls} type="button" className="btn-quiet" disabled={busy} onClick={() => void label(cls)}>
            <span className={`mr-2 inline-block h-3.5 w-3.5 rounded-[3px] border border-a0/40 align-[-2px] ${tintClass(cls)}`} aria-hidden="true" />
            <span className="first-letter:uppercase">{prettyClass(cls)}</span>
            {cls === item.prediction && <span className="ml-1.5 text-muted">(model)</span>}
          </button>
        ))}
      </div>
      {error && (
        <p role="alert" className="mt-3 mb-0 text-[14px]">
          {error}
        </p>
      )}
    </li>
  );
}

const MODE_ORDER: ReviewMode[] = ["second_review", "full_review", "light_review"];

// Reviewers only: the open decisions this reviewer has not reviewed yet, strictest first.
export function ReviewScreen({ active, health }: Props) {
  const [token, setToken] = useState(loadToken);
  const [draft, setDraft] = useState("");
  const [queue, setQueue] = useState<ReviewQueue | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async (t: string) => {
    setLoading(true);
    setError(null);
    try {
      setQueue(await fetchQueue(t));
    } catch (err) {
      setQueue(null);
      setError(message(err));
      if (err instanceof ReviewError && err.kind === "token") {
        saveToken("");
        setToken("");
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (active && token) void load(token);
  }, [active, token, load]);

  const signIn = (e: FormEvent) => {
    e.preventDefault();
    const t = draft.trim();
    if (!t) return;
    saveToken(t);
    setDraft("");
    setToken(t);
  };
  const signOut = () => {
    saveToken("");
    setToken("");
    setQueue(null);
  };

  const off = health?.routing != null && !health.routing.enabled;

  return (
    <>
      <div className="mb-6 flex flex-wrap items-end justify-between gap-x-10 gap-y-3 px-3">
        <h1 className="display m-0 text-[clamp(2rem,3.4vw,3rem)]">Review queue</h1>
        <p className="m-0 max-w-[52ch] text-[15px] text-muted">
          Photos the model was not sure enough about, or read as ice, wait here for a person. Second
          reviews need two reviewers who agree; you never see the other reviewer's answer.
        </p>
      </div>

      {off && (
        <p className="wash mx-3 px-4 py-3 text-[15px]">Review routing is off on this deployment, so there is nothing to review.</p>
      )}

      {!off && !token && (
        <form onSubmit={signIn} className="wash mx-3 max-w-[34rem] p-5">
          <label htmlFor="reviewer-token" className="heading mb-2 block">
            Reviewer token
          </label>
          <div className="flex flex-wrap gap-2">
            <input
              id="reviewer-token"
              type="password"
              autoComplete="off"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              className="h-11 min-w-0 flex-1 rounded-[6px] border border-a10/16 bg-a1 px-3 text-[15px] text-text"
            />
            <button type="submit" className="btn-solid">
              Open queue
            </button>
          </div>
          <p className="mt-2 mb-0 text-[13px] text-muted">Kept for this browser tab only.</p>
          {error && (
            <p role="alert" className="mt-3 mb-0 text-[14px]">
              {error}
            </p>
          )}
        </form>
      )}

      {!off && token && (
        <section aria-label="Queue" className="px-3">
          <div className="mb-5 flex flex-wrap items-center gap-x-4 gap-y-2">
            {queue && (
              <p className="m-0 text-[15px]">
                Signed in as <strong className="font-semibold">{queue.reviewer}</strong>
              </p>
            )}
            {queue && (
              <p className="data m-0 text-muted" data-testid="queue-counts">
                {MODE_ORDER.map((m) => `${MODES[m].name} ${queue.counts[m]?.open ?? 0} open`).join(" · ")}
              </p>
            )}
            <span className="flex gap-2 sm:ml-auto">
              <button type="button" className="btn-quiet" onClick={() => void load(token)} disabled={loading}>
                Refresh
              </button>
              <button type="button" className="btn-quiet" onClick={signOut}>
                Sign out
              </button>
            </span>
          </div>
          {error && (
            <p role="alert" className="wash mb-4 px-4 py-3 text-[15px]">
              {error}
            </p>
          )}
          {loading && !queue && <p className="text-muted">Loading the queue…</p>}
          {queue && queue.items.length === 0 && (
            <p className="wash px-4 py-3 text-[15px]">Nothing is waiting for you.</p>
          )}
          {queue && queue.items.length > 0 && (
            <ul className="m-0 grid list-none gap-6 p-0 lg:grid-cols-2">
              {queue.items.map((item) => (
                <ReviewCard key={item.decision_id} item={item} token={token} onDone={() => void load(token)} />
              ))}
            </ul>
          )}
        </section>
      )}
    </>
  );
}
