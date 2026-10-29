import { useCallback, useEffect, useReducer, useRef, useState, useSyncExternalStore } from "react";
import { ApiError, fetchHealth, fetchSample, fetchStudy, predict, type StudyResult } from "./api";
import { initialState, reducer } from "./state";
import type { Health } from "./types";

export function usePrediction() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const current = useRef(0); // only the newest request may change the page
  const lastUrl = useRef<string | null>(null);
  const retryWith = useRef<(() => void) | null>(null);

  const run = useCallback(
    async (name: string, previewUrl: string, getBlob: () => Promise<Blob>) => {
      const id = ++current.current;
      if (lastUrl.current?.startsWith("blob:") && lastUrl.current !== previewUrl) {
        URL.revokeObjectURL(lastUrl.current);
      }
      lastUrl.current = previewUrl;
      dispatch({ type: "start", name, previewUrl });
      try {
        const blob = await getBlob();
        const result = await predict(blob, name);
        if (id === current.current) dispatch({ type: "success", result });
      } catch (err) {
        const error = err instanceof ApiError ? err : new ApiError("server");
        if (id === current.current) dispatch({ type: "failure", error });
      }
    },
    [],
  );

  const submitFile = useCallback(
    (file: File) => {
      const url = URL.createObjectURL(file);
      const go = () => void run(file.name, url, async () => file);
      retryWith.current = go;
      go();
    },
    [run],
  );

  const submitSample = useCallback(
    (url: string, name: string) => {
      const go = () => void run(name, url, () => fetchSample(url));
      retryWith.current = go;
      go();
    },
    [run],
  );

  const retry = useCallback(() => retryWith.current?.(), []);
  const reset = useCallback(() => {
    current.current++;
    retryWith.current = null;
    dispatch({ type: "reset" });
  }, []);

  return { state, submitFile, submitSample, retry, reset };
}

function subscribeOnline(callback: () => void) {
  window.addEventListener("online", callback);
  window.addEventListener("offline", callback);
  return () => {
    window.removeEventListener("online", callback);
    window.removeEventListener("offline", callback);
  };
}

export function useOnline(): boolean {
  return useSyncExternalStore(
    subscribeOnline,
    () => navigator.onLine,
    () => true,
  );
}

/** What the service says about itself once on load; `reachable` is false when it cannot be asked. */
export function useHealth(): { health: Health | null; reachable: boolean | null } {
  const [value, setValue] = useState<{ health: Health | null; reachable: boolean | null }>({
    health: null,
    reachable: null,
  });
  useEffect(() => {
    let live = true;
    fetchHealth()
      .then((health) => live && setValue({ health, reachable: true }))
      .catch(() => live && setValue({ health: null, reachable: false }));
    return () => {
      live = false;
    };
  }, []);
  return value;
}

export type StudyState =
  | { status: "loading" }
  | { status: "ok"; result: StudyResult }
  | { status: "error"; message: string };

export function useStudy(enabled: boolean): StudyState {
  const [state, setState] = useState<StudyState>({ status: "loading" });
  useEffect(() => {
    if (!enabled) return;
    let live = true;
    fetchStudy()
      .then((result) => live && setState({ status: "ok", result }))
      .catch((err: unknown) => {
        const message = err instanceof ApiError ? err.message : "The study could not be loaded.";
        if (live) setState({ status: "error", message });
      });
    return () => {
      live = false;
    };
  }, [enabled]);
  return state;
}
