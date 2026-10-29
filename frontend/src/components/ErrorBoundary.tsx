import { Component, type ReactNode } from "react";

// Last resort: a bug in the page must never leave a blank screen or a half-drawn verdict.
export class ErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <main className="mx-auto max-w-prose p-8" role="alert">
        <h1 className="display text-6xl">Something broke</h1>
        <p className="mt-4">
          The survey sheet hit an error and stopped. No result is shown. Reload the page to start
          again.
        </p>
        <button type="button" className="btn-primary mt-4" onClick={() => location.reload()}>
          Reload
        </button>
      </main>
    );
  }
}
