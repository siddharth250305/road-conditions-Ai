import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

// jsdom has no object URLs.
let counter = 0;
URL.createObjectURL = () => `blob:test/${++counter}`;
URL.revokeObjectURL = () => {};
