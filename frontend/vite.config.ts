import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// In development the API runs separately (`uvicorn api.main:app`); in production the same
// FastAPI service serves the built files, so the page and the API share one origin.
const api = process.env.ROADCOND_API ?? "http://127.0.0.1:9000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: { "/predict": api, "/study": api, "/health": api },
  },
  // The three.js scene is a lazy chunk loaded only by the home screen, so its size does not
  // delay the Report or Study screens.
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 1000 },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
