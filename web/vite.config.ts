import { defineConfig } from "vite";
import { youtubeBridge } from "./server/youtube.mjs";

// Cross-origin isolation enables SharedArrayBuffer, which ONNX Runtime Web needs for
// multi-threaded WASM. Without these headers it silently falls back to one thread.
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  plugins: [youtubeBridge()],
  server: { headers: isolation },
  preview: { headers: isolation },
  worker: { format: "es" },
  build: {
    target: "es2022",
    rollupOptions: { input: { main: "index.html", bench: "bench.html" } },
  },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
