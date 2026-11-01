// Copies runtime assets into public/ so the app runs fully on-device with no CDN:
// model weights (from ../models, fetched by scripts/download_models.py), the ONNX
// Runtime Web wasm binaries, and the MediaPipe Tasks wasm fileset.
import { copyFileSync, existsSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const web = join(here, "..");
const pub = join(web, "public");

const MODELS = [
  "face_detection_yunet_2023mar_dyn.onnx",
  "scrfd_500m_kps.onnx",
  "blaze_face_short_range.tflite",
  "blaze_face_full_range.tflite",
];

function copyDir(src, dst, filter = () => true) {
  mkdirSync(dst, { recursive: true });
  for (const f of readdirSync(src)) if (filter(f)) copyFileSync(join(src, f), join(dst, f));
}

mkdirSync(join(pub, "models"), { recursive: true });
let missing = 0;
for (const m of MODELS) {
  const src = join(web, "..", "models", m);
  if (!existsSync(src)) {
    console.warn(`missing ${src} -- run: python scripts/download_models.py`);
    missing++;
    continue;
  }
  copyFileSync(src, join(pub, "models", m));
}
copyDir(join(web, "node_modules", "onnxruntime-web", "dist"), join(pub, "ort"), (f) => /^ort-wasm.*\.(wasm|mjs)$/.test(f));
copyDir(join(web, "node_modules", "@mediapipe", "tasks-vision", "wasm"), join(pub, "mediapipe"));
console.log(`assets ready (${MODELS.length - missing}/${MODELS.length} models)`);
