/**
 * Copy the onnxruntime-web files the demo needs into docs/vendor/ort/.
 *
 * The demo makes no external requests at runtime, so the runtime is vendored
 * rather than pulled from a CDN. Only the jsep build is copied: it carries
 * both the WebGPU and WASM execution providers, so the WASM-only build would
 * be dead weight.
 *
 *   npm install && node scripts/vendor_ort.mjs
 */

import { copyFileSync, mkdirSync, existsSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const SRC = join(ROOT, 'node_modules', 'onnxruntime-web', 'dist');
const DEST = join(ROOT, 'docs', 'vendor', 'ort');

const FILES = [
  'ort.webgpu.min.mjs',
  'ort-wasm-simd-threaded.jsep.mjs',
  'ort-wasm-simd-threaded.jsep.wasm',
];

if (!existsSync(SRC)) {
  console.error('onnxruntime-web is not installed. Run: npm install');
  process.exit(1);
}

mkdirSync(DEST, { recursive: true });
let total = 0;
for (const name of FILES) {
  const from = join(SRC, name);
  if (!existsSync(from)) {
    console.error(`missing ${name} in ${SRC}`);
    process.exit(1);
  }
  copyFileSync(from, join(DEST, name));
  const size = statSync(from).size;
  total += size;
  console.log(`  ${name}  ${(size / 1048576).toFixed(1)} MB`);
}
console.log(`vendored ${FILES.length} files (${(total / 1048576).toFixed(1)} MB) into docs/vendor/ort/`);
console.log('Model: copy models/yolov8n.onnx into docs/models/ (see scripts/fetch_models.py)');
