/**
 * Browser demo wiring: camera, inference loop, UI, audio and haptics.
 *
 * Detection and rendering run on separate loops. WASM inference costs a few
 * hundred milliseconds a frame, so driving the display from it would give a
 * juddering three-frames-a-second picture; instead the renderer runs on
 * requestAnimationFrame against whatever the detector last produced, exactly
 * as the desktop build coasts on the tracker between detection passes.
 */

import { BBox } from './geometry.mjs';
import { YoloDetector } from './detector.mjs';
import {
  enrich, focalLengthPx, PROXIMITY_NAME, Proximity, Zone,
} from './spatial.mjs';
import { IoUTracker } from './tracking.mjs';
import { AlertPolicy } from './alerts.mjs';
import { strongestCue } from './haptics.mjs';

const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const dpr = Math.min(window.devicePixelRatio || 1, 2);

const PROX = {
  IMMEDIATE: { c: '#f43f5e', n: 'immediate', f: 880, pulses: 3 },
  NEAR: { c: '#fbbf24', n: 'near', f: 660, pulses: 2 },
  MODERATE: { c: '#22d3ee', n: 'moderate', f: 440, pulses: 1 },
  FAR: { c: '#64748b', n: 'far', f: 0, pulses: 0 },
};
const HFOV_DEG = 65;
const CENTER_FRACTION = 0.34;

const state = {
  running: false, mirror: true, audio: true, voice: true, boxes: true,
  detections: [], haptics: null, fps: 0, transcript: [], lastAlertKey: '',
};

let detector = null;
let tracker = new IoUTracker();
let policy = null;
const smooth = new Map();
const lastSeen = new Map();

const cam = $('cam');
const view = $('view');

/* ── canvas helper ─────────────────────────────────────── */
function fitCanvas(cv, w, h) {
  if (cv.width !== w * dpr || cv.height !== h * dpr) {
    cv.width = w * dpr; cv.height = h * dpr;
    cv.style.width = `${w}px`; cv.style.height = `${h}px`;
  }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

/* ── audio ─────────────────────────────────────────────── */
let actx = null; let master = null; let analyser = null; let waveData = null;

function initAudio() {
  if (actx) return;
  actx = new (window.AudioContext || window.webkitAudioContext)();
  master = actx.createGain(); master.gain.value = 0.34;
  analyser = actx.createAnalyser(); analyser.fftSize = 1024;
  waveData = new Uint8Array(analyser.frequencyBinCount);
  master.connect(analyser); analyser.connect(actx.destination);
}

/** Pitch and pulse count encode urgency; stereo pan encodes bearing. */
function earcon(proximityName, zone) {
  if (!state.audio || !actx) return;
  const p = PROX[proximityName];
  if (!p || !p.f) return;
  const pan = { left: -0.7, center: 0, right: 0.7 }[zone] ?? 0;
  const t0 = actx.currentTime; const dur = 0.075; const gap = 0.05;

  for (let i = 0; i < p.pulses; i += 1) {
    const st = t0 + i * (dur + gap);
    const osc = actx.createOscillator(); osc.type = 'sine'; osc.frequency.value = p.f;
    const g = actx.createGain();
    // Short ramps: a hard start or stop clicks audibly.
    g.gain.setValueAtTime(0, st);
    g.gain.linearRampToValueAtTime(1, st + 0.008);
    g.gain.setValueAtTime(1, st + dur - 0.008);
    g.gain.linearRampToValueAtTime(0, st + dur);
    const pn = actx.createStereoPanner(); pn.pan.value = pan;
    osc.connect(g); g.connect(pn); pn.connect(master);
    osc.start(st); osc.stop(st + dur + 0.01);
  }
}

function speak(text) {
  if (!state.voice || !state.audio || !window.speechSynthesis) return;
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.15;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(u);
}

/** Real vibration where the device has a motor; the meters show it regardless. */
function vibrate(cue) {
  if (!navigator.vibrate || !cue.active) return;
  const ms = Math.round(30 + 70 * cue.intensity);
  navigator.vibrate(ms);
}

/* ── camera + model ────────────────────────────────────── */
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
    audio: false,
  });
  cam.srcObject = stream;
  await cam.play();
  await new Promise((res) => {
    if (cam.videoWidth) res();
    else cam.onloadedmetadata = () => res();
  });
}

async function loadModel(onProgress) {
  const ort = await import('../vendor/ort/ort.webgpu.min.mjs');
  // Absolute URL, resolved from this module rather than from the page or from
  // ort's own location: a relative path is taken as relative to the runtime
  // bundle (giving /vendor/vendor/...), and a root-relative one breaks when
  // GitHub Pages serves the site from a /repo-name/ subpath.
  ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
  // GitHub Pages cannot send the cross-origin isolation headers that
  // SharedArrayBuffer needs, so threading is unavailable; ask for one thread
  // rather than let the runtime fail trying to spawn workers.
  ort.env.wasm.numThreads = 1;
  ort.env.logLevel = 'error';

  const modelUrl = new URL('../models/yolov8n.onnx', import.meta.url).href;

  // WebGPU is far faster than WASM, but is unavailable on Safari, older
  // Firefox, and any machine without a usable GPU. Probing for an actual
  // adapter — rather than merely for `navigator.gpu` — matters twice over:
  // the object exists in environments that cannot supply an adapter, and
  // requesting webgpu anyway leaves ORT to fall back internally without
  // telling us, so the UI would claim a backend it is not using.
  let hasGpu = false;
  try {
    hasGpu = Boolean(navigator.gpu && await navigator.gpu.requestAdapter());
  } catch { hasGpu = false; }

  const providers = hasGpu ? ['webgpu', 'wasm'] : ['wasm'];
  if (!hasGpu && onProgress) onProgress('no GPU — using WASM (slower)');

  try {
    return await YoloDetector.create(ort, modelUrl, { confidence: 0.45, providers }, onProgress);
  } catch (err) {
    if (!hasGpu) throw err;
    console.warn('WebGPU session failed, falling back to WASM', err);
    if (onProgress) onProgress('WebGPU failed — using WASM (slower)');
    return YoloDetector.create(ort, modelUrl, { confidence: 0.45, providers: ['wasm'] }, onProgress);
  }
}

/* ── detection loop ────────────────────────────────────── */
async function detectLoop() {
  let last = performance.now();
  while (state.running) {
    if (!cam.videoWidth) { await new Promise((r) => setTimeout(r, 60)); continue; }

    const w = cam.videoWidth; const h = cam.videoHeight;
    const focal = focalLengthPx(w, HFOV_DEG);

    let raw;
    try {
      raw = await detector.detect(cam, w, h);
    } catch (err) {
      console.error('inference failed', err);
      await new Promise((r) => setTimeout(r, 250));
      continue;
    }

    // The picture is mirrored for display, so mirror the boxes too before any
    // spatial reasoning: otherwise "on your left" comes out backwards.
    if (state.mirror) {
      raw = raw.map((r) => ({
        ...r,
        bbox: new BBox(w - r.bbox.x2, r.bbox.y, r.bbox.w, r.bbox.h),
      }));
    }

    const enriched = raw.map((r) => enrich(r.bbox, r.label, r.confidence, w, h, focal, CENTER_FRACTION));
    state.detections = tracker.update(enriched);

    const announcement = policy.select(state.detections, (id) => tracker.hitsFor(id));
    if (announcement) {
      const key = `${announcement.text}|${performance.now().toFixed(0)}`;
      if (key !== state.lastAlertKey) {
        state.lastAlertKey = key;
        const name = PROXIMITY_NAME[announcement.proximity];
        earcon(name, announcement.zone);
        setTimeout(() => speak(announcement.text), 130);
        state.transcript.push({
          text: announcement.text, proximity: name, zone: announcement.zone,
          distanceM: announcement.distanceM, at: Date.now() / 1000,
        });
        state.transcript = state.transcript.slice(-12);
      }
    }

    state.haptics = strongestCue(state.detections);
    vibrate(state.haptics);

    const now = performance.now();
    const dt = (now - last) / 1000; last = now;
    if (dt > 0) state.fps = state.fps === 0 ? 1 / dt : 0.8 * state.fps + 0.2 * (1 / dt);

    updatePanels();
    // Yield so the renderer and the UI thread get a turn.
    await new Promise((r) => setTimeout(r, 0));
  }
}

/* ── camera view ───────────────────────────────────────── */
function drawBrackets(ctx, x, y, w, h, color, lw) {
  const L = clamp(Math.min(w, h) * 0.24, 9, 30);
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(x, y + L); ctx.lineTo(x, y); ctx.lineTo(x + L, y);
  ctx.moveTo(x + w - L, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + L);
  ctx.moveTo(x + w, y + h - L); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - L, y + h);
  ctx.moveTo(x + L, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - L);
  ctx.stroke();
}

function renderView() {
  requestAnimationFrame(renderView);
  if (!state.running || !cam.videoWidth) return;

  const W = cam.videoWidth; const H = cam.videoHeight;
  const box = view.parentElement.getBoundingClientRect();
  const scale = Math.min((box.width - 16) / W, (box.height - 16) / H);
  const dw = Math.max(80, W * scale); const dh = Math.max(60, H * scale);
  const ctx = fitCanvas(view, dw, dh);
  const k = dw / W;

  ctx.clearRect(0, 0, dw, dh);
  ctx.save();
  if (state.mirror) { ctx.translate(dw, 0); ctx.scale(-1, 1); }
  ctx.drawImage(cam, 0, 0, dw, dh);
  ctx.restore();

  $('nosig').classList.add('hide');
  if (!state.boxes) return;

  const half = CENTER_FRACTION * dw / 2; const mid = dw / 2;
  ctx.save();
  ctx.strokeStyle = 'rgba(203,217,230,.16)'; ctx.lineWidth = 1; ctx.setLineDash([5, 7]);
  [mid - half, mid + half].forEach((x) => {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, dh); ctx.stroke();
  });
  ctx.restore();

  ctx.save();
  ctx.strokeStyle = 'rgba(34,211,238,.35)'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(mid - 9, dh / 2); ctx.lineTo(mid + 9, dh / 2);
  ctx.moveTo(mid, dh / 2 - 9); ctx.lineTo(mid, dh / 2 + 9);
  ctx.stroke(); ctx.restore();

  const now = performance.now();
  for (const d of state.detections) {
    const key = d.trackId != null ? `t${d.trackId}` : `${d.label}${d.bbox.x}`;
    const tgt = { x: d.bbox.x * k, y: d.bbox.y * k, w: d.bbox.w * k, h: d.bbox.h * k };
    let s = smooth.get(key);
    // Detection is far slower than the display, so ease boxes toward their
    // new position instead of teleporting once every few frames.
    if (!s) s = { ...tgt };
    else {
      const a = 0.25;
      s = { x: s.x + (tgt.x - s.x) * a, y: s.y + (tgt.y - s.y) * a,
        w: s.w + (tgt.w - s.w) * a, h: s.h + (tgt.h - s.h) * a };
    }
    smooth.set(key, s); lastSeen.set(key, now);

    const p = PROX[PROXIMITY_NAME[d.proximity]];
    const urgent = d.proximity === Proximity.IMMEDIATE;

    ctx.save();
    if (urgent) { ctx.shadowColor = p.c; ctx.shadowBlur = 14; }
    ctx.strokeStyle = p.c; ctx.lineWidth = urgent ? 2 : 1.25; ctx.globalAlpha = 0.55;
    ctx.strokeRect(s.x, s.y, s.w, s.h);
    ctx.globalAlpha = urgent ? 0.8 + 0.2 * Math.sin(now / 140) : 1;
    drawBrackets(ctx, s.x, s.y, s.w, s.h, p.c, urgent ? 2.6 : 2);
    ctx.restore();

    const dist = d.distanceM != null ? `${d.truncated ? '<' : '~'}${d.distanceM.toFixed(1)}m` : '—';
    const txt = `${d.label.toUpperCase()}  ${dist}`;
    ctx.font = '600 10px ui-monospace,Menlo,Consolas,monospace';
    const tw = ctx.measureText(txt).width + 14;
    const ly = s.y > 20 ? s.y - 18 : s.y + 4;
    const lx = clamp(s.x, 0, dw - tw);
    ctx.fillStyle = p.c; ctx.globalAlpha = 0.92;
    ctx.beginPath(); ctx.roundRect(lx, ly, tw, 15, 4); ctx.fill();
    ctx.globalAlpha = 1; ctx.fillStyle = '#04060a';
    ctx.fillText(txt, lx + 7, ly + 11);
  }

  for (const [k2, t] of lastSeen) {
    if (now - t > 1500) { lastSeen.delete(k2); smooth.delete(k2); }
  }
}

/* ── radar ─────────────────────────────────────────────── */
const ZONE_ANGLE = { left: -0.62, center: 0, right: 0.62 };

function renderRadar() {
  requestAnimationFrame(renderRadar);
  const radar = $('radar');
  const r = radar.parentElement.getBoundingClientRect();
  const w = r.width - 12; const h = r.height - 12;
  if (w < 10 || h < 10) return;
  const ctx = fitCanvas(radar, w, h);
  ctx.clearRect(0, 0, w, h);

  const cx = w / 2; const cy = h - 14; const R = Math.min(w / 2 - 8, h - 26);
  const t = performance.now();
  // Angle 0 is straight ahead (up), positive to the right.
  const polar = (a, rr) => [cx + Math.sin(a) * rr, cy - Math.cos(a) * rr];

  [[1, '#f43f5e'], [0.62, '#fbbf24'], [0.34, '#22d3ee']].forEach(([f, c]) => {
    ctx.beginPath(); ctx.arc(cx, cy, R * f, Math.PI, 0);
    ctx.strokeStyle = c; ctx.globalAlpha = 0.16; ctx.lineWidth = 1; ctx.stroke();
  });
  ctx.globalAlpha = 1;

  ctx.strokeStyle = 'rgba(203,217,230,.13)'; ctx.lineWidth = 1;
  [-0.31, 0.31].forEach((a) => {
    const [ex, ey] = polar(a, R);
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ex, ey); ctx.stroke();
  });

  const sw = -Math.PI / 2 + ((t / 2600) % 1) * Math.PI;
  const [gx, gy] = polar(sw, R);
  const g = ctx.createLinearGradient(cx, cy, gx, gy);
  g.addColorStop(0, 'rgba(34,211,238,.30)'); g.addColorStop(1, 'rgba(34,211,238,0)');
  ctx.strokeStyle = g; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(gx, gy); ctx.stroke();

  ctx.fillStyle = '#33455a'; ctx.font = '8px ui-monospace,monospace';
  ctx.fillText('L', 10, h - 4); ctx.fillText('AHEAD', cx - 13, h - 4); ctx.fillText('R', w - 14, h - 4);

  for (const d of state.detections) {
    const p = PROX[PROXIMITY_NAME[d.proximity]];
    const a = ZONE_ANGLE[d.zone] ?? 0;
    const near = d.distanceM != null ? clamp(1 - d.distanceM / 7, 0.12, 1) : 0.25 + d.proximity * 0.22;
    const [x, y] = polar(a, R * (1 - near) * 0.92 + 8);
    const puls = d.proximity === Proximity.IMMEDIATE ? 3 + Math.sin(t / 130) * 1.6 : 3;
    ctx.beginPath(); ctx.arc(x, y, puls + 5, 0, 7); ctx.fillStyle = p.c; ctx.globalAlpha = 0.16; ctx.fill();
    ctx.beginPath(); ctx.arc(x, y, puls, 0, 7); ctx.fillStyle = p.c; ctx.globalAlpha = 1; ctx.fill();
  }
  ctx.globalAlpha = 1;
}

/* ── waveform ──────────────────────────────────────────── */
const HIST = 190;
const levels = new Array(HIST).fill(0);

function renderWave() {
  requestAnimationFrame(renderWave);
  const wave = $('wave');
  const r = wave.getBoundingClientRect();
  if (r.width < 10) return;
  const ctx = fitCanvas(wave, r.width, r.height);
  const w = r.width; const h = r.height; const mid = h / 2;
  ctx.clearRect(0, 0, w, h);

  let rms = 0;
  if (analyser && state.audio) {
    analyser.getByteTimeDomainData(waveData);
    let sum = 0;
    for (let i = 0; i < waveData.length; i += 1) {
      const v = (waveData[i] - 128) / 128; sum += v * v;
    }
    rms = Math.min(1, Math.sqrt(sum / waveData.length) * 3.2);
  }
  levels.push(rms); levels.shift();

  ctx.strokeStyle = 'rgba(203,217,230,.07)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w, mid); ctx.stroke();

  const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#22d3ee';
  const bw = w / HIST;
  for (let i = 0; i < HIST; i += 1) {
    const v = levels[i]; const x = i * bw;
    if (v <= 0.004) {
      ctx.fillStyle = 'rgba(92,113,134,.30)';
      ctx.fillRect(x, mid - 0.5, Math.max(1, bw - 0.6), 1);
    } else {
      const a = v * (mid - 4);
      ctx.fillStyle = accent; ctx.globalAlpha = 0.35 + 0.65 * (i / HIST);
      ctx.fillRect(x, mid - a, Math.max(1, bw - 0.6), a * 2);
      ctx.globalAlpha = 1;
    }
  }
}

/* ── haptic meters ─────────────────────────────────────── */
function renderHaptics() {
  requestAnimationFrame(renderHaptics);
  const hp = state.haptics || { left: 0, right: 0, pulseHz: 0, strength: 0, proximity: 'FAR', active: false };
  const p = PROX[hp.proximity] || PROX.FAR;

  let env = 1;
  if (hp.pulseHz > 0) env = ((performance.now() / 1000 * hp.pulseHz) % 1) < 0.5 ? 1 : 0.28;

  $('fillL').style.height = `${hp.left * env * 100}%`;
  $('fillR').style.height = `${hp.right * env * 100}%`;
  for (const el of [$('fillL'), $('fillR')]) {
    el.style.background = `linear-gradient(0deg,${p.c},transparent)`;
    el.style.borderTopColor = p.c;
  }
  $('pctL').textContent = `${Math.round(hp.left * 100)}%`;
  $('pctR').textContent = `${Math.round(hp.right * 100)}%`;
  $('hapPct').textContent = `${Math.round(hp.strength * 100)}%`;
  $('arc').style.strokeDashoffset = (276.5 * (1 - hp.strength)).toFixed(1);
  $('arc').style.stroke = p.c;
  $('hapPat').textContent = hp.active ? `${p.n} · ${hp.pulseHz.toFixed(1)}Hz` : 'idle';
}

/* ── side panels ───────────────────────────────────────── */
let tableKey = ''; let transcriptKey = '';

function setAccent(c) { document.documentElement.style.setProperty('--accent', c); }

function updatePanels() {
  const dets = state.detections;
  $('fps').textContent = state.fps.toFixed(1);
  $('objs').textContent = dets.length;

  const sorted = [...dets].sort((a, b) => b.proximity - a.proximity
    || (b.zone === Zone.CENTER) - (a.zone === Zone.CENTER) || b.bbox.area - a.bbox.area);
  const d = sorted[0];

  if (!d) {
    $('tName').textContent = 'standing by';
    $('tDist').innerHTML = '—<span class="u"></span>';
    $('tBand').textContent = '—'; $('tZone').textContent = '—'; $('tConf').textContent = '—';
    $('tBar').style.width = '0%'; $('trunc').textContent = '';
    $('tDist').style.color = 'var(--tx)'; setAccent('#22d3ee');
  } else {
    const p = PROX[PROXIMITY_NAME[d.proximity]];
    $('tName').textContent = d.label;
    $('tDist').innerHTML = d.distanceM != null
      ? `${d.truncated ? '&lt;' : ''}${d.distanceM.toFixed(1)}<span class="u">m</span>`
      : '—<span class="u"></span>';
    $('tDist').style.color = p.c;
    $('tBand').textContent = p.n;
    $('tZone').textContent = d.zone;
    $('tConf').textContent = `${Math.round(d.confidence * 100)}% conf`;
    $('tBar').style.width = `${25 + d.proximity * 25}%`;
    $('tBar').style.background = p.c;
    $('trunc').textContent = d.truncated ? '⚠ upper bound' : '';
    setAccent(p.c);
    $('mark').style.borderColor = p.c;
  }

  // Rebuild only on change: doing it every pass restarts the entry animation
  // and leaves the transcript looking empty.
  const key = dets.map((x) => `${x.trackId}:${x.proximity}:${x.distanceM}:${x.zone}`).join('|');
  if (key !== tableKey) {
    tableKey = key;
    $('detCount').textContent = `${dets.length} object${dets.length === 1 ? '' : 's'}`;
    const wrap = $('tblWrap');
    if (!dets.length) {
      wrap.innerHTML = '<div class="empty">no objects detected</div>';
    } else {
      const rows = [...dets].sort((a, b) => b.proximity - a.proximity).map((x) => {
        const p = PROX[PROXIMITY_NAME[x.proximity]];
        const dist = x.distanceM != null ? `${x.truncated ? '&lt;' : '~'}${x.distanceM.toFixed(1)} m` : '—';
        return `<tr><td style="color:var(--dimmer)">#${x.trackId ?? '–'}</td>
          <td style="font-weight:600">${x.label}</td>
          <td><span class="chip" style="background:${p.c}22;color:${p.c}">${p.n}</span></td>
          <td>${dist}</td><td style="color:var(--dimmer)">${x.distanceMethod || '—'}</td>
          <td style="color:var(--dim)">${x.zone}</td>
          <td style="color:var(--dim)">${Math.round(x.confidence * 100)}%</td></tr>`;
      }).join('');
      wrap.innerHTML = `<table><thead><tr><th>id</th><th>object</th><th>band</th>
        <th>range</th><th>via</th><th>bearing</th><th>conf</th></tr></thead><tbody>${rows}</tbody></table>`;
    }
  }

  if (state.transcript.length) {
    const tkey = `${state.transcript.length}|${state.transcript[state.transcript.length - 1].at}`;
    if (tkey !== transcriptKey) {
      transcriptKey = tkey;
      $('feed').innerHTML = [...state.transcript].reverse().map((e) => {
        const p = PROX[e.proximity] || PROX.FAR;
        const tm = new Date(e.at * 1000).toLocaleTimeString([], { hour12: false });
        const dist = e.distanceM != null ? ` · ${e.distanceM.toFixed(1)}m` : '';
        return `<div class="msg" style="border-left-color:${p.c}">
          <div class="tm">${tm}</div><div><div class="tx">${e.text}</div>
          <div class="mt">${p.n} · ${e.zone}${dist}</div></div></div>`;
      }).join('');
    }
  }
}

/* ── controls ──────────────────────────────────────────── */
function setRunning(on) {
  state.running = on;
  $('runBtn').classList.toggle('stopped', !on);
  $('runTxt').textContent = on ? 'Stop' : 'Start';
  $('dot').className = on ? 'dot on' : 'dot';
  $('conn').textContent = on ? 'live' : 'stopped';
  if (!on) {
    state.detections = []; state.haptics = null; smooth.clear();
    $('nosig').classList.remove('hide');
    updatePanels();
  } else {
    detectLoop();
  }
}

function bindToggle(id, key, after) {
  $(id).onclick = () => {
    state[key] = !state[key];
    $(id).classList.toggle('on', state[key]);
    if (after) after(state[key]);
  };
}

bindToggle('mirrorBtn', 'mirror');
bindToggle('boxBtn', 'boxes');
bindToggle('voiceBtn', 'voice', (on) => { if (!on) window.speechSynthesis?.cancel(); });
bindToggle('audioBtn', 'audio', (on) => {
  $('audTag').textContent = on ? 'live' : 'muted';
  if (on) { initAudio(); actx.resume(); }
});
$('runBtn').onclick = () => setRunning(!state.running);

/* ── boot ──────────────────────────────────────────────── */
$('startBtn').onclick = async () => {
  const btn = $('startBtn');
  btn.disabled = true;
  $('gerr').classList.remove('show');
  $('prog').style.display = 'block';

  const step = (text, pct) => {
    $('progTxt').textContent = text;
    $('progBar').style.width = `${pct}%`;
  };

  try {
    // This click is also the gesture browsers require before audio may play.
    initAudio(); await actx.resume();

    step('requesting camera', 15);
    await startCamera();

    step('downloading model (~24 MB, cached after this)', 40);
    detector = await loadModel((s) => step(s, 70));

    step('warming up', 90);
    policy = new AlertPolicy({
      cooldownS: 4.0, minGapS: 1.2, minHits: 2, minProximity: Proximity.MODERATE,
      clock: () => performance.now() / 1000,
    });
    tracker = new IoUTracker();

    $('srcTag').textContent = `on-device inference · ${detector.backend}`;
    step('ready', 100);
    $('gate').classList.add('hide');
    setRunning(true);
    earcon('MODERATE', 'center');
  } catch (err) {
    console.error(err);
    const msg = String(err && err.message ? err.message : err);
    $('gerr').textContent = msg.includes('Permission') || msg.includes('denied')
      ? 'Camera permission was denied. Allow camera access in your browser and reload.'
      : `Could not start: ${msg}`;
    $('gerr').classList.add('show');
    $('prog').style.display = 'none';
    btn.disabled = false;
  }
};

requestAnimationFrame(renderView);
requestAnimationFrame(renderRadar);
requestAnimationFrame(renderWave);
requestAnimationFrame(renderHaptics);
updatePanels();
