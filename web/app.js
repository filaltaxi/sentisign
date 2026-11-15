// SentiSign v2 browser client. Perception on device; only glosses + emotion go to the server.
import { FilesetResolver, HandLandmarker, PoseLandmarker, FaceLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { sequenceFeatures } from "./features.js";

const $ = id => document.getElementById(id);
const FPS = 30, SHORT_REST = 0.35 * FPS, LONG_REST = 0.8 * FPS, MIN_SIGN = 0.3 * FPS, SPEED_THR = 0.012;
let hand, pose, face, sess, meta, emo, ws;
let frames = [], restN = 0, activeN = 0, glosses = [], blendBuf = [], neutral = null, calibrating = null, lastTop3 = null;

const setStatus = t => ($("status").textContent = t);
const setState = t => ($("state").textContent = t);

async function load() {
  const fileset = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  const opt = path => ({ baseOptions: { modelAssetPath: path, delegate: "GPU" }, runningMode: "VIDEO" });
  hand = await HandLandmarker.createFromOptions(fileset, { ...opt("/models/hand_landmarker.task"), numHands: 2 });
  pose = await PoseLandmarker.createFromOptions(fileset, opt("/models/pose_landmarker_lite.task"));
  face = await FaceLandmarker.createFromOptions(fileset, { ...opt("/models/face_landmarker.task"), outputFaceBlendshapes: true, numFaces: 1 });
  meta = await (await fetch("/models/sign_meta.json")).json();
  sess = await ort.InferenceSession.create("/models/sign.onnx", { executionProviders: ["wasm"] });
  try { emo = await (await fetch("/models/emotion.json")).json(); } catch { emo = null; }
  await connectServer();
  setStatus(`Ready · ${meta.glosses.length} signs`); $("calib").disabled = false;
}

function anchorsFromPose(p) {
  const avg = (a, b) => a.map((v, i) => (v + b[i]) / 2), add = (a, b, k) => a.map((v, i) => v + k * b[i]), sub = (a, b) => a.map((v, i) => v - b[i]);
  const nose = p[0], eyes = avg(p[2], p[5]), mouth = avg(p[9], p[10]);
  return [nose, add(mouth, sub(mouth, nose), 0.8), add(eyes, sub(eyes, mouth), 0.8), p[7], p[8]];
}

function frameFromResults(hr, pr, w, h) {
  const sx = w / h, NaN3 = () => Array.from({ length: 21 }, () => [NaN, NaN, NaN]);
  const hands = [NaN3(), NaN3()];
  hr.landmarks.forEach((lm, i) => { hands[hr.handednesses[i][0].categoryName === "Left" ? 0 : 1] = lm.map(q => [q.x * sx, q.y, q.z * sx]); });
  if (!pr.landmarks.length) return null;
  const p = pr.landmarks[0].map(q => [q.x * sx, q.y, q.z * sx]);
  return { hands, pose: p, anchors: anchorsFromPose(p) };
}

function wristSpeed(a, b) {
  const s = (p) => { const L = p.pose[11], R = p.pose[12]; return [Math.hypot(L[0] - R[0], L[1] - R[1]) || 1e-6]; };
  let v = 0;
  for (const k of [0, 1]) {
    const w1 = a.hands[k][0], w2 = b.hands[k][0];
    if (!Number.isNaN(w1[0]) && !Number.isNaN(w2[0])) v += Math.hypot(w1[0] - w2[0], w1[1] - w2[1]) / s(b)[0];
  }
  return v;
}

async function classify(seg) {
  const X = sequenceFeatures(seg.map(f => f.hands), seg.map(f => f.pose), seg.map(f => f.anchors), meta.T);
  const flat = new Float32Array(X.length * X[0].length);
  X.forEach((r, t) => r.forEach((v, c) => (flat[t * r.length + c] = (v - meta.mu[c]) / meta.sd[c])));
  const out = await sess.run({ x: new ort.Tensor("float32", flat, [1, X.length, X[0].length]) });
  const z = Array.from(out.logits.data).map(v => v / meta.temperature), m = Math.max(...z);
  const e = z.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0), p = e.map(v => v / s);
  const order = p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
  return order.map(([prob, i]) => ({ gloss: meta.glosses[i], prob }));
}

function emotionNow() {
  if (!emo || !blendBuf.length) return { label: "Neutral", probs: null };
  const mean = blendBuf[0].map((_, j) => blendBuf.reduce((s, b) => s + b[j], 0) / blendBuf.length);
  const x = mean.map((v, j) => v - (neutral ? neutral[j] : 0));
  const z = emo.W.map((row, c) => row.reduce((s, w, j) => s + w * (x[j] - emo.mu[j]) / emo.sd[j], emo.b[c]));
  const m = Math.max(...z), e = z.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
  const probs = e.map(v => v / s), i = probs.indexOf(Math.max(...probs));
  return { label: emo.labels[i], probs };
}

function render() {
  $("glosses").innerHTML = glosses.map(g => `<span class="${g.sure ? "" : "unsure"}">${g.gloss}</span>`).join("");
  $("top3").innerHTML = lastTop3 && !lastTop3.sure
    ? `Not sure — did you mean ${lastTop3.cands.map((c, k) => `<button data-k="${k}">${c.gloss} ${(c.prob * 100).toFixed(0)}%</button>`).join("")}`
    : "";
  $("top3").querySelectorAll("button").forEach(b => (b.onclick = () => { glosses[glosses.length - 1] = { gloss: lastTop3.cands[b.dataset.k].gloss, sure: true }; lastTop3 = null; render(); prepare(); }));
}

// Two ways to reach the sentence + voice models: a WebSocket on the local server (laptop),
// or the Gradio API on the Hugging Face Space (free shared GPU, one call per request).
let gradio = null, prepared = null;
async function connectServer() {
  let mode = "ws";
  try { mode = (await (await fetch("/api/mode")).json()).mode; } catch {}
  if (mode === "gradio") {
    const { Client } = await import("https://cdn.jsdelivr.net/npm/@gradio/client@2.8.0/dist/index.min.js");
    gradio = await Client.connect(`${location.origin}/gradio/`);
  } else {
    ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
    ws.onmessage = onServer;
  }
}
const deliver = (o) => onServer({ data: JSON.stringify(o) });
async function sendGradio(o) {
  const g = o.glosses.join(" ");
  try {
    if (o.type === "prepare") {
      if (!g) return;
      const key = g + "|" + o.emotion;
      prepared = { key, p: gradio.predict("/compose", [g, o.emotion]) };
      const r = await prepared.p; if (prepared && prepared.key === key) deliver({ type: "sentence", text: r.data[0] });
    } else if (o.type === "speak") {
      setStatus("Speaking… (the free GPU can take a few seconds)");
      const r = await gradio.predict("/speak", [g, o.emotion, o.sentence || ""]);
      const [text, wav] = r.data; deliver({ type: "sentence", text });
      deliver(wav ? { type: "audio", wav } : { type: "text_only" }); prepared = null;
      setStatus(`Ready · ${meta.glosses.length} signs`);
    }
  } catch (e) { setStatus("Server busy or GPU quota reached: " + (e.message || e)); setState("idle"); }
}
const send = (o) => gradio ? sendGradio(o) : (ws && ws.readyState === 1 && ws.send(JSON.stringify(o)));
function prepare() { send({ type: "prepare", glosses: glosses.map(g => g.gloss), emotion: emotionNow().label }); }
function speak() {
  if (!glosses.length && !$("sentence").value) return;
  send({ type: "speak", glosses: glosses.map(g => g.gloss), emotion: emotionNow().label, sentence: $("sentence").dataset.edited ? $("sentence").value : null });
  setState("speaking");
}
function onServer(ev) {
  const m = JSON.parse(ev.data);
  if (m.type === "sentence" && !$("sentence").dataset.edited) $("sentence").value = m.text;
  if (m.type === "audio") { $("audio").src = `data:audio/wav;base64,${m.wav}`; $("audio").play(); glosses = []; blendBuf = []; $("sentence").dataset.edited = ""; render(); setState("idle"); }
  if (m.type === "text_only") { setState("idle (voice unavailable: text only)"); glosses = []; render(); }
}

async function loop() {
  const v = $("video"), t = performance.now();
  if (v.readyState >= 2) {
    const hr = hand.detectForVideo(v, t), pr = pose.detectForVideo(v, t), fr = face.detectForVideo(v, t);
    const f = frameFromResults(hr, pr, v.videoWidth, v.videoHeight);
    const bs = fr.faceBlendshapes?.[0]?.categories.map(c => c.score);
    if (bs) { if (calibrating) calibrating.push(bs); else blendBuf.push(bs); }
    if (f) {
      const moving = frames.length && wristSpeed(frames[frames.length - 1], f) > SPEED_THR / FPS * 30;
      const present = hr.landmarks.length > 0;
      if (moving && present) { activeN++; restN = 0; } else if (activeN || glosses.length) restN++;
      if (activeN) frames.push(f);
      setState(activeN ? "signing" : glosses.length ? "pause" : "idle");
      if (activeN >= MIN_SIGN && restN >= SHORT_REST) {            // one sign ended
        const seg = frames.slice(0, frames.length - restN); frames = []; activeN = 0;
        const cands = await classify(seg), sure = cands[0].prob >= meta.abstain_thr;
        glosses.push({ gloss: cands[0].gloss, sure }); lastTop3 = { cands, sure }; render(); prepare();
      } else if (!activeN && glosses.length && restN >= LONG_REST) { // utterance ended
        restN = 0; speak();
      }
    }
  }
  requestAnimationFrame(loop);
}

$("start").onclick = async () => {
  $("video").srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, frameRate: 30 } });
  $("start").disabled = true; loop();
};
$("calib").onclick = () => { calibrating = []; setState("hold a neutral face…");
  setTimeout(() => { neutral = calibrating[0].map((_, j) => calibrating.reduce((s, b) => s + b[j], 0) / calibrating.length); calibrating = null; setState("calibrated"); }, 3000); };
$("send").onclick = speak;
$("undo").onclick = () => { glosses.pop(); lastTop3 = null; render(); prepare(); };
$("clear").onclick = () => { glosses = []; lastTop3 = null; $("sentence").value = ""; $("sentence").dataset.edited = ""; render(); };
$("sentence").oninput = () => ($("sentence").dataset.edited = "1");
setInterval(() => { const e = emotionNow(); $("emotion").textContent = e.label; if (blendBuf.length > 150) blendBuf = blendBuf.slice(-150); }, 500);
load().catch(e => setStatus("Load failed: " + e.message));
