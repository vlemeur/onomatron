"use strict";

/* onomatron browser demo — vanilla JS + onnxruntime-web (CDN, pinned).
 *
 * Loads model.onnx + vocab.json (both produced by scripts/export_onnx.py) and
 * offers two generation modes, both mirroring the models' sample() (draw from
 * softmax(logits / temp), reject the end-of-name sentinel until MIN_LEN chars,
 * stop at MAX_LEN):
 *  - "Next char": completes the current name one character per click, from the
 *    optional prefix (interactive streaming);
 *  - "Generate": samples N full names at once, each starting from the sentinel
 *    + the optional prefix (like scripts/app_streamlit.py).
 *
 * Two graph kinds are supported (see scripts/export_onnx.py):
 *  - "rnn-step":   step(tok (1,1) int64, h (1,1,H) f32) -> {logits, h_new}
 *  - "ff-context": context (1, block_size) int64 -> {logits}
 */

// Must match the <script src> version in index.html; the .wasm binaries are
// served from the same CDN directory.
const ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";

// Same defaults as the models' sample() and the Streamlit app.
const MIN_LEN = 3;
const MAX_LEN = 40;

const els = {
  prefix: document.getElementById("prefix"),
  next: document.getElementById("next"),
  reset: document.getElementById("reset"),
  generate: document.getElementById("generate"),
  count: document.getElementById("count"),
  temp: document.getElementById("temp"),
  tempVal: document.getElementById("temp-val"),
  name: document.getElementById("name"),
  names: document.getElementById("names"),
  status: document.getElementById("status"),
  modelInfo: document.getElementById("model-info"),
};

let session = null;
let vocab = null; // parsed vocab.json
let eos = 0;
let busy = false;

// Interactive ("Next char") state: {h, context, logits}, warmed on the prefix.
let state = null;
let generated = "";
let done = false;
let warmedPrefix = null;

function sanitizePrefix(raw) {
  // Keep only characters the model knows (the alphabet of names.txt);
  // names are lowercase in the dataset.
  return raw
    .toLowerCase()
    .split("")
    .filter((c) => c in vocab.stoi && c !== vocab.sentinel)
    .join("");
}

function int64Tensor(ix) {
  return new ort.Tensor("int64", BigInt64Array.of(BigInt(ix)), [1, 1]);
}

// One graph step on character ix: updates the state in place.
async function feed(st, ix) {
  let out;
  if (vocab.kind === "rnn-step") {
    out = await session.run({ tok: int64Tensor(ix), h: st.h });
    st.h = out.h_new;
  } else {
    // ff-context: roll the window, pad left with the sentinel index
    st.context.push(BigInt(ix));
    st.context.splice(0, st.context.length - vocab.block_size);
    const padded = Array(vocab.block_size - st.context.length).fill(BigInt(eos));
    const ctx = new ort.Tensor(
      "int64",
      BigInt64Array.from([...padded, ...st.context]),
      [1, vocab.block_size],
    );
    out = await session.run({ context: ctx });
  }
  st.logits = out.logits.data;
}

// Fresh state warmed on the sentinel + prefix (sample() starts from eos).
async function warmState(prefix) {
  const st =
    vocab.kind === "rnn-step"
      ? {
          h: new ort.Tensor("float32", new Float32Array(vocab.h_dim), [
            1,
            1,
            vocab.h_dim,
          ]),
          context: null,
          logits: null,
        }
      : { h: null, context: [], logits: null };
  await feed(st, eos);
  for (const ch of prefix) {
    await feed(st, vocab.stoi[ch]);
  }
  return st;
}

// Draw one index from logits with temperature (temp <= 0.01 = greedy).
function sample(logitsArr, temp) {
  const n = logitsArr.length;
  if (temp <= 0.01) {
    let best = 0;
    for (let i = 1; i < n; i++) if (logitsArr[i] > logitsArr[best]) best = i;
    return best;
  }
  const max = Math.max(...logitsArr);
  const probs = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    probs[i] = Math.exp((logitsArr[i] - max) / temp);
    sum += probs[i];
  }
  let r = Math.random() * sum;
  for (let i = 0; i < n; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return n - 1;
}

// Draw a char, rejecting eos while the name is shorter than MIN_LEN
// (same behavior as sample() in the Python models).
function draw(logitsArr, temp, lenSoFar) {
  let ix;
  do {
    ix = sample(logitsArr, temp);
  } while (ix === eos && lenSoFar < MIN_LEN);
  return ix;
}

// Sample one full name (without the prefix) from a fresh warmed state.
async function sampleName(prefix, temp) {
  const st = await warmState(prefix);
  let name = "";
  let ix = draw(st.logits, temp, prefix.length + name.length);
  while (ix !== eos && prefix.length + name.length < MAX_LEN) {
    name += vocab.chars[ix];
    await feed(st, ix);
    ix = draw(st.logits, temp, prefix.length + name.length);
  }
  return name;
}

function render() {
  const prefix = sanitizePrefix(els.prefix.value);
  els.name.textContent = "";
  els.name.append(document.createTextNode(prefix));
  const gen = document.createElement("span");
  gen.className = "generated";
  gen.textContent = generated;
  els.name.append(gen);
  const locked = busy || !session;
  els.next.disabled = locked || done;
  els.reset.disabled = locked;
  els.generate.disabled = locked;
  if (done) {
    setStatus(`done — '${prefix}${generated}' (end-of-name sampled)`);
  } else if (session) {
    setStatus(`${prefix.length + generated.length}/${MAX_LEN} chars`);
  }
}

// One interactive step on the current name.
async function next() {
  if (busy || done || !session) return;
  busy = true;
  render();
  try {
    const prefix = sanitizePrefix(els.prefix.value);
    if (prefix !== warmedPrefix) {
      state = await warmState(prefix);
      warmedPrefix = prefix;
      generated = "";
      done = false;
    }
    if (prefix.length + generated.length >= MAX_LEN) {
      done = true;
      return;
    }
    const temp = parseFloat(els.temp.value);
    const ix = draw(state.logits, temp, prefix.length + generated.length);
    if (ix === eos) {
      done = true;
      return;
    }
    generated += vocab.chars[ix];
    await feed(state, ix);
    if (prefix.length + generated.length >= MAX_LEN) done = true;
  } finally {
    busy = false;
    render();
  }
}

async function reset() {
  generated = "";
  done = false;
  warmedPrefix = null; // force re-warm on the current prefix
  state = await warmState(sanitizePrefix(els.prefix.value));
  warmedPrefix = sanitizePrefix(els.prefix.value);
  render();
}

// Batch mode: sample N full names, each from the sentinel + current prefix.
async function generate() {
  if (busy || !session) return;
  busy = true;
  render();
  try {
    const n = Math.max(1, Math.min(200, parseInt(els.count.value, 10) || 20));
    const prefix = sanitizePrefix(els.prefix.value);
    const temp = parseFloat(els.temp.value);
    els.names.textContent = "";
    for (let i = 0; i < n; i++) {
      const name = await sampleName(prefix, temp);
      const li = document.createElement("li");
      li.textContent = prefix + name;
      els.names.append(li);
    }
    setStatus(`${n} names (prefix '${prefix || "none"}', T=${temp.toFixed(2)})`);
  } finally {
    busy = false;
    // re-enable the buttons without overwriting the batch status message
    els.next.disabled = done;
    els.reset.disabled = false;
    els.generate.disabled = false;
  }
}

function setStatus(msg) {
  els.status.textContent = msg;
}

function withTimeout(promise, ms, label) {
  return Promise.race([
    promise,
    new Promise((_, reject) =>
      setTimeout(
        () => reject(new Error(`${label} — timed out after ${ms / 1000}s`)),
        ms,
      ),
    ),
  ]);
}

async function init() {
  if (typeof ort === "undefined") {
    setStatus(
      "onnxruntime-web failed to load from the CDN (network/adblocker?) — " +
        `check ${ORT_CDN}ort.min.js in a browser tab, then reload`,
    );
    return;
  }
  // GitHub Pages / plain http.server send no COOP/COEP headers, so
  // SharedArrayBuffer is unavailable: single-threaded wasm is required.
  ort.env.wasm.wasmPaths = ORT_CDN;
  ort.env.wasm.numThreads = 1;
  try {
    setStatus("loading wasm backend + model.onnx + vocab.json…");
    const [vocabDoc, sess] = await withTimeout(
      Promise.all([
        fetch("vocab.json").then((r) => {
          if (!r.ok) throw new Error(`vocab.json: HTTP ${r.status}`);
          return r.json();
        }),
        ort.InferenceSession.create("model.onnx"),
      ]),
      30_000,
      "model load",
    );
    vocab = vocabDoc;
    session = sess;
    eos = vocab.stoi[vocab.sentinel];
    const m = vocab.metrics || {};
    els.modelInfo.textContent =
      `${vocab.model} · test NLL ${Number(m.test_nll).toFixed(3)} · ` +
      `${Math.round(m.n_parameters)} params · ${vocab.kind}`;
    await reset();
    els.next.disabled = false;
    els.reset.disabled = false;
    els.generate.disabled = false;
    setStatus("ready");
  } catch (err) {
    setStatus(`failed to load the model: ${err.message}`);
    console.error(err);
  }
}

els.next.addEventListener("click", next);
els.reset.addEventListener("click", reset);
els.generate.addEventListener("click", generate);
els.prefix.addEventListener("keydown", (e) => {
  if (e.key === "Enter") next();
});
els.temp.addEventListener("input", () => {
  els.tempVal.textContent = parseFloat(els.temp.value).toFixed(2);
});

init();
