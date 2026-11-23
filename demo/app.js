"use strict";

/* onomatron browser demo — vanilla JS + onnxruntime-web (CDN, pinned).
 *
 * Loads model.onnx + vocab.json (both produced by scripts/export_onnx.py),
 * then generates one character per "Next char" click, exactly like the
 * model's sample(): feed the sentinel, draw from softmax(logits), reject
 * the sentinel until the name reaches MIN_LEN, stop at MAX_LEN.
 *
 * Two graph kinds are supported (see scripts/export_onnx.py):
 *  - "rnn-step":   step(tok (1,1) int64, h (1,1,H) f32) -> {logits, h_new}
 *                  the hidden state h is chained from one char to the next;
 *  - "ff-context": context (1, block_size) int64 -> {logits}
 *                  the context window rolls (last block_size chars).
 */

// Must match the <script src> version in index.html. The .wasm binaries are
// served from the same CDN directory.
const ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";

// Same defaults as the models' sample() in src/onomatron/models/.
const MIN_LEN = 3;
const MAX_LEN = 40;

const els = {
  prefix: document.getElementById("prefix"),
  next: document.getElementById("next"),
  reset: document.getElementById("reset"),
  temp: document.getElementById("temp"),
  tempVal: document.getElementById("temp-val"),
  name: document.getElementById("name"),
  status: document.getElementById("status"),
  modelInfo: document.getElementById("model-info"),
};

let session = null;
let vocab = null; // parsed vocab.json
let eos = 0;
let busy = false;

// Generation state. `h`/`context` is what the graph is fed next; `logits` is
// the distribution after the last fed character.
let h = null;
let context = null;
let logits = null;
let generated = "";
let done = false;
let warmedPrefix = null; // prefix the current state was warmed with

function setStatus(msg) {
  els.status.textContent = msg;
}

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

// One graph step on character `ix`: returns the new logits (Float32Array).
async function feed(ix) {
  let out;
  if (vocab.kind === "rnn-step") {
    out = await session.run({ tok: int64Tensor(ix), h });
    h = out.h_new;
    return out.logits.data;
  }
  // ff-context: roll the window, pad left with the sentinel index
  context.push(BigInt(ix));
  context.splice(0, context.length - vocab.block_size);
  const padded = Array(vocab.block_size - context.length).fill(BigInt(eos));
  const ctx = new ort.Tensor("int64", BigInt64Array.from([...padded, ...context]), [
    1,
    vocab.block_size,
  ]);
  out = await session.run({ context: ctx });
  return out.logits.data;
}

// Warm the graph on the sentinel + prefix, like sample() starts from eos.
async function warm(prefix) {
  if (vocab.kind === "rnn-step") {
    h = new ort.Tensor("float32", new Float32Array(vocab.h_dim), [1, 1, vocab.h_dim]);
  } else {
    context = [];
  }
  logits = await feed(eos);
  for (const ch of prefix) {
    logits = await feed(vocab.stoi[ch]);
  }
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

function render() {
  const prefix = sanitizePrefix(els.prefix.value);
  els.name.textContent = "";
  els.name.append(document.createTextNode(prefix));
  const gen = document.createElement("span");
  gen.className = "generated";
  gen.textContent = generated;
  els.name.append(gen);
  els.next.disabled = busy || done || !session;
  setStatus(
    done
      ? `done — “${prefix}${generated}” (end-of-name sampled)`
      : `${prefix.length + generated.length}/${MAX_LEN} chars`,
  );
}

async function ensureWarmed() {
  const prefix = sanitizePrefix(els.prefix.value);
  if (prefix !== warmedPrefix) {
    await warm(prefix);
    warmedPrefix = prefix;
    generated = "";
    done = false;
  }
}

async function next() {
  if (busy || done || !session) return;
  busy = true;
  els.next.disabled = true;
  try {
    await ensureWarmed();
    const prefix = warmedPrefix;
    if (prefix.length + generated.length >= MAX_LEN) {
      done = true;
      render();
      return;
    }
    const temp = parseFloat(els.temp.value);
    // Reject eos until the name reaches MIN_LEN, redrawing from the same
    // distribution (same behavior as sample() in the Python models).
    let ix;
    do {
      ix = sample(logits, temp);
    } while (ix === eos && prefix.length + generated.length < MIN_LEN);
    if (ix === eos) {
      done = true;
    } else {
      generated += vocab.chars[ix];
      logits = await feed(ix);
      if (prefix.length + generated.length >= MAX_LEN) done = true;
    }
    render();
  } finally {
    busy = false;
    render();
  }
}

async function reset() {
  generated = "";
  done = false;
  warmedPrefix = null; // force re-warm on the current prefix
  await ensureWarmed();
  render();
}

async function init() {
  // GitHub Pages sends no COOP/COEP headers, so SharedArrayBuffer is
  // unavailable: single-threaded wasm is required.
  ort.env.wasm.wasmPaths = ORT_CDN;
  ort.env.wasm.numThreads = 1;
  try {
    const [vocabDoc, sess] = await Promise.all([
      fetch("vocab.json").then((r) => {
        if (!r.ok) throw new Error(`vocab.json: HTTP ${r.status}`);
        return r.json();
      }),
      ort.InferenceSession.create("model.onnx"),
    ]);
    vocab = vocabDoc;
    session = sess;
    eos = vocab.stoi[vocab.sentinel];
    const m = vocab.metrics || {};
    els.modelInfo.textContent =
      `${vocab.model} · test NLL ${Number(m.test_nll).toFixed(3)} · ` +
      `${Math.round(m.n_parameters)} params · ${vocab.kind}`;
    els.next.disabled = false;
    els.reset.disabled = false;
    await reset();
    setStatus(`ready — click “Next char” (model=${vocab.model}, kind=${vocab.kind})`);
  } catch (err) {
    setStatus(`failed to load the model: ${err.message}`);
    console.error(err);
  }
}

els.next.addEventListener("click", next);
els.reset.addEventListener("click", reset);
els.prefix.addEventListener("keydown", (e) => {
  if (e.key === "Enter") next();
});
els.temp.addEventListener("input", () => {
  els.tempVal.textContent = parseFloat(els.temp.value).toFixed(2);
});

init();
