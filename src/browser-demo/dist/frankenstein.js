const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const K1 = 1.2;
const B = 0.75;
const MARKER = "\n\nStory so far (excerpts):\n";
const CFG = window.FRANKENSTEIN || {
  index: "frankenstein-index/",
  prompt: "frankenstein-prompt/",
  model: "models/Qwen3-4B-q4f16_1-MLC/",
  wasm: "models/Qwen3-4B-q4f16_1_cs1k-webgpu.wasm",
  vendor: "./vendor/web-llm.js",
};
const SUGGEST = {
  victor: [
    { from: 4, q: "What do Walton and the crew see on the ice?" },
    { from: 5, q: "Who is Elizabeth, in this book?" },
    { from: 8, q: "What happened on the night the creature woke?" },
  ],
  creature: [
    { from: 13, q: "Why should Victor listen to you?" },
    { from: 17, q: "Who are the cottagers you watched?" },
    { from: 19, q: "What did you do at Plainpalais?" },
  ],
};

const state = {
  pos: 4,
  char: "victor",
  units: [],
  messages: [],
  engine: null,
  loading: false,
  busy: false,
  fetched: [],
  manifest: null,
};

function tokenize(text) {
  return (text || "").toLowerCase().match(/[a-z0-9]+(?:'[a-z]+)?/g) || [];
}

function roundPpm(raw) {
  const scaled = raw * 1000000;
  const base = Math.floor(scaled);
  const diff = scaled - base;
  if (diff > 0.5) return base + 1;
  if (diff < 0.5) return base;
  return base % 2 === 0 ? base : base + 1;
}

/** Same ranking as scripts/bm25.py. `chunks` must already be pos <= upto. */
function search(chunks, query, k) {
  if (!chunks.length || k < 1) return [];
  const total = chunks.length;
  let avgdl = chunks.reduce((sum, chunk) => sum + chunk.n, 0) / total;
  if (avgdl <= 0) avgdl = 1;
  const df = {};
  for (const chunk of chunks) {
    for (const tok of Object.keys(chunk.tf)) df[tok] = (df[tok] || 0) + 1;
  }
  const qtf = {};
  for (const tok of tokenize(query)) qtf[tok] = (qtf[tok] || 0) + 1;
  const scored = chunks.map((chunk, index) => {
    let raw = 0;
    const length = chunk.n || 1;
    for (const [tok, qf] of Object.entries(qtf)) {
      const freq = chunk.tf[tok] || 0;
      if (!freq) continue;
      const nDocs = df[tok];
      const idf = Math.log(1 + (total - nDocs + 0.5) / (nDocs + 0.5));
      const denom = freq + K1 * (1 - B + B * length / avgdl);
      raw += idf * (freq * (K1 + 1)) / denom * qf;
    }
    return [roundPpm(raw), index];
  });
  scored.sort((a, b) => b[0] - a[0] || a[1] - b[1]);
  return scored.slice(0, k).map((row) => row[1]).sort((a, b) => a - b).map((index) => chunks[index]);
}

function excerpt(text, limit = 280) {
  const flat = (text || "").split(/\s+/).filter(Boolean).join(" ");
  if (flat.length <= limit) return flat;
  const cut = flat.slice(0, limit);
  const space = cut.lastIndexOf(" ");
  return (space > 40 ? cut.slice(0, space) : cut) + "…";
}

async function getJSON(url) {
  state.fetched.push(url);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} returned ${response.status}`);
  return response.json();
}

const cache = new Map();

function forgetAfter(n) {
  for (const key of [...cache.keys()]) {
    if (key > n) cache.delete(key);
  }
}

async function chunksThrough(n) {
  const chunks = [];
  for (let i = 0; i <= n; i += 1) {
    if (!cache.has(i)) cache.set(i, await getJSON(`${CFG.index}u${i}.json`));
    const doc = cache.get(i);
    if (doc.pos !== i) throw new Error(`${CFG.index}u${i}.json is pos ${doc.pos}`);
    chunks.push(...doc.chunks);
  }
  return chunks;
}

function chatFrom(char) {
  return (state.manifest && state.manifest.chat_from[char]) || { victor: 4, creature: 13 }[char];
}

function lockLine(char) {
  return (state.manifest && state.manifest.lock_lines[char]) || "";
}

function isLocked(char, pos) {
  return pos < chatFrom(char);
}

function charLabel(char) {
  return char === "creature" ? "the creature" : "Victor";
}

function lockNote() {
  const n = state.pos;
  const unit = state.units.find((row) => row.pos === n);
  const title = unit ? unit.title : `position ${n}`;
  if (isLocked(state.char, n)) {
    $("lock-note").textContent = `${charLabel(state.char)} is not on the page yet (${title}). The reply is the lock line. No model call. Chunks after position ${n} are not loaded.`;
  } else {
    $("lock-note").textContent = `${charLabel(state.char)} may speak. Answers and sources use positions 0–${n} (${title}) only. Later units are not loaded.`;
  }
}

function renderUnits() {
  const box = $("chapters");
  box.replaceChildren();
  for (const row of state.units) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = row.title || `pos ${row.pos}`;
    button.className = row.pos === state.pos ? "on" : "";
    button.addEventListener("click", () => selectPos(row.pos));
    box.appendChild(button);
  }
}

function renderChars() {
  const box = $("chars");
  if (!box) return;
  box.replaceChildren();
  for (const char of ["victor", "creature"]) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = charLabel(char);
    button.className = char === state.char ? "on" : "";
    button.addEventListener("click", () => selectChar(char));
    box.appendChild(button);
  }
}

function renderChips() {
  const box = $("chips");
  box.replaceChildren();
  const items = SUGGEST[state.char] || [];
  for (const item of items) {
    if (item.from > state.pos) continue;
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = item.q;
    button.addEventListener("click", () => {
      $("q").value = item.q;
      $("ask").requestSubmit();
    });
    box.appendChild(button);
  }
}

function selectPos(n) {
  if (n === state.pos || state.busy) return;
  state.pos = n;
  state.messages = [];
  forgetAfter(n);
  $("log").replaceChildren();
  lockNote();
  renderUnits();
  renderChips();
}

function selectChar(char) {
  if (char === state.char || state.busy) return;
  state.char = char;
  state.messages = [];
  $("log").replaceChildren();
  $("talk-label").textContent = charLabel(char);
  lockNote();
  renderChars();
  renderChips();
}

function addTurn(role, text, passages) {
  const turn = document.createElement("div");
  turn.className = "turn" + (role === "user" ? " you" : "");
  const who = document.createElement("div");
  who.className = "who";
  who.textContent = role === "user" ? "You" : charLabel(state.char);
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  turn.append(who, bubble);
  if (passages && passages.length) {
    const sources = document.createElement("div");
    sources.className = "sources";
    const heading = document.createElement("h3");
    heading.textContent = "Sources";
    sources.appendChild(heading);
    for (const passage of passages) {
      const cite = document.createElement("div");
      cite.className = "cite";
      const id = document.createElement("div");
      id.className = "id";
      id.textContent = `${passage.id} · ${passage.title || "pos " + passage.pos}`;
      const body = document.createElement("p");
      body.textContent = passage.excerpt;
      cite.append(id, body);
      sources.appendChild(cite);
    }
    turn.appendChild(sources);
  }
  $("log").appendChild(turn);
  turn.scrollIntoView({ block: "nearest" });
}

function promptText(prefix, hits, history, question) {
  const block = hits.map((hit) => `[${hit.title || "pos " + hit.pos}] ${hit.text}`).join("\n---\n");
  let out = `<|im_start|>system\n${prefix}${MARKER}${block}<|im_end|>\n`;
  for (const turn of history) {
    out += `<|im_start|>${turn.role}\n`;
    if (turn.role === "assistant") out += "<think>\n\n</think>\n\n";
    out += `${turn.content}<|im_end|>\n`;
  }
  out += `<|im_start|>user\n${question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
  return out;
}

function fitHistory(prefix, hits, question) {
  const history = state.messages.slice();
  while (history.length && promptText(prefix, hits, history, question).length > 14000) {
    history.shift();
  }
  return history;
}

function visibleAnswer(text) {
  let out = text || "";
  const end = out.lastIndexOf("</think>");
  if (end >= 0) out = out.slice(end + "</think>".length);
  else out = out.replace(/<think>[\s\S]*$/, "");
  return out.replace(/<\|im_end\|>/g, "").replace(/<\|endoftext\|>/g, "").trim();
}

function setStatus(text, isError) {
  const node = $("status");
  node.textContent = text;
  node.className = isError ? "note err" : "note";
}

async function loadModel() {
  if (state.engine || state.loading) return state.engine;
  if (!navigator.gpu) {
    throw new Error("This browser has no WebGPU. Use a current Chrome or Edge on this machine.");
  }
  state.loading = true;
  $("load").disabled = true;
  setStatus("Loading stock Qwen3-4B into this browser. Place mlc-ai/Qwen3-4B-q4f16_1-MLC in models/Qwen3-4B-q4f16_1-MLC/ first. About 2.2 GB.");
  try {
    const { CreateMLCEngine } = await import(new URL(CFG.vendor, location.href).href);
    const modelId = "qwen3-4b-stock-q4f16_1";
    state.engine = await CreateMLCEngine(modelId, {
      appConfig: {
        model_list: [{
          model: new URL(CFG.model, location.href).href,
          model_id: modelId,
          model_lib: new URL(CFG.wasm, location.href).href,
          overrides: { context_window_size: 4096, prefill_chunk_size: 1024 },
          vram_required_MB: 3500,
        }],
      },
      initProgressCallback: (report) => setStatus(report.text || "Loading the model"),
    });
    setStatus("Stock Qwen3-4B is loaded. No voice LoRA. Ask about the pages you have read.");
    return state.engine;
  } catch (error) {
    $("load").disabled = false;
    throw error;
  } finally {
    state.loading = false;
  }
}

async function ask(question) {
  const text = (question || "").trim();
  if (!text || state.busy) return;
  state.busy = true;
  $("send").disabled = true;
  $("q").value = "";
  addTurn("user", text);
  const n = state.pos;
  const who = state.char;
  try {
    if (isLocked(who, n)) {
      const line = lockLine(who);
      addTurn("assistant", line, []);
      setStatus(`Lock line. ${charLabel(who)} is not yet a speaker at position ${n}.`);
      return;
    }
    if (!state.engine) {
      setStatus("Load the model before asking, or pick a position before the character speaks to see the lock line.", true);
      addTurn("assistant", "Load the stock Qwen3-4B weights first. They belong in models/Qwen3-4B-q4f16_1-MLC/ (Hugging Face mlc-ai/Qwen3-4B-q4f16_1-MLC). This page does not download them.");
      return;
    }
    setStatus(`${charLabel(who)} is writing.`);
    const chunks = await chunksThrough(n);
    if (chunks.some((chunk) => chunk.pos > n)) throw new Error("search corpus passed the chapter lock");
    const hits = search(chunks, text, 6);
    if (hits.some((hit) => hit.pos > n)) throw new Error("a later unit entered the prompt");
    const prefix = (await getJSON(`${CFG.prompt}${who}-${n}.json`)).prefix;
    const history = fitHistory(prefix, hits, text);
    const prompt = promptText(prefix, hits, history, text);
    const response = await state.engine.completions.create({
      prompt,
      max_tokens: 200,
      temperature: 0.7,
      top_p: 0.95,
      seed: 1,
      stop: ["<|im_end|>", "<|endoftext|>"],
    });
    const answer = visibleAnswer(response.choices[0].text) || "…";
    const passages = hits.map((hit) => ({
      id: hit.id,
      pos: hit.pos,
      title: hit.title,
      excerpt: excerpt(hit.text),
    }));
    state.messages.push({ role: "user", content: text }, { role: "assistant", content: answer });
    addTurn("assistant", answer, passages);
    setStatus(`Answered from positions 0–${n}. ${passages.length} sources. Stock Qwen3-4B, no voice LoRA.`);
  } catch (error) {
    addTurn("assistant", error.message || String(error));
    setStatus(error.message || String(error), true);
  } finally {
    state.busy = false;
    $("send").disabled = false;
    $("q").focus();
  }
}

async function runSelftest() {
  const box = $("selftest");
  box.hidden = false;
  try {
    state.fetched = [];
    const spec = await getJSON(`${CFG.index}selftest.json`);
    const fail = [];
    for (const c of spec.cases) {
      const got = c.upto < spec.chat_from[c.char];
      if (got !== c.expect_lock) fail.push(`${c.id} lock`);
    }
    state.pos = 3;
    const chunks3 = await chunksThrough(3);
    const lateFetch = state.fetched.filter((url) => /u(?:[4-9]|[12][0-9])\.json/.test(url));
    if (lateFetch.length) fail.push(`later fetches ${lateFetch.join(" ")}`);
    if (chunks3.some((chunk) => chunk.pos > 3)) fail.push("upto=3 corpus leaked");
    const s3 = spec.search.find((row) => row.id === "no-letter-iv-before-pos-4");
    const ids3 = search(chunks3, s3.query, s3.k).map((h) => h.id);
    if (ids3.join(",") !== s3.ids.join(",")) fail.push(`upto=3 ids ${ids3.join(",")}`);
    if (search(chunks3, s3.query, s3.k).some((h) => h.pos > 3)) fail.push("upto=3 hit pos>3");

    state.fetched = [];
    state.pos = 12;
    const chunks12 = await chunksThrough(12);
    if (chunks12.some((chunk) => chunk.pos > 12)) fail.push("upto=12 corpus leaked");
    if (chunks12.some((chunk) => chunk.pos >= 13)) fail.push("creature unit visible before pos 13");
    const s12 = spec.search.find((row) => row.id === "no-vol3-at-pos-12");
    const hits12 = search(chunks12, s12.query, s12.k);
    if (hits12.some((h) => h.pos > 12)) fail.push("upto=12 hit pos>12");
    const ids12 = hits12.map((h) => h.id);
    if (ids12.join(",") !== s12.ids.join(",")) fail.push(`upto=12 ids ${ids12.join(",")}`);

    if (fail.length) throw new Error(fail.join("; "));
    box.textContent = "PASS chapter lock (Victor hidden before pos 4, creature hidden before pos 13, later chunks not returned)";
    box.dataset.result = "pass";
  } catch (error) {
    box.textContent = `FAIL ${error.message}`;
    box.dataset.result = "fail";
  }
}

async function main() {
  if (!navigator.gpu) {
    setStatus("No WebGPU in this browser. You can still pick a chapter and see lock lines. The stock Qwen3-4B model needs Chrome or Edge with WebGPU.");
  }
  const manifest = await getJSON(`${CFG.index}manifest.json`);
  state.manifest = manifest;
  state.units = manifest.units;
  if (!state.units.some((row) => row.pos === state.pos)) {
    state.pos = state.units[0].pos;
  }
  lockNote();
  renderUnits();
  renderChars();
  renderChips();
  $("ask").addEventListener("submit", (event) => {
    event.preventDefault();
    ask($("q").value);
  });
  $("load").addEventListener("click", () => {
    loadModel().catch((error) => setStatus(error.message || String(error), true));
  });
  $("reset").addEventListener("click", () => {
    state.messages = [];
    $("log").replaceChildren();
  });
  if (params.get("selftest") === "1") await runSelftest();
}

main().catch((error) => setStatus(error.message || String(error), true));
