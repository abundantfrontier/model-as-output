const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const K1 = 1.2;
const B = 0.75;
const MARKER = "\n\nStory so far (excerpts):\n";
const SUGGEST = [
  { from: 1, q: "What did Diane say about the air today?" },
  { from: 1, q: "What heading did you type at the console?" },
  { from: 3, q: "Who was standing at the gauges when you got there?" },
];

const state = {
  chapter: 3,
  chapters: [],
  messages: [],
  engine: null,
  loading: false,
  busy: false,
  fetched: [],
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

/** Same ranking as scripts/bm25.py. `chunks` must already be chapters 1..N only. */
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

async function chaptersThrough(n) {
  const chunks = [];
  for (let i = 1; i <= n; i += 1) {
    if (!cache.has(i)) cache.set(i, await getJSON(`index/ch${i}.json`));
    const doc = cache.get(i);
    if (doc.chapter !== i) throw new Error(`index/ch${i}.json is chapter ${doc.chapter}`);
    chunks.push(...doc.chunks);
  }
  return chunks;
}

function lockNote() {
  const n = state.chapter;
  $("lock-note").textContent = `Wren is at chapter ${n}. Answers and sources use chapters 1–${n} only. Later chapters are not loaded.`;
}

function renderChapters() {
  const box = $("chapters");
  box.replaceChildren();
  for (const row of state.chapters) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = row.title || `Chapter ${row.n}`;
    button.className = row.n === state.chapter ? "on" : "";
    button.addEventListener("click", () => selectChapter(row.n));
    box.appendChild(button);
  }
}

function renderChips() {
  const box = $("chips");
  box.replaceChildren();
  for (const item of SUGGEST) {
    if (item.from > state.chapter) continue;
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

function selectChapter(n) {
  if (n === state.chapter || state.busy) return;
  state.chapter = n;
  state.messages = [];
  forgetAfter(n);
  $("log").replaceChildren();
  lockNote();
  renderChapters();
  renderChips();
}

function addTurn(role, text, passages) {
  const turn = document.createElement("div");
  turn.className = "turn" + (role === "user" ? " you" : "");
  const who = document.createElement("div");
  who.className = "who";
  who.textContent = role === "user" ? "You" : "Wren";
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
      id.textContent = `${passage.id} · chapter ${passage.chapter}`;
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
  const block = hits.map((hit) => `[day ${hit.chapter}] ${hit.text}`).join("\n---\n");
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
  setStatus("Loading Wren into this browser. The first time reads about 2.2 GB from this site.");
  try {
    const { CreateMLCEngine } = await import("./vendor/web-llm.js");
    const modelId = "wren-q4f16_1";
    state.engine = await CreateMLCEngine(modelId, {
      appConfig: {
        model_list: [{
          model: new URL("models/wren-q4f16_1-MLC/", location.href).href,
          model_id: modelId,
          model_lib: new URL("models/Qwen3-4B-q4f16_1_cs1k-webgpu.wasm", location.href).href,
          overrides: { context_window_size: 4096, prefill_chunk_size: 1024 },
          vram_required_MB: 3500,
        }],
      },
      initProgressCallback: (report) => setStatus(report.text || "Loading the model"),
    });
    setStatus("Wren is loaded in this browser. Ask about the chapters you have read.");
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
  if (!state.engine) {
    setStatus("Load the model before asking. That stays on this machine.", true);
    return;
  }
  state.busy = true;
  $("send").disabled = true;
  $("q").value = "";
  addTurn("user", text);
  setStatus("Wren is writing.");
  try {
    const n = state.chapter;
    const chunks = await chaptersThrough(n);
    if (chunks.some((chunk) => chunk.chapter > n)) throw new Error("search corpus passed the chapter lock");
    const hits = search(chunks, text, 6);
    if (hits.some((hit) => hit.chapter > n)) throw new Error("a later chapter entered the prompt");
    const prefix = (await getJSON(`prompt/wren-${n}.json`)).prefix;
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
      chapter: hit.chapter,
      excerpt: excerpt(hit.text),
    }));
    state.messages.push({ role: "user", content: text }, { role: "assistant", content: answer });
    addTurn("assistant", answer, passages);
    setStatus(`Answered from chapters 1–${n}. ${passages.length} sources.`);
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
    state.chapter = 3;
    const chunks = await chaptersThrough(3);
    await getJSON("prompt/wren-3.json");
    const spec = await getJSON("index/selftest.json");
    const hits = search(chunks, spec.query, spec.k);
    const ids = hits.map((hit) => hit.id);
    const badFetch = state.fetched.filter((url) => /(?:ch|wren-)[4-9]/.test(url));
    const late = hits.filter((hit) => hit.chapter > 3);
    const corpusLate = chunks.filter((chunk) => chunk.chapter > 3);
    if (badFetch.length || late.length || corpusLate.length || ids.join(",") !== spec.ids.join(",")) {
      throw new Error(`ids ${ids.join(",")} expected ${spec.ids.join(",")}; later fetches ${badFetch.join(" ") || "none"}`);
    }
    box.textContent = "PASS chapter lock";
    box.dataset.result = "pass";
  } catch (error) {
    box.textContent = `FAIL ${error.message}`;
    box.dataset.result = "fail";
  }
}

async function main() {
  if (!navigator.gpu) {
    setStatus("No WebGPU in this browser. You can still pick a chapter. Wren's model needs Chrome or Edge with WebGPU.");
  }
  const manifest = await getJSON("index/manifest.json");
  state.chapters = manifest.chapters;
  if (!state.chapters.some((row) => row.n === state.chapter)) {
    state.chapter = state.chapters[state.chapters.length - 1].n;
  }
  lockNote();
  renderChapters();
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
