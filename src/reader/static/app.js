const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const forcePack = params.get("mode") === "pack";
const packUrl = params.get("pack") || "/packs/sample.json";
const MISS = "That question is not in this static pack. No model was called.";
const PENDING = "This question is listed in the pack, but no answer has been precomputed yet. No model was called.";

const state = {
  mode: "live",
  clientPack: false,
  sid: null,
  chapter: 3,
  chapters: [],
  suggestions: [],
  pack: null,
  messages: [],
  busy: false,
};

function norm(text) {
  return (text || "").trim().toLowerCase().split(/\s+/).filter(Boolean).join(" ");
}

function storeGet(key) {
  try { return sessionStorage.getItem(key); } catch (error) { return null; }
}

function storeSet(key, value) {
  try { sessionStorage.setItem(key, value); } catch (error) { /* private mode */ }
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

async function post(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail || response.statusText));
  }
  return data;
}

function lookup(question) {
  const want = norm(question);
  const items = (state.pack && state.pack.items) || [];
  for (const item of items) {
    if (Number(item.chapter) !== Number(state.chapter)) continue;
    if ((item.perspective || "wren") !== "wren") continue;
    if (norm(item.question) !== want) continue;
    if (!item.reply) {
      return { reply: PENDING, citations: [], guard_fallback: null };
    }
    return {
      reply: item.reply,
      citations: item.citations || [],
      guard_fallback: !!item.guard_fallback,
    };
  }
  return { reply: MISS, citations: [], guard_fallback: null };
}

function lockNote() {
  const n = state.chapter;
  $("lock-note").textContent = `Wren is at chapter ${n}. Answers and sources use chapters 1–${n} only. Later chapters stay out.`;
}

function renderChapters() {
  const box = $("chapters");
  box.replaceChildren();
  for (const row of state.chapters) {
    const button = el("button", row.n === state.chapter ? "on" : "", `${row.n} ${row.title.replace(/^Chapter\s+\d+\s+[—-]\s+/, "")}`);
    button.type = "button";
    button.setAttribute("aria-pressed", row.n === state.chapter ? "true" : "false");
    button.onclick = () => selectChapter(row.n);
    box.append(button);
  }
  lockNote();
}

function renderChips() {
  const box = $("chips");
  box.replaceChildren();
  const rows = state.suggestions.filter((row) => Number(row.chapter) === Number(state.chapter));
  for (const row of rows) {
    const button = el("button", "", row.question);
    button.type = "button";
    button.onclick = () => ask(row.question);
    box.append(button);
  }
}

function renderLog() {
  const log = $("log");
  log.replaceChildren();
  if (!state.messages.length) {
    log.append(el("p", "note", "Ask Wren something from the chapters you have reached."));
    return;
  }
  for (const message of state.messages) {
    const turn = el("div", message.role === "user" ? "turn you" : "turn");
    turn.append(el("div", "who", message.role === "user" ? "You" : "Wren"));
    const bubble = el("div", "bubble" + (message.error ? " err" : ""), message.content || "");
    turn.append(bubble);
    if (message.role === "assistant" && message.citations && message.citations.length) {
      const sources = el("div", "sources");
      sources.append(el("h3", "", "Sources"));
      for (const cite of message.citations) {
        const block = el("div", "cite");
        block.append(el("div", "id", `Chapter ${cite.chapter} · ${cite.id}`));
        block.append(el("p", "", cite.excerpt || ""));
        sources.append(block);
      }
      turn.append(sources);
    }
    if (message.role === "assistant" && message.guard_fallback != null) {
      const line = message.guard_fallback
        ? "Wording check kept the first answer."
        : "Wording check kept the rewrite.";
      turn.append(el("div", "check", line));
    }
    log.append(turn);
  }
  log.lastElementChild && log.lastElementChild.scrollIntoView({ block: "nearest" });
}

function render() {
  const label = state.clientPack ? "static pack" : (state.mode === "pack" ? "pack" : "live");
  $("mode").textContent = label;
  renderChapters();
  renderChips();
  renderLog();
}

function applyServerMessages(messages) {
  state.messages = (messages || []).map((message) => ({
    role: message.role,
    content: message.content,
    citations: message.citations || [],
    guard_fallback: message.guard_fallback,
  }));
}

async function selectChapter(chapter) {
  if (state.busy) return;
  state.chapter = chapter;
  storeSet("nr-chapter", String(chapter));
  if (!state.clientPack && state.sid) {
    const data = await post("/api/chapter", { sid: state.sid, chapter });
    applyServerMessages(data.messages);
  } else {
    state.messages = [];
  }
  render();
}

async function ask(text) {
  const question = (text || "").trim();
  if (!question || state.busy) return;
  $("q").value = "";
  state.busy = true;
  $("ask").querySelector("button").disabled = true;
  state.messages.push({ role: "user", content: question });
  const wait = { role: "assistant", content: state.clientPack || state.mode === "pack" ? "Looking in the pack…" : "Wren is thinking…", citations: [] };
  state.messages.push(wait);
  render();
  try {
    let answer;
    if (state.clientPack) {
      answer = lookup(question);
    } else {
      answer = await post("/api/chat", {
        sid: state.sid,
        text: question,
        chapter: state.chapter,
        perspective: "wren",
      });
    }
    wait.content = answer.reply;
    wait.citations = answer.citations || [];
    wait.guard_fallback = answer.guard_fallback;
  } catch (error) {
    wait.content = error.message || String(error);
    wait.error = true;
  } finally {
    state.busy = false;
    $("ask").querySelector("button").disabled = false;
    render();
  }
}

async function loadPack() {
  const response = await fetch(packUrl);
  if (!response.ok) throw new Error("Pack file is missing.");
  state.pack = await response.json();
  if (!state.chapters.length && state.pack.chapters) state.chapters = state.pack.chapters;
  state.suggestions = (state.pack.items || []).map((item) => ({
    id: item.id,
    chapter: item.chapter,
    question: item.question,
    spoiler_probe: !!item.spoiler_probe,
  }));
}

async function boot() {
  if (forcePack) {
    state.clientPack = true;
    state.mode = "pack";
    try {
      await loadPack();
    } catch (error) {
      state.messages = [{ role: "assistant", content: error.message, error: true }];
    }
    try {
      const response = await fetch("/api/config");
      if (response.ok) {
        const config = await response.json();
        if (!state.chapters.length) state.chapters = config.chapters || [];
      }
    } catch (error) {
      /* static file host: the pack carries the chapter list */
    }
    if (!state.chapters.length) {
      state.chapters = [1, 2, 3, 4, 5].map((n) => ({ n, title: `Chapter ${n}` }));
    }
    render();
    return;
  }
  try {
    const response = await fetch("/api/config");
    if (!response.ok) throw new Error("no api");
    const config = await response.json();
    state.mode = config.mode || "live";
    state.chapters = config.chapters || [];
    state.suggestions = config.suggestions || [];
    const savedChapter = Number(storeGet("nr-chapter"));
    state.chapter = state.chapters.some((row) => row.n === savedChapter)
      ? savedChapter
      : (config.default_chapter || 3);
    const saved = storeGet("nr-sid");
    if (saved) {
      const hist = await fetch(`/api/history?sid=${encodeURIComponent(saved)}&chapter=${state.chapter}`);
      if (hist.ok) {
        state.sid = saved;
        applyServerMessages((await hist.json()).messages);
      }
    }
    if (!state.sid) {
      const session = await post("/api/session", { chapter: state.chapter });
      state.sid = session.sid;
      storeSet("nr-sid", state.sid);
      storeSet("nr-chapter", String(state.chapter));
      state.messages = [];
    }
  } catch (error) {
    state.clientPack = true;
    state.mode = "pack";
    try {
      await loadPack();
    } catch (packError) {
      state.messages = [{ role: "assistant", content: "The reader API is not running, and the pack could not be loaded.", error: true }];
    }
    if (!state.chapters.length) {
      state.chapters = [1, 2, 3, 4, 5].map((n) => ({ n, title: `Chapter ${n}` }));
    }
  }
  render();
}

$("ask").addEventListener("submit", (event) => {
  event.preventDefault();
  ask($("q").value);
});

$("reset").addEventListener("click", async () => {
  if (state.clientPack || !state.sid) {
    state.messages = [];
    render();
    return;
  }
  await post("/api/reset", { sid: state.sid });
  state.messages = [];
  render();
});

boot();
