# No Return reader

A local reader for the *No Return* story. You pick a chapter, talk with Wren, and see the passages the answer used. Wren is chapter-locked: retrieval and the knowledge block only include chapters 1 through the chapter you picked.

This build runs on this machine. It is not a public deploy.

The answer engine is the locked two-pass pipeline `p2_light_dpo_guarded` (pass 1 `v_merged_v_rag`, pass 2 the exp7 light DPO rewrite, then the fact-diff guard). It does not use the proto server on port 8090, and it does not use the 14B Ollama chat as the voice that answers. Ollama `qwen3:14b` is only the fact-diff judge. Ollama `bge-m3` is only the retriever.

## Run it locally

From this directory:

```bash
cd ~/output-model/reader
./run.sh
```

Then open `http://127.0.0.1:8091`.

`run.sh` uses `~/output-model/exp3/.venv` (the locked torch stack) and adds the proto venv's site-packages so FastAPI can be imported. Nothing new is installed. If 8091 is taken, the script picks the next free port from 8092 upward, skips 8090 and 18765, and writes the port to `PORT`. The bind is `127.0.0.1` only.

Useful variables:

| Variable | Default | Role |
|---|---|---|
| `READER_MODE` | `live` | `live` or `pack` |
| `READER_HOST` | `127.0.0.1` | Loopback. Other hosts need `READER_ALLOW_PUBLIC=1` |
| `READER_PORT` | `8091` | Fixed port. Refuses 8090 |
| `READER_PACK` | `packs/sample.json` | Static pack file |
| `READER_LOCK_WAIT` | `180` | Seconds a live question waits for `~/gpu.lock` |
| `READER_SEED` | `1` | Torch seed set before each pass |

Logs: `logs/chat.jsonl` (session, chapter, reply, whether the guard fell back). Sessions: `sessions/`. The server pid is `server.pid`.

Stop the reader with `kill "$(cat server.pid)"`. Do not stop the proto server on port 8090.

## Live mode and pack mode

Live mode is the default. Each question takes `~/gpu.lock`, runs the locked pipeline on the 3090, cites the retrieved passages, and releases the lock. If another job holds the lock, the page gets a clear "GPU is in use" error and no model is loaded.

Pack mode never calls a model. The same page looks up a curated question in the JSON pack.

```bash
READER_MODE=pack ./run.sh
```

If the question is not in the pack, the reply says so. A listed question with no precomputed answer says that too.

The page can also do that lookup in the browser, with no FastAPI process:

```bash
cd ~/output-model/reader
python3 -m http.server 8092 --bind 127.0.0.1
```

Open `http://127.0.0.1:8092/static/index.html?mode=pack`. The pack is `/packs/sample.json`. `?mode=pack` does not call `/api/chat`.

## GPU lock

`~/gpu.lock` is shared with another GPU job. The line this app writes is:

```text
output-model <pid> <iso time> reader live wren
```

Pack generation uses `reader pack generation`. A smoke run uses `reader smoke`.

Rules this app follows:

- Before a model load, an embed, a fact-diff, or a pack run, it reads the lock.
- A live PID is left alone. The app waits, then stops without starting.
- A dead PID is stale and may be replaced.
- The lock is deleted only when the project name is `output-model` and the PID is still this process.
- UI, API, and static files do not take the lock.
- After the job, the 4B adapters are unloaded and `ollama stop` is called for `bge-m3` and `qwen3:14b` so the 3090 is free.

## How an answer is made

1. Pass 1 loads `exp6/merged_q4_facts` plus `exp6/lora_v_merged`. The prompt is the locked RAG prompt (`common.rag_sysprompt`): Wren's card, what Wren knows and believes at this chapter, and bge-m3 excerpts from chapters 1 through the selected chapter only. Sampling matches the locked stack: `do_sample`, temperature 0.7, top_p 0.95, 200 new tokens. The torch seed is set to 1 before the pass so a chat turn does not depend on earlier turns' random state.
2. Pass 2 unloads that adapter and loads `exp10/init/lora_v_light_dpo` on the same merged base (sha256 `7980fad31a4a3023cc738662a8af6220d1c0722eeaf70d4b204f7368fd29dcef`). The prompt is the locked rewrite: the question, the pass-1 answer, and the style-only instruction. No retrieved passages.
3. The fact-diff judge is local Ollama `qwen3:14b`, temperature 0, seed 1, JSON. The guard keeps the rewrite only when the judge reports no added claim and no dropped claim, the reply parses, and pass 2 did not introduce a `common.fail_hits` marker that pass 1 lacked. Otherwise the reply is the pass-1 answer.

Recent turns in the same chapter are included in pass 1 so the chat can continue. Pass 2 still rewrites only the current answer. Switching chapter does not carry the other chapter's transcript into the prompt.

The retriever reads the locked harness copy in `exp10/data` (the chapters that pipeline was locked against), via `common.chapter_text` and `common.knowledge`. The `no-return/` package is left untouched.

## Sample pack

Questions live in `questions/sample.json`: five chapter-3 Wren questions, including two later-chapter probes (`What happens in chapter 5?` and the Mara probe).

If `packs/sample.json` says `"status": "pending"`, the question list is there and the replies are not. Pack mode will say that no answer was precomputed. It still will not call a model. Fill the replies with `gen_pack.py` when `~/gpu.lock` is free. The build that produced this tree found the lock held by another GPU job, so it did not invent answers and did not run the model.

```bash
# question list only, no GPU
python gen_pack.py --pending

# fill replies; holds ~/gpu.lock for the whole run, then releases it
~/output-model/exp3/.venv/bin/python gen_pack.py
```

`python smoke.py` writes `logs/smoke.json` from the pack and does not use the GPU. `python smoke.py --live` runs the two smoke questions (the air question and the chapter-5 probe) under the lock.

Offline checks, also with no GPU:

```bash
~/output-model/exp3/.venv/bin/python test_offline.py
```

## Later hooks

`perspectives.py` is the registry. This build enables Wren only. Another perspective is a new entry with a `card` key and an optional `system_prompt`. `None` keeps the locked card. A string replaces only that card line; the style line, the guard, the day, and the chapter lock stay. Pass 1 still builds the RAG prompt. Pass 2 still refuses a retrieval block.

## Deploying later

The page and the API are already split. A GPU-free host can serve `static/` and `packs/` as files, or run `READER_MODE=pack`. A GPU host runs `./run.sh` with the weights still at the `READER_MERGED`, `READER_PASS1_ADAPTER`, and `READER_PASS2_ADAPTER` paths (defaults are the exp6 and exp10 directories). Point `READER_EXP10` at the locked harness so `common.py`, `prompts.py`, `chat.py`, and `factdiff.py` import from there. Do not edit those files. Ollama must still provide `bge-m3` and `qwen3:14b` for live mode.

Keep the public bind off until you choose a host. This tree does not open a tunnel and does not listen on 8090.
