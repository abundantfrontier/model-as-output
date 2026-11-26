"""FastAPI app for the No Return reader.

Live mode runs the locked pipeline. Pack mode answers only from the JSON pack.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import gpu_lock
import logbook
import packstore
import pipeline
import retrieve
import sessions
from paths import HOST, LOGS, MODE, PACK_PATH, PIPE, PORT, ROOT, SESSIONS, STATIC, STORY
from perspectives import enabled, get as perspective_get

if MODE not in ("live", "pack"):
    raise SystemExit(f"READER_MODE must be live or pack, got {MODE}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    SESSIONS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    (ROOT / "server.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    print(f"No Return reader mode={MODE} http://{HOST}:{PORT}", flush=True)
    yield


app = FastAPI(title="No Return reader", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def chapters() -> list[dict]:
    return retrieve.chapter_catalog()


def chapter_ok(chapter: int) -> int:
    nums = [row["n"] for row in chapters()]
    if chapter not in nums:
        raise HTTPException(400, f"chapter must be one of {nums}")
    return chapter


def config_payload() -> dict:
    pack = packstore.load_pack(PACK_PATH)
    return {
        "story": STORY,
        "tagline": "Chapter-locked Wren",
        "mode": MODE,
        "public_deploy": False,
        "pipeline": PIPE,
        "perspective": "wren",
        "perspectives": enabled(),
        "chapters": pack.get("chapters") or chapters(),
        "default_chapter": 3,
        "suggestions": packstore.suggestions(pack),
        "pack": "packs/sample.json",
        "host": HOST,
        "port": PORT,
    }


class SessionIn(BaseModel):
    chapter: int = 3


class ChapterIn(BaseModel):
    sid: str
    chapter: int


class ChatIn(BaseModel):
    sid: str
    text: str = Field(max_length=1000)
    chapter: int | None = None
    perspective: str = "wren"


class ResetIn(BaseModel):
    sid: str


def load_session(sid: str) -> dict:
    if not sessions.valid_sid(sid):
        raise HTTPException(400, "bad sid")
    try:
        return sessions.load(sid)
    except FileNotFoundError:
        raise HTTPException(404, "unknown session")


@app.get("/")
def root():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    status = gpu_lock.lock_status()
    return {
        "ok": True,
        "story": STORY,
        "mode": MODE,
        "public_deploy": False,
        "gpu_lock_held": bool(status.get("held")),
        "gpu_lock_stale": bool(status.get("stale")),
    }


@app.get("/api/config")
def config():
    return config_payload()


@app.post("/api/session")
def new_session(body: SessionIn | None = None):
    chapter = chapter_ok((body.chapter if body else None) or 3)
    row = sessions.create(chapter)
    logbook.write("session", {"session": row["sid"], "chapter": chapter, "mode": MODE})
    return {"sid": row["sid"], "chapter": row["chapter"], "perspective": "wren", "mode": MODE}


@app.post("/api/chapter")
def set_chapter(body: ChapterIn):
    chapter = chapter_ok(body.chapter)
    row = load_session(body.sid)
    row["chapter"] = chapter
    sessions.save(row)
    return {"sid": row["sid"], "chapter": chapter, "messages": sessions.history(row, chapter)}


@app.get("/api/history")
def history(sid: str, chapter: int | None = None):
    row = load_session(sid)
    chapter = chapter_ok(chapter if chapter is not None else int(row.get("chapter") or 3))
    return {"sid": sid, "chapter": chapter, "messages": sessions.history(row, chapter)}


@app.post("/api/reset")
def reset(body: ResetIn):
    row = load_session(body.sid)
    chapter = int(row.get("chapter") or 3)
    sessions.clear_chapter(row, chapter)
    sessions.save(row)
    logbook.write("reset", {"session": body.sid, "chapter": chapter, "mode": MODE})
    return {"ok": True, "chapter": chapter}


@app.post("/api/chat")
def chat(body: ChatIn):
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(400, "empty question")
    try:
        perspective_get(body.perspective)
    except KeyError:
        raise HTTPException(400, "This build only answers as Wren.")
    row = load_session(body.sid)
    chapter = chapter_ok(body.chapter if body.chapter is not None else int(row.get("chapter") or 3))
    row["chapter"] = chapter
    row["perspective"] = "wren"
    prior = sessions.history(row, chapter)
    if MODE == "pack":
        found = packstore.lookup(packstore.load_pack(PACK_PATH), chapter, text, "wren")
        answer = {
            "reply": found["reply"],
            "citations": found["citations"],
            "guard_fallback": found["guard_fallback"],
            "chapter": chapter,
            "perspective": "wren",
            "mode": "pack",
            "model_called": False,
        }
        sessions.append_turn(row, chapter, text, answer)
        sessions.save(row)
        logbook.write("chat", {
            "session": body.sid,
            "chapter": chapter,
            "mode": "pack",
            "question": text,
            "reply": answer["reply"],
            "guard_fallback": answer["guard_fallback"],
            "citation_ids": [c.get("id") for c in answer["citations"]],
            "pack_hit": found["found"],
            "model_called": False,
        })
        return answer
    try:
        result = pipeline.answer_one(text, chapter, "wren", history=prior)
    except gpu_lock.GpuBusy as exc:
        logbook.write("gpu-busy", {"session": body.sid, "chapter": chapter, "question": text, "detail": exc.detail})
        raise HTTPException(503, "The GPU is in use. Wait and try again. No model was started.")
    except Exception as exc:
        logbook.write("error", {"session": body.sid, "chapter": chapter, "question": text, "error": str(exc)})
        raise HTTPException(500, f"Wren could not answer: {exc}")
    answer = pipeline.public_answer(result, "live")
    sessions.append_turn(row, chapter, text, answer)
    sessions.save(row)
    logbook.write("chat", {
        "session": body.sid,
        "chapter": chapter,
        "mode": "live",
        "question": text,
        "reply": answer["reply"],
        "guard_fallback": result.get("guard_fallback"),
        "citation_ids": [c.get("id") for c in answer["citations"]],
        "parse_ok": result.get("parse_ok"),
        "dropped": result.get("dropped"),
        "added": result.get("added"),
        "new_markers": result.get("new_markers"),
        "model_called": True,
    })
    return answer


app.mount("/static", StaticFiles(directory=STATIC), name="static")
Path(PACK_PATH).parent.mkdir(parents=True, exist_ok=True)
app.mount("/packs", StaticFiles(directory=Path(PACK_PATH).parent), name="packs")
