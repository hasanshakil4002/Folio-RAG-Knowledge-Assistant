"""FastAPI server: chat (SSE stream), PDF upload (progress stream), document management."""
import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config as cfg
from .cleaner import StreamCleaner
from .ingest import chunk_text, iter_pages, ocr_ready
from .memory import Memory
from .providers import Hub
from .rag import (LANGS, MODES, NO_KB, build_messages, embed_and_index,
                  max_tokens_for, retrieve, rewrite_question)
from .vectorstore import Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rag.main")

hub = Hub()
store = Store()
memory = Memory()

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@asynccontextmanager
async def lifespan(_: FastAPI):
    ok = await hub.refresh_ollama()
    s = hub.status()
    log.info("LLM chain: %s", [p.name for p in hub.llm_chain()])
    log.info("Embedding chain: %s", [p.name for p in hub.embed_chain()])
    if ok:
        log.info("Ollama models: %s", hub.ollama_models)
    else:
        log.warning("Ollama not reachable at %s", cfg.OLLAMA_BASE_URL)
    log.info("Status: %s", {k: s[k] for k in ("llm", "embedding", "ready")})
    yield


app = FastAPI(title="RAG Knowledge Assistant", lifespan=lifespan)


# ---------------------------------------------------------------- status / docs
@app.get("/api/status")
async def status():
    await hub.refresh_ollama()
    return {**hub.status(), "documents": len(store.docs), "chunks": len(store.chunks)}


@app.get("/api/documents")
async def documents():
    return sorted(store.docs.values(), key=lambda d: d["added"])


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    if not await asyncio.to_thread(store.delete_document, doc_id):
        raise HTTPException(404, "Document not found")
    return {"ok": True}


@app.get("/api/history/{sid}")
async def history(sid: str):
    return memory.get(sid)


@app.delete("/api/history/{sid}")
async def clear_history(sid: str):
    memory.clear(sid)
    return {"ok": True}


# ---------------------------------------------------------------- upload
@app.post("/api/upload")
async def upload(file: UploadFile = File(...), ocr: str = Form("false")):
    content = await file.read()
    name = file.filename or "document.pdf"
    use_ocr = ocr.lower() in ("1", "true", "yes", "on")

    if content[:5] != b"%PDF-":
        raise HTTPException(400, "Only PDF files are accepted.")
    if len(content) > cfg.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(400, f"The file must be smaller than {cfg.MAX_UPLOAD_MB} MB.")
    if use_ocr:
        ok, msg = ocr_ready()
        if not ok:
            raise HTTPException(400, msg)

    fname = f"{uuid.uuid4().hex}.pdf"
    path = cfg.UPLOAD_DIR / fname
    path.write_bytes(content)

    async def gen():
        doc = None
        try:
            yield sse("progress", {"pct": 2, "msg": "Extracting text with OCR" if use_ocr
                                   else "Extracting text"})
            span = 55 if use_ocr else 30
            items, pages = [], 0
            it = iter_pages(path, use_ocr)
            while True:
                r = await asyncio.to_thread(next, it, None)
                if r is None:
                    break
                no, total, text = r
                pages = total
                items += [(no, c) for c in chunk_text(text, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)]
                yield sse("progress", {"pct": 2 + int(no / total * span),
                                       "msg": f"Read page {no} of {total}"})

            if not items:
                hint = "" if use_ocr else " If this is a scanned PDF, choose 'OCR needed' and try again."
                raise ValueError("No text could be extracted from this PDF." + hint)

            base = 2 + span
            yield sse("progress", {"pct": base, "msg": f"Embedding {len(items)} chunks"})
            doc = await asyncio.to_thread(store.add_document, name, items, use_ocr, pages, fname)
            used = None
            async for ev in embed_and_index(hub, store):
                used = ev["provider"]
                pct = base + int(ev["done"] / max(ev["total"], 1) * (100 - base))
                yield sse("progress", {"pct": min(pct, 99), "msg": f"Adding to knowledge base ({used})"})
            yield sse("done", {"doc": doc, "embedding": used})
        except Exception as e:  # noqa: BLE001
            log.exception("Upload failed")
            if doc:  # roll back a half-added document so the knowledge base stays consistent
                await asyncio.to_thread(store.delete_document, doc["id"])
            else:
                path.unlink(missing_ok=True)
            yield sse("error", {"message": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


# ---------------------------------------------------------------- chat
class ChatReq(BaseModel):
    session_id: str
    question: str
    language: str = "en"
    mode: str = "short"


@app.post("/api/chat")
async def chat(req: ChatReq):
    q = req.question.strip()
    lang = req.language if req.language in LANGS else "en"
    mode = req.mode if req.mode in MODES else "short"
    if not q:
        raise HTTPException(400, "The question cannot be empty.")
    sid = req.session_id[:64]

    async def gen():
        try:
            if not store.chunks:
                yield sse("token", {"t": NO_KB})
                yield sse("done", {})
                return

            hist = memory.get(sid)
            standalone = await rewrite_question(hub, hist, q)
            hits, emb = await retrieve(hub, store, standalone, cfg.TOP_K + (2 if mode == "research" else 0))

            seen, sources = set(), []
            for c, score in hits:
                k = (c["doc_id"], c["page"])
                if k not in seen:
                    seen.add(k)
                    sources.append({"doc": c["doc_name"], "page": c["page"], "score": round(score, 3)})
            yield sse("sources", {"sources": sources, "embedding": emb})

            messages = build_messages(q, standalone, hits, hist, lang, mode)
            max_tokens = max_tokens_for(mode, lang)

            answer, ok = "", False
            for p in hub.llm_chain():
                cleaner, buf, emitted = StreamCleaner(), "", False
                try:
                    async for tok in hub.stream(p, messages, max_tokens):
                        out = cleaner.feed(tok)
                        if out:
                            if not emitted:
                                emitted = True
                                yield sse("provider", {"llm": p.name, "model": p.llm_model})
                            buf += out
                            yield sse("token", {"t": out})
                    tail = cleaner.flush()
                    if tail:
                        if not emitted:
                            yield sse("provider", {"llm": p.name, "model": p.llm_model})
                        buf += tail
                        yield sse("token", {"t": tail})
                    if not buf.strip():
                        raise RuntimeError("Empty response")
                    answer, ok = buf, True
                    break
                except Exception as e:  # noqa: BLE001
                    log.warning("LLM %s failed: %s", p.name, e)
                    p.mark_failed()
                    if emitted:  # failed mid-stream: the UI clears the partial text and restarts
                        yield sse("reset", {})

            if not ok:
                yield sse("error", {"message": "No LLM provider is working. Check your API keys or make sure Ollama is running."})
                return

            memory.add(sid, q, answer)
            yield sse("done", {})
        except Exception as e:  # noqa: BLE001
            log.exception("Chat failed")
            yield sse("error", {"message": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


# ---------------------------------------------------------------- frontend
app.mount("/", StaticFiles(directory=cfg.FRONTEND_DIR, html=True), name="ui")
