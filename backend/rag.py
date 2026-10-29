"""Core RAG logic: indexing, retrieval, question rewriting and prompt building."""
import logging

from . import config as cfg
from .providers import Hub
from .vectorstore import Store

log = logging.getLogger("rag.core")

BULLET = "\u2022"

LANGS = {"en": "English", "bn": "Bengali", "hi": "Hindi"}
TOKEN_FACTOR = {"bn": 6, "hi": 5, "en": 2}  # rough tokens per word

MODES = {
    "short": {"label": "Short Answer", "target": 190,
              "rule": "Between 80 and 190 words. Give a direct, compact answer in one or two short paragraphs."},
    "describe": {"label": "Describe Answer", "target": 480,
                 "rule": "Between 250 and 480 words. Give a clear explanation with short paragraphs and, where useful, a bullet list."},
    "research": {"label": "Research Answer", "target": 900,
                 "rule": ("Between 550 and 900 words. Organize into sections, each with a short title line ending "
                          "with a colon: an overview, a detailed analysis, key points as bullet lines, and a conclusion.")},
}

NO_KB = "The knowledge base is empty. Add a PDF from the left panel, then ask your question."


def max_tokens_for(mode: str, lang: str) -> int:
    t = MODES[mode]["target"] * TOKEN_FACTOR.get(lang, 4) + 200
    return min(t, 7000)


# ---------- indexing ----------
async def sync_index(hub: Hub, store: Store, p):
    """Embed whatever is missing from provider p's index and add it; yields (done, total)."""
    ids = store.missing(p.embed_key)
    total = len(ids)
    for i in range(0, total, cfg.EMBED_BATCH):
        batch = ids[i:i + cfg.EMBED_BATCH]
        vecs = await hub.embed(p, [store.chunks[j]["text"] for j in batch])
        store.add_vectors(p.embed_key, batch, vecs)
        yield min(i + cfg.EMBED_BATCH, total), total


async def embed_and_index(hub: Hub, store: Store):
    """Update the index with the first working embedding provider; fall back on failure."""
    last = None
    for p in hub.embed_chain():
        try:
            async for done, total in sync_index(hub, store, p):
                yield {"provider": p.name, "done": done, "total": total}
            yield {"provider": p.name, "done": 1, "total": 1}
            return
        except Exception as e:  # noqa: BLE001
            log.warning("Embedding %s failed: %s", p.name, e)
            p.mark_failed()
            last = e
    raise RuntimeError("No embedding provider is working. Check your OpenAI key or Ollama.") from last


async def retrieve(hub: Hub, store: Store, query: str, k: int):
    """Return (hits, provider_name). The query is embedded by the same provider whose index is searched."""
    if not store.chunks:
        return [], None
    last = None
    for p in hub.embed_chain():
        try:
            async for _ in sync_index(hub, store, p):  # on fallback, sync any missing chunks first
                pass
            qv = (await hub.embed(p, [query]))[0]
            return store.search(p.embed_key, qv, k), p.name
        except Exception as e:  # noqa: BLE001
            log.warning("Retrieval via %s failed: %s", p.name, e)
            p.mark_failed()
            last = e
    raise RuntimeError("No embedding provider is working.") from last


# ---------- prompts ----------
async def rewrite_question(hub: Hub, history: list[dict], question: str) -> str:
    """Turn a follow-up question into a standalone one using the conversation so far."""
    if not history:
        return question
    convo = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-6:])
    msgs = [
        {"role": "system", "content": (
            "Rewrite the user's last question as a single standalone question, using the conversation "
            "to resolve pronouns and references. Keep the same language as the question. "
            "Output only the rewritten question, nothing else.")},
        {"role": "user", "content": f"Conversation:\n{convo}\n\nLast question: {question}"},
    ]
    try:
        out = (await hub.complete(msgs, 150)).strip().strip('"').strip()
        return out or question
    except Exception:  # noqa: BLE001
        return question


def build_messages(question: str, standalone: str, hits, history: list[dict],
                   lang: str, mode: str) -> list[dict]:
    m = MODES[mode]
    ctx = "\n\n".join(f"Document: {c['doc_name']} | Page: {c['page']}\n{c['text']}" for c, _ in hits)
    system = f"""You are a professional knowledge assistant. Answer using ONLY the provided CONTEXT taken from the user's documents.
If the context does not contain the answer, say so clearly and politely. Never invent facts.

Answer language: {LANGS[lang]}. Write the entire answer in this language.
Answer type: {m['label']}. Length: {m['rule']}

Formatting rules (strict):
- Plain text only. Do not use markdown: no asterisks, underscores, hash signs, backticks, tables, horizontal rules, or emojis.
- Use short paragraphs separated by one blank line.
- For a section title, write a short line that ends with a colon.
- For lists, start each line with the bullet character "{BULLET} ".
- Never write the characters backslash-n. Do not write citation markers like [1].
- Do not mention the words "context" or "document excerpts". Start the answer directly."""
    msgs = [{"role": "system", "content": system}]
    for h in history[-cfg.MAX_HISTORY_TURNS:]:
        msgs.append({"role": h["role"], "content": h["content"][:1500]})
    note = f"\n(Standalone meaning: {standalone})" if standalone != question else ""
    msgs.append({"role": "user", "content": f"CONTEXT:\n{ctx}\n\nQUESTION: {question}{note}"})
    return msgs
