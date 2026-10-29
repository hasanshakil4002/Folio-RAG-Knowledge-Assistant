"""OpenAI -> Groq -> Ollama fallback.

All three services expose an OpenAI-compatible API, so a single AsyncOpenAI client class
drives every provider. A provider that fails goes into a short cooldown and the next
provider in the chain takes over.
"""
import logging
import time
from typing import AsyncIterator

import httpx
import numpy as np
from openai import AsyncOpenAI

from . import config as cfg

log = logging.getLogger("rag.providers")


class Provider:
    def __init__(self, name, base_url, api_key, llm_model, embed_model):
        self.name = name
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.down_until = 0.0
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "ollama",
            timeout=httpx.Timeout(120.0, connect=8.0),
            max_retries=0,
        )

    @property
    def available(self) -> bool:
        return time.time() >= self.down_until

    def mark_failed(self, seconds: int = 60) -> None:
        self.down_until = time.time() + seconds

    @property
    def embed_key(self) -> str:
        return f"{self.name}:{self.embed_model}"


class Hub:
    def __init__(self):
        self.providers: dict[str, Provider] = {}
        if cfg.OPENAI_API_KEY:
            self.providers["openai"] = Provider(
                "openai", cfg.OPENAI_BASE_URL, cfg.OPENAI_API_KEY,
                cfg.OPENAI_LLM_MODEL, cfg.OPENAI_EMBED_MODEL)
        if cfg.GROQ_API_KEY:
            self.providers["groq"] = Provider(
                "groq", cfg.GROQ_BASE_URL, cfg.GROQ_API_KEY,
                cfg.GROQ_LLM_MODEL, None)  # Groq has no embedding model
        self.providers["ollama"] = Provider(
            "ollama", cfg.OLLAMA_BASE_URL.rstrip("/") + "/v1", "ollama",
            cfg.OLLAMA_LLM_MODEL, cfg.OLLAMA_EMBED_MODEL)
        self.ollama_models: list[str] = []

    # ---------- health ----------
    async def refresh_ollama(self) -> bool:
        p = self.providers["ollama"]
        try:
            async with httpx.AsyncClient(timeout=2.0) as c:
                r = await c.get(cfg.OLLAMA_BASE_URL.rstrip("/") + "/api/tags")
                r.raise_for_status()
            self.ollama_models = [m.get("name", "") for m in r.json().get("models", [])]
            return True
        except Exception:
            p.mark_failed(20)
            return False

    def _chain(self, order: list[str], attr: str) -> list[Provider]:
        ps = [self.providers[n] for n in order
              if n in self.providers and getattr(self.providers[n], attr)]
        ready = [p for p in ps if p.available]
        return ready + [p for p in ps if not p.available]

    def llm_chain(self) -> list[Provider]:
        return self._chain(cfg.LLM_PRIORITY, "llm_model")

    def embed_chain(self) -> list[Provider]:
        return self._chain(cfg.EMBED_PRIORITY, "embed_model")

    def status(self) -> dict:
        llm, emb = self.llm_chain(), self.embed_chain()
        return {
            "llm": llm[0].name if llm else None,
            "llm_model": llm[0].llm_model if llm else None,
            "embedding": emb[0].name if emb else None,
            "embed_model": emb[0].embed_model if emb else None,
            "ready": bool(llm and emb and (llm[0].available or emb[0].available)),
            "providers": [
                {"name": p.name, "available": p.available,
                 "llm": p.llm_model, "embed": p.embed_model}
                for p in self.providers.values()
            ],
        }

    # ---------- calls ----------
    async def embed(self, p: Provider, texts: list[str]) -> np.ndarray:
        r = await p.client.embeddings.create(model=p.embed_model, input=texts)
        arr = np.array([d.embedding for d in r.data], dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    async def stream(self, p: Provider, messages: list[dict], max_tokens: int,
                     temperature: float = 0.2) -> AsyncIterator[str]:
        resp = await p.client.chat.completions.create(
            model=p.llm_model, messages=messages, max_tokens=max_tokens,
            temperature=temperature, stream=True)
        async for ev in resp:
            if ev.choices and ev.choices[0].delta and ev.choices[0].delta.content:
                yield ev.choices[0].delta.content

    async def complete(self, messages: list[dict], max_tokens: int = 200) -> str:
        last = None
        for p in self.llm_chain():
            try:
                r = await p.client.chat.completions.create(
                    model=p.llm_model, messages=messages,
                    max_tokens=max_tokens, temperature=0)
                return r.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                log.warning("LLM %s failed: %s", p.name, e)
                p.mark_failed()
                last = e
        raise RuntimeError("No LLM provider is available") from last
