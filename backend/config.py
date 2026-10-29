"""Configuration: every setting is read from the .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _e(key: str, default: str = "") -> str:
    return (os.getenv(key) or default).strip()


# ---------- paths ----------
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"
for _d in (DATA_DIR, UPLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
DOCS_FILE = DATA_DIR / "documents.json"
SESSIONS_FILE = DATA_DIR / "sessions.json"

# ---------- providers ----------
OPENAI_API_KEY = _e("OPENAI_API_KEY")
OPENAI_BASE_URL = _e("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
OPENAI_LLM_MODEL = _e("OPENAI_LLM_MODEL", "openai/gpt-4o-mini")
OPENAI_EMBED_MODEL = _e("OPENAI_EMBED_MODEL", "openai/text-embedding-3-small")

# OPENAI_API_KEY="sk-or-v1-আপনার-নতুন-key"
# OPENAI_BASE_URL="https://openrouter.ai/api/v1"
# OPENAI_LLM_MODEL="openai/gpt-4o-mini"
# OPENAI_EMBED_MODEL="openai/text-embedding-3-small"

GROQ_API_KEY = _e("GROQ_API_KEY")
GROQ_BASE_URL = _e("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_LLM_MODEL = _e("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")

OLLAMA_BASE_URL = _e("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = _e("OLLAMA_LLM_MODEL", "llama3.1:8b")
OLLAMA_EMBED_MODEL = _e("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def _csv(key: str, default: str) -> list[str]:
    return [x.strip().lower() for x in _e(key, default).split(",") if x.strip()]


# Priority order: if the first one fails, the next one is used.
LLM_PRIORITY = _csv("LLM_PRIORITY", "openai,groq,ollama")#
EMBED_PRIORITY = _csv("EMBED_PRIORITY", "openai,ollama")  # Groq has no embedding models ,ollama

# ---------- RAG ----------
CHUNK_SIZE = int(_e("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(_e("CHUNK_OVERLAP", "150"))
TOP_K = int(_e("TOP_K", "6"))
EMBED_BATCH = int(_e("EMBED_BATCH", "32"))
MAX_HISTORY_TURNS = int(_e("MAX_HISTORY_TURNS", "8"))

# ---------- OCR ----------
OCR_LANG = _e("OCR_LANG", "ben+eng")
OCR_DPI = int(_e("OCR_DPI", "220"))
TESSERACT_CMD = _e("TESSERACT_CMD")  # may be required on Windows

MAX_UPLOAD_MB = int(_e("MAX_UPLOAD_MB", "50"))
