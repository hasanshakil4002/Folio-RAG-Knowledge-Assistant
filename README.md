# Folio: RAG Knowledge Assistant

Answers questions from your PDFs. Stack: HTML + CSS (with vanilla JavaScript), Python 3.12 + FastAPI, FAISS.

## Run

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
python run.py
```

Open http://localhost:8000

## Local testing with Ollama

```bash
ollama pull llama3.1:8b
ollama pull bge-m3
```

Leave OPENAI_API_KEY and GROQ_API_KEY empty in `.env` and Ollama is used.

## Delivering to the buyer

Put the keys in `.env`. No code changes are needed.

- OPENAI_API_KEY present: OpenAI is used.
- Missing or failing: Groq (LLM only), then Ollama.
- Embeddings: OpenAI, otherwise Ollama (Groq has no embedding models).

Each embedding provider and model gets its own FAISS index (`data/index_*.faiss`) because vector sizes differ. When the provider changes, existing PDF chunks are embedded with the new provider automatically on the first question, so that first request can take a little longer.

## OCR (Tesseract)

Tesseract is a system program, not a Python package. It is only needed for the "OCR needed" option.

- Ubuntu/Debian: `sudo apt install tesseract-ocr tesseract-ocr-ben`
- macOS: `brew install tesseract tesseract-lang`
- Windows: install the UB-Mannheim build, include the Bengali data, then set `TESSERACT_CMD` in `.env`.

## Layout

```
backend/   main.py, providers.py, rag.py, ingest.py, vectorstore.py, memory.py, cleaner.py, config.py
frontend/  index.html, style.css, app.js
data/      created automatically: chunks, FAISS indexes, sessions, uploads
```

## Notes

- Answer length is steered by the prompt and max_tokens, so word counts are approximate.
- Some newer OpenAI models (for example the gpt-5 series) reject `max_tokens`. Use gpt-4o-mini or an equivalent model.


D:\venvs\lang_venv\Scripts\Activate.ps1
