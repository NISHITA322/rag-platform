# Internal AI Knowledge Platform

A RAG backend that ingests PDF and Python source files, chunks and embeds them, and
answers natural-language questions with cited sources.

**Stack:** FastAPI · ChromaDB · Gemini (`gemini-embedding-001`) · Groq
(`llama-3.3-70b-versatile`) · SQLite

## Setup

```bash
# 1. Create and activate a virtual environment (do not install globally)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies inside the venv
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# edit .env and set GEMINI_API_KEY and GROQ_API_KEY
```

## Running the API

```bash
source venv/bin/activate
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive Swagger UI, where you can
exercise all endpoints manually:
- `POST /documents` — upload a `.pdf` or `.py` file
- `GET /documents/{id}/status` — check ingestion progress and validation results
- `GET /documents` — list all uploaded documents
- `POST /query` — ask a question
- `DELETE /documents/{id}` — soft delete (add `?hard=true` for hard delete)

## Proof of execution

Run the two assignment task files through the full pipeline and see validation +
sample query output printed to the console:

```bash
source venv/bin/activate
python scripts/ingest_sample_files.py \
    --pdf path/to/Knowledge_Base_Sample.pdf \
    --code path/to/Source_Code_Sample.py
```

This script prints, in order:
1. Every validation report from extraction and chunking (page counts, empty-chunk
   checks, table/code structural integrity checks)
2. Embedding dimensionality and zero-vector checks
3. Confirmation that Chroma's stored chunk count matches the computed chunk count
4. Sample query results with generated answers and cited sources

**This console output is what to screenshot or record as your submission's proof
of execution** — it's evidence the pipeline actually worked correctly, not just
that it ran without crashing.

## Running tests

```bash
source venv/bin/activate
pytest tests/test_pipeline.py -v
```

These tests run offline (no Gemini/Groq calls) and cover extraction, chunking, and
validation logic for both the PDF and code pipelines.

## Inspecting stored data

- SQLite: `rag_platform.db` (created on first run) — inspect with `sqlite3
  rag_platform.db` or any SQLite browser. See `documents`, `chunks`, and
  `query_logs` tables.
- Chroma: `chroma_data/` directory (created on first run) — the persistent vector
  store.

## Design decisions and tradeoffs

See `docs/ARCHITECTURE.md`, `docs/DB_SCHEMA.md`, and `docs/SCALING_TRADEOFFS.md` for
the system diagram, schema rationale, and scaling/tradeoff discussion required by
the assignment deliverables.

Key decisions worth calling out up front:
- **Extraction is source-type-specific**: PDF uses pymupdf (text) + pdfplumber
  (tables, kept structurally separate from body text) + image references; Python
  files are parsed with `ast` so a function is never split mid-body.
- **Every pipeline stage is validated**, not assumed correct — see the
  `ValidationReport` objects produced at each step in `app/ingestion.py`, and the
  `validation_report` field stored per document.
- **Soft delete is the default**; hard delete is opt-in via `?hard=true` and
  explicitly surfaces partial failures instead of reporting false success.
- **FastAPI `BackgroundTasks`** (not Celery/Redis) for async ingestion — appropriate
  at this scale; see `docs/SCALING_TRADEOFFS.md` for when that stops being true.

## Known limitations

- PDF image content is referenced by page/index, not captioned or embedded — could
  be extended with Gemini vision captioning as a `chunk_type: image_caption` chunk.
- No semantic cache yet for repeated queries (see `docs/SCALING_TRADEOFFS.md`).
- `tiktoken`'s vocab file requires a one-time network fetch; if that fetch is
  blocked by network policy, `app/ingestion.py` falls back to an approximate
  4-chars-per-token estimate so chunking still works, just less precisely.
