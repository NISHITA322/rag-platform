# Internal AI Knowledge Platform

A RAG backend that ingests PDF and Python source files, chunks and embeds them, and
answers natural-language questions with cited sources.

**Stack:** FastAPI · ChromaDB · Gemini (`gemini-embedding-001`) · Groq
(`llama-3.3-70b-versatile`) · SQLite · PyMuPDF · pdfplumber · Tesseract OCR

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

## Install Tesseract OCR

Scanned or image-based PDFs require the **Tesseract OCR** engine. The ingestion pipeline automatically falls back to OCR when native PDF text extraction returns little or no text.

### Windows

1. Download and install Tesseract OCR from the **UB Mannheim** distribution:
   https://github.com/UB-Mannheim/tesseract/wiki

2. Ensure the installation path is accessible. The default installation path is:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

3. Update the Tesseract executable path in `app/ingestion.py` if your installation directory is different:

```python
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
```

4. Verify the installation:

```bash
tesseract --version
```

If Tesseract is installed correctly, you should see version information printed in the terminal.

Open `http://127.0.0.1:8000/docs` for the interactive Swagger UI, where you can
exercise all endpoints manually:
- `POST /documents` — upload a `.pdf` or `.py` file
- `POST /query` — ask a question
- `DELETE /documents/{id}` — soft delete (add `?hard=true` for hard delete)


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
- **Extraction is source-type-specific**:
  - **PDF:** PyMuPDF performs native text extraction. If a page contains little or no extractable text (e.g., scanned/image-based PDFs), the pipeline automatically falls back to **Tesseract OCR** after rendering the page as a high-resolution image. Tables are extracted separately using **pdfplumber** to preserve their structure and avoid mixing them with body text.
  - **Python:** Source files are parsed with Python's `ast` module so functions and classes remain intact during chunking and are never split mid-body.
- **Soft delete is the default**; hard delete is opt-in via `?hard=true` and
  explicitly surfaces partial failures instead of reporting false success.
- **FastAPI `BackgroundTasks`** (not Celery/Redis) for async ingestion — appropriate
  at this scale; see `docs/SCALING_TRADEOFFS.md` for when that stops being true.
