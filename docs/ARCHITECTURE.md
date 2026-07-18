# Architecture

## Overview

The platform ingests documents (PDF or Python source) and answers natural-language
questions about them using retrieval-augmented generation. It exposes three core
REST endpoints (`POST /documents`, `POST /query`, `DELETE /documents/{id}`).

## Ingestion flow

```mermaid
flowchart TD
    A[POST /documents] --> B[Save file to disk, insert Document row status=processing]
    B --> C[Return document_id immediately]
    B --> D[Background task starts]
    D --> E{Source type?}
    E -->|PDF| F1[Extract: pymupdf text per page + pdfplumber tables + image refs]
    F1 --> O1{Page text < 20 chars?}
    O1 -->|Yes| O2[OCR fallback: render page to image, Tesseract extracts text]
    O1 -->|No| F1b[Use native extracted text]
    O2 --> V1
    F1b --> V1
    E -->|Python| F2[Extract: ast-parse into function/class units]
    V1[Validate: page count, low-text pages, table count]
    F2 --> V2[Validate: unit count vs regex def/class count]
    V1 --> G1[Chunk: recursive token split for text, one chunk per table]
    V2 --> G2[Chunk: one chunk per function/class, split by method if oversized]
    G1 --> W1[Validate: no empty chunks, no oversized chunks, table structure intact]
    G2 --> W2[Validate: no empty chunks, chunks independently ast-parseable]
    W1 --> H[Embed chunks via Gemini gemini-embedding-001]
    W2 --> H
    H --> X[Validate: consistent vector dimensionality, no zero-vectors]
    X --> I[Upsert vectors + metadata into Chroma]
    I --> Y[Validate: Chroma stored count == computed chunk count]
    Y --> J[Write Chunk rows to SQLite, mark Document status=ready]
    Y -->|any validation fails| K[Mark Document status=failed, store error_detail]
```

Every arrow into a "Validate" box is a real assertion in code (see `app/ingestion.py`
`ValidationReport` objects), not just a comment. A failed validation short-circuits
the pipeline and marks the document `failed` with the reason recorded, rather than
silently proceeding on bad data.

**Why OCR was added:** the provided `Knowledge_Base_Sample.pdf` turned out to be
image-based rather than text-native — native PyMuPDF extraction returned near-empty
text on most pages. Rather than proceed with near-empty chunks (which would have
silently produced a broken knowledge base), extraction falls back to Tesseract OCR
on a per-page basis whenever native extraction yields under 20 characters. This is
a direct consequence of the extraction-validation discipline described above: the
low-text-page check caught the problem, and OCR was the fix rather than ignoring
the warning.

## Query flow

```mermaid
flowchart TD
    A[POST /query] --> B[Embed query text via Gemini]
    B --> C[Chroma similarity search: top_k, filter status=active, optional source_type/doc_id filter]
    C --> D{Any hits?}
    D -->|No| E[Return: no relevant content found]
    D -->|Yes| F[Build context block with filename + page/function labels]
    F --> G[Groq llama-3.3-70b-versatile generates answer citing sources]
    G --> H[Log query to QueryLog table]
    H --> I[Return answer + sources + latency_ms]
```

## Delete flow

```mermaid
flowchart TD
    A[DELETE /documents/id] --> B[Mark Chroma vectors status=deleted for doc_id]
    B --> C[Set Document.status=deleted in SQLite]
    C --> D{hard=true?}
    D -->|No| E[Return: soft-deleted]
    D -->|Yes| F[Purge vectors from Chroma + delete Document/Chunk rows]
    F --> G[Return: hard-deleted]
```

## Component responsibilities

| Component | File | Responsibility |
|---|---|---|
| API layer | `app/routes.py` | Thin orchestration: receives requests, calls the other layers, returns responses |
| Ingestion | `app/ingestion.py` | Extraction, chunking, and validation logic for both PDF and code |
| External model calls | `app/ai_clients.py` | Gemini embeddings, Groq generation, shared retry/backoff |
| Vector store | `app/vectorstore.py` | Chroma collection management, similarity search, soft/hard delete |
| Relational data | `app/database.py` | SQLAlchemy async engine + Document/Chunk/QueryLog models |
| Config | `app/config.py` | All environment-driven settings in one place |

## Why background tasks instead of Celery/Redis

FastAPI's `BackgroundTasks` (implemented here via `asyncio.create_task`) is sufficient
for ~100 internal users uploading documents occasionally. It avoids the operational
overhead of a broker + worker pool for a system at this scale. See
`docs/SCALING_TRADEOFFS.md` for when this stops being true and Celery+Redis becomes
worth the complexity.

## System dependency: Tesseract OCR

The OCR fallback requires Tesseract installed as a system binary (not a pip package)
and available on PATH, or its path set explicitly via
`pytesseract.pytesseract.tesseract_cmd`. See https://github.com/UB-Mannheim/tesseract/wiki
for Windows installers. Without Tesseract installed, PDFs with image-only pages will
fail extraction validation (`low_text_pages` will be non-empty) rather than silently
producing empty chunks — the validation step surfaces this rather than hiding it.