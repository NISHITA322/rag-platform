# Scaling Strategy, Assumptions, and Tradeoffs

## Stated assumptions

- ~100 internal developers use this platform, uploading documents occasionally
  rather than continuously — this is not a public-internet-scale ingestion system.
- Free-tier Gemini and Groq API quotas are in play, so rate-limit handling matters
  more than raw throughput.
- A single-node Chroma instance and a single SQLite file are acceptable for the
  corpus size implied by ~100 internal users, not for millions of documents.

## BackgroundTasks now, Celery+Redis later

**Now:** `asyncio.create_task` inside the FastAPI process handles ingestion
asynchronously without a broker. Simpler to operate, no extra infrastructure, fully
sufficient for occasional uploads from ~100 users.

**Tradeoff:** if the API process restarts mid-ingestion, that job is lost — there's
no persisted job queue to resume from. There's also no built-in retry-with-delay
scheduling or multi-worker fan-out.

**When to move:** once ingestion volume is high enough that a single process's
background tasks can't keep up, or job durability across restarts matters (e.g.
regulated environments where a lost ingestion job is unacceptable), move to
Celery + Redis (or an equivalent durable task queue). This is a drop-in replacement
for the `_process_document` background task — the ingestion/embedding/storage logic
itself doesn't change, only how the job is scheduled and retried.

## Chroma now, Qdrant/PgVector later

**Now:** Chroma's persistent local client requires no separate service to run and
is simple to reason about for a bounded internal corpus.

**Tradeoff:** Chroma's single-node model doesn't horizontally scale reads/writes,
and its filtering is less expressive than a dedicated vector database at high
cardinality.

**When to move:** once total chunk count reaches the low millions, or multiple API
instances need to share a vector store (Chroma's persistent client isn't designed
for concurrent multi-process writers at that scale) — Qdrant or PgVector (the
latter especially if the team wants to keep vectors and relational data in the same
Postgres instance) are the natural next step. The `app/vectorstore.py` interface
(`upsert_chunks`, `query`, `mark_deleted`, `purge`) is written so swapping the
backend means rewriting that one file, not touching ingestion or routes.

## Rate limits (Gemini + Groq free tiers)

Both `GeminiEmbedder` and `GroqClient` share a retry-with-exponential-backoff
wrapper (`_with_retry` in `app/ai_clients.py`) that specifically detects
rate-limit and 5xx errors and backs off rather than failing immediately. At
higher volume, this should be paired with client-side request throttling
(e.g. a token-bucket limiter) to stay under quota proactively rather than
reactively retrying after a 429.

