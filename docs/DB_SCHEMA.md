# Database Schema

Relational data lives in SQLite (via async SQLAlchemy); vectors + their raw text
live in Chroma. This split exists because SQLite is good at structured queries and
status tracking, while Chroma is purpose-built for similarity search — using SQLite
for vector search or Chroma for relational joins would fight both tools' design.

## Tables

### `documents`
One row per uploaded file.

| Column | Type | Notes |
|---|---|---|
| id | String(36), PK | UUID |
| filename | String(512) | Original filename |
| source_type | Enum(pdf, code) | Drives which extraction pipeline runs |
| status | Enum(processing, ready, failed, deleted), indexed | Tracks the async ingestion pipeline |
| file_path | String(1024) | Where the raw upload is stored on disk |
| upload_timestamp | DateTime | |
| chunk_count | Integer | Set once ingestion completes |
| error_detail | Text, nullable | Populated if status=failed |
| validation_report | Text (JSON), nullable | Serialized `ValidationReport` list from ingestion — this is what you inspect to confirm extraction/chunking actually worked, not just that the pipeline "ran" |

### `chunks`
One row per chunk, mirroring what's stored in Chroma.

| Column | Type | Notes |
|---|---|---|
| id | String(36), PK | UUID |
| document_id | String(36), FK -> documents.id, indexed | |
| chunk_type | String(32) | text / table / function / class |
| page_num | Integer, nullable | Set for PDF chunks |
| function_name | String(256), nullable | Set for code chunks |
| start_line / end_line | Integer, nullable | Set for code chunks |
| token_count | Integer | |
| chroma_vector_id | String(64), unique | Links this row to its vector in Chroma — this is how deletes target the right vectors |

### `query_logs`
One row per `/query` call.

| Column | Type | Notes |
|---|---|---|
| id | String(36), PK | UUID |
| query_text | Text | |
| timestamp | DateTime, indexed | |
| top_k | Integer | |
| returned_doc_ids | JSON | Which documents' chunks were returned |
| latency_ms | Float | |

## Indexing strategy

- `documents.status` — the status-check query (`is this document ready yet?`) runs
  far more often than full-table scans, so this needs an index even at small scale.
- `chunks.document_id` — every delete and every "how many chunks does this doc have"
  check filters by this column.
- `query_logs.timestamp` — needed for any future "queries in the last N days"
  analytics or cache-warming logic.

## Partitioning

Not implemented — unnecessary at ~100 internal users' document volume, and SQLite
doesn't support it natively anyway. If this moved to Postgres at scale, the natural
partition key would be `documents.upload_timestamp` (monthly partitions), since
`query_logs` and `chunks` both grow roughly proportionally with ingestion volume.

## Metadata modeling

Chroma metadata mirrors the SQL columns needed for filtering at query time
(`doc_id`, `filename`, `source_type`, `chunk_type`, `page_num`, `function_name`,
`status`). This intentional duplication exists because Chroma's `where` filter
can't join against SQLite — anything you need to filter search results by must be
present directly on the Chroma metadata, not just in the relational tables.

## Query patterns

1. "Is my upload done processing?" → `GET /documents/{id}/status` → single-row
   lookup by primary key.
2. "What can I ask about?" → `GET /documents` → full scan, acceptable at this scale.
3. "Answer this question" → embed → Chroma similarity search (not a SQL query at
   all) → SQLite write only for the log.
4. "Delete this document" → SQLite status update + Chroma metadata update (soft) or
   SQLite row delete + Chroma vector delete (hard).

## Caching

Not implemented in this version, but the natural next step is a semantic cache:
store `(query_embedding, answer)` pairs and, before calling Groq, check whether an
incoming query's embedding is within a similarity threshold of a cached query. This
would cut both latency and free-tier API usage for repeated/similar questions,
which matters given Gemini and Groq's free-tier rate limits. See
`docs/SCALING_TRADEOFFS.md`.
