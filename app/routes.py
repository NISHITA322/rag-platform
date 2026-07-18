"""
All API endpoints. Each route is a thin orchestration layer that calls
ingestion.py / ai_clients.py / vectorstore.py - the actual logic lives in
those modules, not here.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import ingestion, vectorstore
from app.ai_clients import GeminiEmbedder, GroqClient
from app.config import settings
from app.database import get_session, Document, Chunk, QueryLog, DocumentStatus, SourceType, AsyncSessionLocal
from app.schemas import (
    DocumentUploadResponse, DocumentStatusResponse, DocumentListResponse, DocumentListItem,
    QueryRequest, QueryResponse, SourceChunk, DeleteResponse,
)

logger = logging.getLogger("routes")
router = APIRouter()

_embedder = GeminiEmbedder()
_groq = GroqClient()


def _source_type_for(filename: str) -> SourceType:
    return SourceType.pdf if filename.lower().endswith(".pdf") else SourceType.code


async def _process_document(document_id: str, file_path: str, source_type: SourceType) -> None:
    """Background task: extract -> validate -> chunk -> validate -> embed
    -> validate -> store -> validate -> mark ready/failed. Runs in its own
    session since BackgroundTasks execute after the request session closes."""
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, document_id)
        try:
            if source_type == SourceType.pdf:
                chunks, reports = ingestion.process_pdf(
                    file_path, settings.pdf_chunk_token_size, settings.pdf_chunk_overlap_tokens
                )
            else:
                print("in else")
                chunks, reports = ingestion.process_code(file_path, settings.pdf_chunk_token_size)

            if not chunks:
                raise ValueError("No chunks produced during extraction/chunking - check validation report")

            texts = [c["text"] for c in chunks]
            embeddings = await _embedder.embed_batch(texts)

            dims = {len(v) for v in embeddings}
            if len(dims) > 1:
                raise ValueError(f"Inconsistent embedding dimensionality across chunks: {dims}")
            if any(all(x == 0 for x in v) for v in embeddings):
                raise ValueError("Found a zero-vector embedding - embedding call likely failed silently")

            vector_ids = [f"{document_id}:{i}:{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
            vectorstore.upsert_chunks(
                doc_id=document_id,
                filename=doc.filename,
                source_type=source_type.value,
                chunks=chunks,
                embeddings=embeddings,
                vector_ids=vector_ids,
            )

            stored_count = vectorstore.count_for_doc(document_id)
            if stored_count != len(chunks):
                raise ValueError(f"Chroma stored count ({stored_count}) != computed chunk count ({len(chunks)})")

            for chunk_data, vector_id in zip(chunks, vector_ids):
                session.add(Chunk(
                    document_id=document_id,
                    chunk_type=chunk_data["chunk_type"],
                    page_num=chunk_data["page_num"],
                    function_name=chunk_data["function_name"],
                    start_line=chunk_data["start_line"],
                    end_line=chunk_data["end_line"],
                    token_count=chunk_data["token_count"],
                    chroma_vector_id=vector_id,
                ))

            doc.status = DocumentStatus.ready
            doc.chunk_count = len(chunks)
            doc.validation_report = json.dumps([r.as_dict() for r in reports])
            await session.commit()
            logger.info(json.dumps({"event": "ingestion_complete", "document_id": document_id, "chunks": len(chunks)}))

        except Exception as exc:  # noqa: BLE001
            doc.status = DocumentStatus.failed
            doc.error_detail = str(exc)
            await session.commit()
            logger.error(json.dumps({"event": "ingestion_failed", "document_id": document_id, "error": str(exc)}))


@router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...), session: AsyncSession = Depends(get_session)):
    if not (file.filename.lower().endswith(".pdf") or file.filename.lower().endswith(".py")):
        raise HTTPException(status_code=400, detail="Only .pdf and .py files are supported")

    os.makedirs(settings.upload_dir, exist_ok=True)
    document_id = str(uuid.uuid4())
    source_type = _source_type_for(file.filename)
    print("source type ",source_type)
    saved_path = os.path.join(settings.upload_dir, f"{document_id}_{file.filename}")

    with open(saved_path, "wb") as f:
        f.write(await file.read())

    doc = Document(
        id=document_id,
        filename=file.filename,
        source_type=source_type,
        status=DocumentStatus.processing,
        file_path=saved_path,
    )
    session.add(doc)
    await session.commit()

    import asyncio
    asyncio.create_task(_process_document(document_id, saved_path, source_type))

    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        status=DocumentStatus.processing.value,
        job_status_url=f"/documents/{document_id}/status",
    )



@router.post("/query", response_model=QueryResponse)
async def query_documents(payload: QueryRequest, session: AsyncSession = Depends(get_session)):
    start = time.perf_counter()
    top_k = payload.top_k or settings.default_top_k
    source_type = payload.filters.source_type if payload.filters else None
    doc_id = payload.filters.doc_id if payload.filters else None

    query_embedding = await _embedder.embed_query(payload.query)
    print("source type ",source_type)
    hits = vectorstore.query(query_embedding, top_k=top_k, source_type=source_type, doc_id=doc_id)

    print("data : ",hits)
   
    if not hits:
        answer = "No relevant content was found in the knowledge base for this query."
    else:
        answer = await _groq.generate_answer(payload.query, hits)

    latency_ms = (time.perf_counter() - start) * 1000

    session.add(QueryLog(
        query_text=payload.query,
        top_k=top_k,
        returned_doc_ids=json.dumps([h["filename"] for h in hits]),
        latency_ms=latency_ms,
    ))
    await session.commit()

    return QueryResponse(
        answer=answer,
        sources=[SourceChunk(**h) for h in hits],
        latency_ms=latency_ms,
    )


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str, hard: bool = Query(False), session: AsyncSession = Depends(get_session)):
    doc = await session.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    soft_ok, hard_ok = False, False
    detail_parts = []

    try:
        vectorstore.mark_deleted(document_id)
        soft_ok = True
    except Exception as exc:  
        detail_parts.append(f"Chroma soft-delete failed: {exc}")

    try:
        doc.status = DocumentStatus.deleted
        await session.commit()
    except Exception as exc:
        detail_parts.append(f"SQLite status update failed: {exc}")
        # Partial-failure handling: if SQLite update fails after Chroma
        # succeeded, we now have an inconsistency - surface it rather than
        # silently reporting success.
        raise HTTPException(status_code=500, detail="; ".join(detail_parts))

    if hard:
        try:
            removed = vectorstore.purge(document_id)
            await session.delete(doc)
            await session.commit()
            hard_ok = True
            detail_parts.append(f"Hard-deleted {removed} vectors and document row")
        except Exception as exc: 
            detail_parts.append(f"Hard delete failed: {exc}")
            raise HTTPException(status_code=500, detail="; ".join(detail_parts))

    return DeleteResponse(
        document_id=document_id,
        soft_deleted=soft_ok,
        hard_deleted=hard_ok,
        detail="; ".join(detail_parts) if detail_parts else "Soft delete successful",
    )
