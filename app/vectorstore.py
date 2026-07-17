"""
Chroma vector store wrapper: one persistent collection, all upsert/query/
delete logic for it lives here.
"""
from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings

_COLLECTION_NAME = "knowledge_base"

_client = chromadb.PersistentClient(
    path=settings.chroma_persist_dir,
    settings=ChromaSettings(anonymized_telemetry=False),
)
_collection = _client.get_or_create_collection(name=_COLLECTION_NAME)


def upsert_chunks(
    doc_id: str,
    filename: str,
    source_type: str,
    chunks: list[dict],
    embeddings: list[list[float]],
    vector_ids: list[str],
) -> None:
    """Store chunk vectors + metadata + raw text. Raw text is stored in
    Chroma's `documents` field so a chunk's source text is always
    retrievable alongside its vector - never store a vector without it."""
    metadatas = []
    for c in chunks:
        metadatas.append({
            "doc_id": doc_id,
            "filename": filename,
            "source_type": source_type,
            "chunk_type": c["chunk_type"],
            "page_num": c["page_num"] if c["page_num"] is not None else -1,
            "function_name": c["function_name"] or "",
            "status": "active",
        })

    _collection.upsert(
        ids=vector_ids,
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=metadatas,
    )


def count_for_doc(doc_id: str) -> int:
    result = _collection.get(where={"doc_id": doc_id})
    return len(result["ids"])


def query(
    query_embedding: list[float],
    top_k: int,
    source_type: str,
    doc_id: str | None = None,
) -> list[dict]:
    """Similarity search excluding soft-deleted documents. Returns a list
    of dicts with text, metadata, and similarity score."""
    where_clauses = [{"status": {"$eq": "active"}}]
    if source_type:
        where_clauses.append({"source_type": {"$eq": source_type}})
    if doc_id:
        where_clauses.append({"doc_id": {"$eq": doc_id}})

    print("soucre type ",source_type)

    where = {"$and": where_clauses} if len(where_clauses) > 1 else where_clauses[0]

    result = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )

    print("collection ",_collection.count())
    docs = _collection.get()

    for meta in docs["metadatas"]:
        print("=============================")
        print(meta["filename"], meta["source_type"])

    # print("results ",result)

    if not result["ids"] or not result["ids"][0]:
        return []

    hits = []
    for i in range(len(result["ids"][0])):
        meta = result["metadatas"][0][i]
        distance = result["distances"][0][i]
        hits.append({
            "chunk_text": result["documents"][0][i],
            "filename": meta["filename"],
            "chunk_type": meta["chunk_type"],
            "page_num": meta["page_num"] if meta["page_num"] != -1 else None,
            "function_name": meta["function_name"] or None,
            "score": 1.0 - distance,  # cosine distance -> similarity
        })
    return hits


def mark_deleted(doc_id: str) -> None:
    """Soft delete: flip status metadata so future queries exclude these
    vectors, without physically removing them."""
    existing = _collection.get(where={"doc_id": doc_id})
    if not existing["ids"]:
        return
    updated_metadatas = [{**m, "status": "deleted"} for m in existing["metadatas"]]
    _collection.update(ids=existing["ids"], metadatas=updated_metadatas)


def purge(doc_id: str) -> int:
    """Hard delete: physically remove all vectors for a document. Returns
    the number of vectors removed."""
    existing = _collection.get(where={"doc_id": doc_id})
    if not existing["ids"]:
        return 0
    _collection.delete(ids=existing["ids"])
    return len(existing["ids"])
