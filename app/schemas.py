"""
All request/response Pydantic models for the API, in one file since each
is small and a reviewer benefits from seeing the full request/response
contract of the system at a glance.
"""
from pydantic import BaseModel, Field


# ---------- POST /documents ----------

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    job_status_url: str


class DocumentStatusResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunk_count: int
    error_detail: str | None = None
    validation_report: dict | None = None


class DocumentListItem(BaseModel):
    document_id: str
    filename: str
    source_type: str
    status: str
    chunk_count: int
    upload_timestamp: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem]


# ---------- POST /query ----------

class QueryFilters(BaseModel):
    source_type: str | None = None
    doc_id: str | None = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = None
    filters: QueryFilters | None = None


class SourceChunk(BaseModel):
    filename: str
    chunk_type: str
    page_num: int | None = None
    function_name: str | None = None
    chunk_text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    latency_ms: float


# ---------- DELETE /documents/{id} ----------

class DeleteResponse(BaseModel):
    document_id: str
    soft_deleted: bool
    hard_deleted: bool
    detail: str
