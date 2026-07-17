"""
Database layer: async SQLAlchemy engine/session plus the three ORM models
(Document, Chunk, QueryLog).

Kept in one file deliberately - the engine setup and the models it serves
are a single conceptual unit for a project this size, and a reviewer should
be able to see the entire data model in one screen.

Schema summary (mirrored in docs/DB_SCHEMA.md with full rationale):
- documents:  one row per uploaded file. status tracks the async pipeline
  (processing -> ready | failed | deleted).
- chunks:     one row per chunk stored in Chroma. chroma_vector_id links
  back to the vector store so deletes can target the right vectors.
- query_logs: one row per /query call, for observability and future
  semantic-cache work.
"""
import datetime as dt
import enum
import uuid

from sqlalchemy import String, Integer, Text, DateTime, ForeignKey, Enum, JSON, Float
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import settings


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


class DocumentStatus(str, enum.Enum):
    processing = "processing"
    ready = "ready"
    failed = "failed"
    deleted = "deleted"


class SourceType(str, enum.Enum):
    pdf = "pdf"
    code = "code"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(512))
    source_type: Mapped[SourceType] = mapped_column(Enum(SourceType))
    status: Mapped[DocumentStatus] = mapped_column(Enum(DocumentStatus), default=DocumentStatus.processing, index=True)
    file_path: Mapped[str] = mapped_column(String(1024))
    upload_timestamp: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    validation_report: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON string

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id"), index=True)
    chunk_type: Mapped[str] = mapped_column(String(32))  # text | table | function | class
    page_num: Mapped[int | None] = mapped_column(Integer, nullable=True)
    function_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    start_line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    chroma_vector_id: Mapped[str] = mapped_column(String(64), unique=True)

    document: Mapped["Document"] = relationship(back_populates="chunks")


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    query_text: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)
    top_k: Mapped[int] = mapped_column(Integer)
    returned_doc_ids: Mapped[str] = mapped_column(JSON)  # list[str] stored as JSON
    latency_ms: Mapped[float] = mapped_column(Float)


engine = create_async_engine(settings.sqlalchemy_url, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
