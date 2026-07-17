"""
FastAPI application entrypoint. Registers routes and runs DB init on
startup. Run with: uvicorn app.main:app --reload
"""
import logging

from fastapi import FastAPI

from app.config import settings
from app.database import init_db
from app.routes import router

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="Internal AI Knowledge Platform",
    description="RAG backend for document/code ingestion and semantic query.",
    version="1.0.0",
)

app.include_router(router)


@app.on_event("startup")
async def on_startup():
    await init_db()


@app.get("/health")
async def health():
    return {"status": "ok"}
