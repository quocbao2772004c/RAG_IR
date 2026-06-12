from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from rag_engine import HybridRAG
from versioned_rag import AVAILABLE_VERSIONS, VersionedRAG

try:
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover
    ConfigDict = None  # type: ignore


load_dotenv()


STATE_PATH = Path(os.getenv("RAG_STATE_PATH", "rag_state/index.pkl"))
FOLDER_VERSION = Path(__file__).resolve().parent.name.lower()
DEFAULT_RAG_VERSION = "v5"
RAG_VERSION = os.getenv("RAG_VERSION", DEFAULT_RAG_VERSION).lower()
if RAG_VERSION not in AVAILABLE_VERSIONS:
    RAG_VERSION = DEFAULT_RAG_VERSION
USE_INDEX_SEMANTIC = os.getenv("RAG_USE_INDEX_SEMANTIC", "0") == "1"


class UploadResponse(BaseModel):
    status: str
    doc_id: str | None = None
    chunks: int


class AskRequest(BaseModel):
    question: str
    options: list[str] | None = None

    if ConfigDict is not None:
        model_config = ConfigDict(extra="allow")
    else:
        class Config:
            extra = "allow"


class AskResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)


app = FastAPI(title="Student Offline RAG Server", version="1.0.0")
engine = HybridRAG(use_semantic=USE_INDEX_SEMANTIC)
answerer = VersionedRAG(
    engine,
    version=RAG_VERSION,
    use_semantic_reranker=os.getenv("RAG_USE_SEMANTIC_RERANKER", "1") != "0",
)


def load_local_qa_cache() -> None:
    if os.getenv("RAG_USE_QA_CACHE", "0") != "1":
        return
    raw_paths = os.getenv(
        "RAG_QA_CACHE_PATHS",
        "data_ir/generated_questions_2.json:data_ir/generated_questions.json",
    )
    paths = [part for part in raw_paths.split(":") if part]
    if paths:
        info = engine.load_qa_cache(paths)
        if info["items"]:
            print(f"[startup] loaded QA cache -> {info}")


def clear_qa_cache_if_disabled() -> None:
    if os.getenv("RAG_USE_QA_CACHE", "0") == "1":
        return
    engine.qa_cache_full.clear()
    engine.qa_cache_stem.clear()


def preload_or_restore_index() -> None:
    if STATE_PATH.exists() and os.getenv("RAG_IGNORE_STATE", "0") != "1":
        try:
            info = engine.load_state(STATE_PATH)
            clear_qa_cache_if_disabled()
            print(f"[startup] restored vector index -> {info}")
            return
        except Exception as exc:
            print(f"[startup] restore failed: {exc}")

    preload_path = os.getenv("RAG_PRELOAD_PATH", "data_ir/all_contexts.md")
    if not preload_path:
        return
    path = Path(preload_path)
    if path.exists():
        try:
            info = engine.load_path(path)
            clear_qa_cache_if_disabled()
            engine.save_state(STATE_PATH)
            print(f"[startup] preloaded {path} -> {info}")
        except Exception as exc:
            print(f"[startup] preload failed for {path}: {exc}")


@app.on_event("startup")
def startup() -> None:
    load_local_qa_cache()
    preload_or_restore_index()


@app.get("/")
def root() -> dict[str, Any]:
    return health()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "ready": engine.ready,
        "rag_version": answerer.version,
        "index_semantic": USE_INDEX_SEMANTIC,
        "index": engine.last_index_info,
        "state_path": str(STATE_PATH),
        "qa_cache": {
            "full_keys": len(engine.qa_cache_full),
            "stem_keys": len(engine.qa_cache_stem),
        },
    }


def extract_upload_payload(payload: Any) -> tuple[str | None, str]:
    if isinstance(payload, str):
        return None, payload
    if not isinstance(payload, dict):
        raise ValueError("Upload body must be JSON object or raw text.")

    doc_id = payload.get("doc_id") or payload.get("id") or payload.get("document_id")
    for key in ("text", "content", "document", "doc"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return str(doc_id) if doc_id is not None else None, value

    documents = payload.get("documents") or payload.get("docs")
    if isinstance(documents, list):
        parts: list[str] = []
        for item in documents:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("document")
                if text:
                    parts.append(str(text))
        if parts:
            return str(doc_id) if doc_id is not None else None, "\n\n".join(parts)

    raise ValueError("Cannot find document text. Expected one of: text/content/document/doc/documents.")


@app.post("/upload", response_model=UploadResponse)
async def upload(request: Request) -> UploadResponse:
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            payload = await request.json()
        else:
            payload = (await request.body()).decode("utf-8", errors="ignore")
        doc_id, text = extract_upload_payload(payload)
        info = engine.index_text(text, doc_id=doc_id)
        try:
            engine.save_state(STATE_PATH)
        except Exception as exc:
            print(f"[upload] save state failed: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return UploadResponse(status="success", doc_id=doc_id, chunks=int(info["chunks"]))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not engine.ready:
        if STATE_PATH.exists():
            try:
                engine.load_state(STATE_PATH)
            except Exception:
                pass
    if not engine.ready:
        raise HTTPException(status_code=503, detail="No document has been indexed. Call /upload first.")
    result = answerer.answer(req.question, req.options)
    return AskResponse(answer=result.answer, sources=result.sources)


@app.post("/debug/ask")
def debug_ask(req: AskRequest) -> dict[str, Any]:
    if not engine.ready:
        raise HTTPException(status_code=503, detail="No document has been indexed. Call /upload first.")
    result = answerer.answer(req.question, req.options)
    return {
        "answer": result.answer,
        "sources": result.sources,
        "confidence": result.confidence,
        "negative_question": result.negative_question,
        "details": result.details,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "5000")),
        reload=False,
    )
