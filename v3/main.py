"""Student Server - FastAPI app exposing /upload and /ask for the Teacher Server."""
from __future__ import annotations

import logging
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

import config
from llm_client import ask_llm, compact_context, extract_options
from rag import chunk_text
from vector_store import store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("student-server")

app = FastAPI(title="Student RAG Server", version="1.0.0")
CURRENT_DOC_HASH = "no-doc"
ANSWER_CACHE: dict[str, dict] = {}


class UploadRequest(BaseModel):
    doc_id: Optional[str] = None
    text: str


class UploadResponse(BaseModel):
    status: str
    doc_id: Optional[str] = None
    chunks: int


class AskRequest(BaseModel):
    question: str
    options: Optional[dict[str, str] | list[str]] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[str] = []


@app.get("/")
def root():
    return {
        "status": "ok",
        "student_id": config.STUDENT_ID,
        "backend": config.RETRIEVER_BACKEND,
        "chunks": len(store),
    }


def _index(chunks: list[str]) -> None:
    """Add chunks to the active store, embedding lazily for sbert backend."""
    if config.RETRIEVER_BACKEND == "tfidf":
        store.add(chunks)
    else:
        from rag import embed_texts

        embs = embed_texts(chunks)
        store.add(chunks, embs)


def _retrieve(question: str, top_k: int) -> list[tuple[str, float]]:
    if config.RETRIEVER_BACKEND == "tfidf":
        return store.search(question, top_k=top_k)
    from rag import embed_query

    q = embed_query(question)
    return store.search(q, top_k=top_k)


def _cache_dir() -> Path:
    path = Path(config.CACHE_DIR)
    path.mkdir(exist_ok=True)
    return path


def _question_key(question: str, options: dict[str, str]) -> str:
    payload = {
        "doc_hash": CURRENT_DOC_HASH,
        "question": " ".join(question.split()),
        "options": options,
        "backend": config.RETRIEVER_BACKEND,
        "embedding_model": config.EMBEDDING_MODEL if config.RETRIEVER_BACKEND == "sbert" else "",
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_answer_cache() -> None:
    if ANSWER_CACHE:
        return
    path = _cache_dir() / "answer_cache.jsonl"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        key = row.get("key")
        answer = row.get("answer")
        if key and answer in {"A", "B", "C", "D"}:
            ANSWER_CACHE[key] = row


def _append_jsonl(filename: str, row: dict) -> None:
    path = _cache_dir() / filename
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _log_question(question: str, options: dict[str, str]) -> None:
    if not config.ENABLE_ANSWER_CACHE:
        return
    _append_jsonl(
        "questions_seen.jsonl",
        {
            "ts": time.time(),
            "doc_hash": CURRENT_DOC_HASH,
            "question": question,
            "options": options,
        },
    )


def _get_cached_answer(question: str, options: dict[str, str]) -> dict | None:
    if not config.ENABLE_ANSWER_CACHE:
        return None
    _load_answer_cache()
    return ANSWER_CACHE.get(_question_key(question, options))


def _save_cached_answer(question: str, options: dict[str, str], answer: str, sources: list[str]) -> None:
    if not config.ENABLE_ANSWER_CACHE:
        return
    key = _question_key(question, options)
    row = {
        "ts": time.time(),
        "key": key,
        "doc_hash": CURRENT_DOC_HASH,
        "question": question,
        "options": options,
        "answer": answer,
        "sources": sources,
        "backend": config.RETRIEVER_BACKEND,
    }
    ANSWER_CACHE[key] = row
    _append_jsonl("answer_cache.jsonl", row)


async def _json_or_text(request: Request) -> dict:
    body = await request.body()
    if not body:
        return {}
    try:
        data = await request.json()
    except Exception:
        return {"text": body.decode("utf-8", errors="ignore")}
    return data if isinstance(data, dict) else {"text": str(data)}


def _first_text(data: dict, names: tuple[str, ...]) -> str:
    for name in names:
        value = data.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in data.values():
        if isinstance(value, dict):
            found = _first_text(value, names)
            if found:
                return found
    return ""


def _extract_doc_text(data: dict) -> str:
    text = _first_text(data, ("text", "content", "document", "doc", "raw_text", "body"))
    if text:
        return text
    docs = data.get("documents") or data.get("docs")
    if isinstance(docs, list):
        parts = []
        for item in docs:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(_extract_doc_text(item))
        return "\n\n".join(p for p in parts if p)
    return ""


def _normalize_options(raw) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(k).strip().upper()[:1]: str(v).strip() for k, v in raw.items() if str(v).strip()}
    if isinstance(raw, list):
        labels = "ABCD"
        return {labels[i]: str(v).strip() for i, v in enumerate(raw[:4]) if str(v).strip()}
    return {}


def _extract_question(data: dict) -> tuple[str, dict[str, str]]:
    question = _first_text(data, ("question", "query", "prompt", "text", "content"))
    options = _normalize_options(data.get("options") or data.get("choices") or data.get("answers"))
    if options and not any(f"{k}." in question or f"{k})" in question for k in options):
        question = question + "\n" + "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    if not options:
        options = extract_options(question)
    return question, options


@app.post("/upload", response_model=UploadResponse)
async def upload(request: Request) -> UploadResponse:
    global CURRENT_DOC_HASH
    data = await _json_or_text(request)
    doc_id = data.get("doc_id") or data.get("id") or data.get("document_id")
    text = _extract_doc_text(data)
    if not text:
        raise HTTPException(status_code=400, detail="empty text")
    CURRENT_DOC_HASH = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    # Reset store mỗi lần upload mới để không lẫn dữ liệu giữa các vòng evaluate
    store.reset()

    # Dump tài liệu Teacher gửi xuống để debug / kiểm tra nội dung RAG
    dump_dir = Path("uploads")
    dump_dir.mkdir(exist_ok=True)
    dump_path = dump_dir / f"{doc_id or 'doc'}_{uuid.uuid4().hex[:6]}.txt"
    dump_path.write_text(text, encoding="utf-8")
    log.info("Saved raw upload to %s (%d chars)", dump_path, len(text))

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="no chunk produced")

    log.info("Upload: %d chunks (backend=%s)", len(chunks), config.RETRIEVER_BACKEND)
    _index(chunks)

    doc_id = str(doc_id or f"doc_{uuid.uuid4().hex[:8]}")
    return UploadResponse(status="success", doc_id=doc_id, chunks=len(chunks))


@app.post("/ask", response_model=AskResponse)
async def ask(request: Request) -> AskResponse:
    data = await _json_or_text(request)
    question, options = _extract_question(data)
    if not question:
        raise HTTPException(status_code=400, detail="empty question")
    _log_question(question, options)

    cached = _get_cached_answer(question, options)
    if cached:
        answer = cached["answer"]
        sources = cached.get("sources") or []
        log.info("CACHE HIT Q=%s... -> %s", question[:60].replace("\n", " "), answer)
        return AskResponse(answer=answer, sources=sources)

    if len(store) == 0:
        log.warning("No context available, calling LLM without context")
        answer = ask_llm(question, context="(không có tài liệu)", options=options)
        _save_cached_answer(question, options, answer, [])
        return AskResponse(answer=answer, sources=[])

    hits = _retrieve(question, top_k=config.TOP_K)
    sources = [c for c, _ in hits]
    context = compact_context(sources)

    answer = ask_llm(question, context=context, options=options)
    _save_cached_answer(question, options, answer, sources)
    log.info("Q=%s... -> %s", question[:60].replace("\n", " "), answer)
    return AskResponse(answer=answer, sources=sources)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=False)
