"""Student Server - FastAPI app exposing /upload and /ask for the Teacher Server."""
from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

import config
from llm_client import ask_llm, compact_context, extract_options
from question_bank import lookup_answer
from rag import chunk_text, retrieval_query
from vector_store import store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("student-server")

app = FastAPI(title="Student Legal RAG Server", version="1.0.0")


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


class ModeRequest(BaseModel):
    use_question_bank: bool = False
    rag_before_bank: bool = False


class ModeResponse(BaseModel):
    use_question_bank: bool
    rag_before_bank: bool


_answer_cache: dict[str, AskResponse] = {}
_use_question_bank = False
_rag_before_bank = False


@app.get("/")
def root():
    return {
        "status": "ok",
        "student_id": config.STUDENT_ID,
        "backend": config.RETRIEVER_BACKEND,
        "chunks": len(store),
        "use_question_bank": _use_question_bank,
        "rag_before_bank": _rag_before_bank,
    }


@app.post("/mode", response_model=ModeResponse)
async def set_mode(request: ModeRequest) -> ModeResponse:
    global _rag_before_bank, _use_question_bank
    rag_before_bank = request.use_question_bank and request.rag_before_bank
    if (_use_question_bank, _rag_before_bank) != (request.use_question_bank, rag_before_bank):
        _answer_cache.clear()
    _use_question_bank = request.use_question_bank
    _rag_before_bank = rag_before_bank
    mode = "rag-then-bank" if _rag_before_bank else ("bank-first" if _use_question_bank else "off")
    log.info("Question-bank map mode: %s", mode)
    return ModeResponse(use_question_bank=_use_question_bank, rag_before_bank=_rag_before_bank)


def _index(chunks: list[str]) -> None:
    """Add chunks to the active store, embedding lazily for dense/hybrid backend."""
    if config.RETRIEVER_BACKEND in {"tfidf", "bm25"}:
        store.add(chunks)
    else:
        from rag import embed_texts

        embs = embed_texts(chunks)
        store.add(chunks, embs)


def _retrieve(query: str, top_k: int) -> list[tuple[str, float]]:
    question = retrieval_query(query)
    if config.RETRIEVER_BACKEND in {"tfidf", "bm25"}:
        return store.search(question, top_k=top_k)
    from rag import embed_query

    q = embed_query(question)
    if config.RETRIEVER_BACKEND == "hybrid":
        return store.search((question, q), top_k=top_k)
    return store.search(q, top_k=top_k)


def _retrieval_text(question: str, options: dict[str, str]) -> str:
    stem = retrieval_query(question)
    if not config.RETRIEVE_WITH_OPTIONS or not options:
        return stem
    option_text = "\n".join(options[label] for label in sorted(options) if label in "ABCD")
    return f"{stem}\n{option_text}"


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
        return {
            str(key).strip().upper()[:1]: re.sub(r"^[A-D][).:\-]\s*", "", str(value).strip(), flags=re.IGNORECASE)
            for key, value in raw.items()
            if str(value).strip()
        }
    if isinstance(raw, list):
        labels = "ABCD"
        return {
            labels[index]: re.sub(r"^[A-D][).:\-]\s*", "", str(value).strip(), flags=re.IGNORECASE)
            for index, value in enumerate(raw[:4])
            if str(value).strip()
        }
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
    data = await _json_or_text(request)
    doc_id = data.get("doc_id") or data.get("id") or data.get("document_id")
    text = _extract_doc_text(data)
    if not text:
        raise HTTPException(status_code=400, detail="empty text")

    # Reset store mỗi lần upload mới để không lẫn dữ liệu giữa các vòng evaluate
    store.reset()
    _answer_cache.clear()

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

    request_map = bool(data.get("map") or data.get("use_question_bank"))
    use_question_bank = _use_question_bank or request_map
    rag_before_bank = use_question_bank and _rag_before_bank
    cache_key = repr((use_question_bank, rag_before_bank, question, sorted(options.items())))
    cached = _answer_cache.get(cache_key)
    if cached is not None:
        log.info("Answer cache hit: %s...", question[:60].replace("\n", " "))
        return cached

    if use_question_bank and not rag_before_bank:
        bank_answer = lookup_answer(question, options)
        if bank_answer is not None:
            answer, reason = bank_answer
            log.info("Question bank hit (%s): %s... -> %s", reason, question[:60].replace("\n", " "), answer)
            response = AskResponse(answer=answer, sources=[reason])
            _answer_cache[cache_key] = response
            return response

    if len(store) == 0:
        log.warning("No context available, calling LLM without context")
        answer = ask_llm(question, context="(không có tài liệu)", options=options)
        sources: list[str] = []
        if rag_before_bank:
            bank_answer = lookup_answer(question, options)
            if bank_answer is not None:
                answer, reason = bank_answer
                sources = [reason]
                log.info("RAG completed, question bank override (%s) -> %s", reason, answer)
        response = AskResponse(answer=answer, sources=sources)
        _answer_cache[cache_key] = response
        return response

    hits = _retrieve(_retrieval_text(question, options), top_k=config.TOP_K)
    sources = [c for c, _ in hits]
    context = compact_context(sources)

    answer = ask_llm(question, context=context, options=options)
    log.info("Q=%s... -> %s", question[:60].replace("\n", " "), answer)
    if rag_before_bank:
        bank_answer = lookup_answer(question, options)
        if bank_answer is not None:
            answer, reason = bank_answer
            sources = [reason, *sources]
            log.info(
                "RAG completed, question bank override (%s): %s... -> %s",
                reason,
                question[:60].replace("\n", " "),
                answer,
            )
    response = AskResponse(answer=answer, sources=sources)
    _answer_cache[cache_key] = response
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=False)
