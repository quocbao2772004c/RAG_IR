"""Load configuration from .env for the legal RAG server."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str | None = None, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and (val is None or val == ""):
        raise RuntimeError(f"Missing required env var: {name}")
    return val  # type: ignore[return-value]


# Teacher / Proxy
TEACHER_BASE_URL = _get("TEACHER_BASE_URL", "http://192.168.50.218:8000/api/v1").rstrip("/")
LLM_BASE_URL = _get("LLM_BASE_URL", f"{TEACHER_BASE_URL}/proxy").rstrip("/")

# Student
STUDENT_ID = _get("STUDENT_ID", required=True).strip().upper()
STUDENT_SERVER_URL = _get("STUDENT_SERVER_URL", required=True).rstrip("/")
if not STUDENT_SERVER_URL.startswith(("http://", "https://")):
    raise RuntimeError(
        f"STUDENT_SERVER_URL phai bat dau bang http:// hoac https:// "
        f"(dang la: {STUDENT_SERVER_URL!r})"
    )

# FastAPI
HOST = _get("HOST", "0.0.0.0")
PORT = int(_get("PORT", "5005"))

# LLM
LLM_MODEL = _get("LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY = _get("LLM_API_KEY", STUDENT_ID).strip() or STUDENT_ID
LLM_TEMPERATURE = float(_get("LLM_TEMPERATURE", "0.0"))
LLM_TIMEOUT = float(_get("LLM_TIMEOUT", "45"))
LLM_MAX_TOKENS = int(_get("LLM_MAX_TOKENS", "8"))
LLM_VERIFY = _get("LLM_VERIFY", "true").strip().lower() in {"1", "true", "yes", "on"}

# Embedding / Retrieval
RETRIEVER_BACKEND = _get("RETRIEVER_BACKEND", "bm25").strip().lower()
OPENAI_EMBEDDING_MODEL = _get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_MODEL = _get(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
DEVICE = _get("DEVICE", "cpu")
MODEL_LOCAL_ONLY = _get("MODEL_LOCAL_ONLY", "true").strip().lower() in {"1", "true", "yes", "on"}
CHROMA_DB_DIR = _get("CHROMA_DB_DIR", "chroma_db")
CHROMA_COLLECTION = _get("CHROMA_COLLECTION", "student_rag_luat1")
CHROMA_BATCH_SIZE = int(_get("CHROMA_BATCH_SIZE", "256"))

# Hybrid retrieval, adapted from the original v4/rag.py script.
VECTOR_CANDIDATES = int(_get("VECTOR_CANDIDATES", "80"))
BM25_WEIGHT = float(_get("BM25_WEIGHT", "0.88"))
RERANKER_MODEL = _get("RERANKER_MODEL", "").strip()
RERANK_CANDIDATES = int(_get("RERANK_CANDIDATES", "20"))
RERANKER_BATCH_SIZE = int(_get("RERANKER_BATCH_SIZE", "4"))
RERANKER_MAX_LENGTH = int(_get("RERANKER_MAX_LENGTH", "512"))

# RAG
EMBEDDING_BATCH_SIZE = int(_get("EMBEDDING_BATCH_SIZE", "64"))
CHUNK_SIZE = int(_get("CHUNK_SIZE", "2200"))
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "0"))
TOP_K = int(_get("TOP_K", "14"))
MAX_CONTEXT_CHARS = int(_get("MAX_CONTEXT_CHARS", "18000"))
RETRIEVE_WITH_OPTIONS = _get("RETRIEVE_WITH_OPTIONS", "false").strip().lower() in {"1", "true", "yes", "on"}
QUESTION_BANK_PATHS = _get(
    "QUESTION_BANK_PATHS",
    "../data/generated_questions_2.json;../data/generated_questions.json",
)
QUESTION_BANK_MIN_SIMILARITY = float(_get("QUESTION_BANK_MIN_SIMILARITY", "0.94"))
