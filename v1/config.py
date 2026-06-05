"""Load configuration from .env"""
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
        f"STUDENT_SERVER_URL phải bắt đầu bằng http:// hoặc https:// "
        f"(đang là: {STUDENT_SERVER_URL!r})"
    )

# FastAPI
HOST = _get("HOST", "0.0.0.0")
PORT = int(_get("PORT", "5000"))

# LLM
LLM_MODEL = _get("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(_get("LLM_TEMPERATURE", "0.0"))
LLM_TIMEOUT = float(_get("LLM_TIMEOUT", "45"))

# Embedding / Retrieval
RETRIEVER_BACKEND = _get("RETRIEVER_BACKEND", "tfidf").strip().lower()
OPENAI_EMBEDDING_MODEL = _get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_MODEL = _get(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# RAG
CHUNK_SIZE = int(_get("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "100"))
TOP_K = int(_get("TOP_K", "5"))
MAX_CONTEXT_CHARS = int(_get("MAX_CONTEXT_CHARS", "6500"))
