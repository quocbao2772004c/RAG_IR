"""RAG utilities: chunking + retriever (TF-IDF default, optional sentence-transformers)."""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

import config


# --------- Chunking ---------

def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split text into overlapping chunks, respecting paragraph boundaries when possible."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    text = clean_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= chunk_size:
            buf = f"{buf}\n\n{p}" if buf else p
            continue
        if buf:
            chunks.append(buf)
            buf = buf[-overlap:] if overlap > 0 else ""
        if len(p) > chunk_size:
            start = 0
            while start < len(p):
                end = min(start + chunk_size, len(p))
                piece = p[start:end]
                if buf:
                    piece = buf + "\n" + piece
                    buf = ""
                chunks.append(piece)
                if end == len(p):
                    break
                start = end - overlap if overlap > 0 else end
            buf = chunks[-1][-overlap:] if overlap > 0 else ""
        else:
            buf = (buf + "\n\n" + p) if buf else p

    if buf:
        chunks.append(buf)

    deduped: list[str] = []
    for c in chunks:
        if not deduped or deduped[-1] != c:
            deduped.append(c)
    return deduped


# --------- Sentence-transformers backend (optional) ---------

@lru_cache(maxsize=1)
def _load_sbert():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def _sbert_encode(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    model = _load_sbert()
    embs = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    return embs.astype(np.float32)


# --------- Public API ---------

def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts according to RETRIEVER_BACKEND."""
    backend = config.RETRIEVER_BACKEND
    if backend == "openai":
        from llm_client import embed_openai

        return embed_openai(texts)
    if backend == "sbert":
        return _sbert_encode(texts)
    raise RuntimeError("embed_texts not supported for backend=tfidf")


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
