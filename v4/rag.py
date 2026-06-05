"""v4 RAG helpers: chunking, embeddings, BM25, hybrid retrieval, optional rerank."""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Any

import numpy as np

import config

_RERANKER_CACHE: dict[tuple[str, str], Any] = {}


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

    for paragraph in paragraphs:
        if len(buf) + len(paragraph) + 2 <= chunk_size:
            buf = f"{buf}\n\n{paragraph}" if buf else paragraph
            continue

        if buf:
            chunks.append(buf)
            buf = buf[-overlap:] if overlap > 0 else ""

        if len(paragraph) > chunk_size:
            start = 0
            while start < len(paragraph):
                end = min(start + chunk_size, len(paragraph))
                piece = paragraph[start:end]
                if buf:
                    piece = buf + "\n" + piece
                    buf = ""
                chunks.append(piece)
                if end == len(paragraph):
                    break
                start = end - overlap if overlap > 0 else end
            buf = chunks[-1][-overlap:] if overlap > 0 else ""
        else:
            buf = (buf + "\n\n" + paragraph) if buf else paragraph

    if buf:
        chunks.append(buf)

    deduped: list[str] = []
    for chunk in chunks:
        if not deduped or deduped[-1] != chunk:
            deduped.append(chunk)
    return deduped


@lru_cache(maxsize=1)
def _load_sbert():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL, device=config.DEVICE)


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


def embed_texts(texts: list[str]) -> np.ndarray:
    backend = config.RETRIEVER_BACKEND
    if backend == "openai":
        from llm_client import embed_openai

        return embed_openai(texts)
    if backend in {"sbert", "hybrid", "vector"}:
        return _sbert_encode(texts)
    if backend == "bm25":
        return np.zeros((len(texts), 0), dtype=np.float32)
    raise RuntimeError(f"embed_texts not supported for RETRIEVER_BACKEND={backend!r}")


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def tokenize_text(text: str) -> list[str]:
    words = re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


def bm25_rank(question: str, chunks: list[str], k: int) -> list[tuple[int, float]]:
    query_terms = tokenize_text(question)
    if not query_terms or not chunks:
        return []

    tokenized_docs = [tokenize_text(chunk) for chunk in chunks]
    doc_count = len(tokenized_docs)
    avg_doc_len = sum(len(tokens) for tokens in tokenized_docs) / max(doc_count, 1)
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized_docs:
        doc_freq.update(set(tokens))

    k1 = 1.5
    b = 0.75
    ranked: list[tuple[int, float]] = []
    for index, tokens in enumerate(tokenized_docs):
        term_freq = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        for term in query_terms:
            freq = term_freq.get(term, 0)
            if not freq:
                continue
            idf = math.log(1 + (doc_count - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = freq + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1))
            score += idf * (freq * (k1 + 1)) / denom
        if score > 0:
            ranked.append((index, score))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:k]


def normalize_scores(scores: dict[int, float], higher_is_better: bool) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    if not higher_is_better:
        values = [-value for value in values]
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return {key: 1.0 for key in scores}
    normalized: dict[int, float] = {}
    for key, value in scores.items():
        comparable = value if higher_is_better else -value
        normalized[key] = (comparable - min_value) / (max_value - min_value)
    return normalized


def load_reranker(model_name: str, device: str):
    cache_key = (model_name, device)
    if cache_key in _RERANKER_CACHE:
        return _RERANKER_CACHE[cache_key]

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    reranker = {"torch": torch, "tokenizer": tokenizer, "model": model, "device": device}
    _RERANKER_CACHE[cache_key] = reranker
    return reranker


def rerank_results(question: str, results: list[tuple[str, float]]) -> list[tuple[str, float]]:
    model_name = config.RERANKER_MODEL
    if not model_name or not results:
        return results[: config.TOP_K]

    reranker = load_reranker(model_name, config.DEVICE)
    torch = reranker["torch"]
    tokenizer = reranker["tokenizer"]
    model = reranker["model"]
    device = reranker["device"]

    scored: list[tuple[str, float]] = []
    for start in range(0, len(results), config.RERANKER_BATCH_SIZE):
        batch = results[start : start + config.RERANKER_BATCH_SIZE]
        pairs = [[question, chunk] for chunk, _score in batch]
        encoded = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=config.RERANKER_MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
            scores = logits.view(-1).float().cpu().tolist()
        scored.extend((chunk, float(score)) for (chunk, _base_score), score in zip(batch, scores))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[: config.TOP_K]
