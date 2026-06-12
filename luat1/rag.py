"""Legal-document chunking and retrieval helpers."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import Any

import numpy as np

import config

_RERANKER_CACHE: dict[tuple[str, str], Any] = {}
_OPTION_START_RE = re.compile(r"(?:^|\s)[ABCD][\).:\-]\s+", re.IGNORECASE)
_ARTICLE_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_SECTION_RE = re.compile(r"^###\s+(.+)$", re.MULTILINE)
_CLAUSE_RE = re.compile(r"(?<!Điều )(?<!Khoản )(?<!khoản )(?=(?:\d{1,2})\.\s+\S)")
_POINT_RE = re.compile(r"(?=(?:[a-zđ])\)\s+\S)", re.IGNORECASE)
_LEGAL_ID_RE = re.compile(r"\bĐiều\s+[\d.]+(?:[A-ZĐa-zđÀ-ỹ]+)?(?:\.[\wÀ-ỹ]+)*", re.UNICODE)
_DOCUMENT_NO_RE = re.compile(r"\b\d{1,4}/\d{4}/[A-ZĐ\-]+(?:-[A-ZĐ]+)*\b", re.UNICODE)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def retrieval_query(question: str) -> str:
    match = _OPTION_START_RE.search(question or "")
    stem = question[: match.start()].strip() if match else (question or "").strip()
    return stem or question


def _section(article: str, name: str) -> str:
    pattern = re.compile(
        rf"^###\s+{re.escape(name)}\s*$\n(.*?)(?=^###\s+|\Z)",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(article)
    return match.group(1).strip() if match else ""


def _info_value(info: str, label: str) -> str:
    match = re.search(rf"^-\s*{re.escape(label)}:\s*(.+)$", info, re.MULTILINE | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _split_long_unit(unit: str, max_chars: int) -> list[str]:
    if len(unit) <= max_chars:
        return [unit]
    points = [part.strip() for part in _POINT_RE.split(unit) if part.strip()]
    if len(points) > 1:
        return _pack_units(points, max_chars)
    sentences = [part.strip() for part in re.split(r"(?<=[.;:!?])\s+", unit) if part.strip()]
    if len(sentences) > 1:
        return _pack_units(sentences, max_chars)
    return [unit[start : start + max_chars] for start in range(0, len(unit), max_chars)]


def _pack_units(units: list[str], max_chars: int) -> list[str]:
    packed: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > max_chars:
            if current:
                packed.append(current)
                current = ""
            packed.extend(_split_long_unit(unit, max_chars))
            continue
        candidate = f"{current} {unit}".strip()
        if current and len(candidate) > max_chars:
            packed.append(current)
            current = unit
        else:
            current = candidate
    if current:
        packed.append(current)
    return packed


def _article_chunks(article: str, chunk_size: int) -> list[str]:
    title_match = _ARTICLE_RE.search(article)
    if not title_match:
        return []
    title = title_match.group(1).strip()
    info = _section(article, "Thông tin")
    content = _section(article, "Nội dung")
    source = re.sub(r"\s+", " ", _section(article, "Nguồn")).strip()
    source = re.sub(r"\s*---\s*$", "", source).strip()
    cross_refs = _section(article, "Liên kết chéo")
    topic = _info_value(info, "Chủ đề")
    subject = _info_value(info, "Đề mục")

    linked_ids: list[str] = []
    for legal_id in _LEGAL_ID_RE.findall(cross_refs):
        if legal_id not in linked_ids:
            linked_ids.append(legal_id)
    prefix_parts = [f"ĐIỀU: {title}"]
    if topic:
        prefix_parts.append(f"CHỦ ĐỀ: {topic}")
    if subject:
        prefix_parts.append(f"ĐỀ MỤC: {subject}")
    if source:
        prefix_parts.append(f"NGUỒN: {source}")
    if linked_ids:
        prefix_parts.append(f"LIÊN KẾT: {'; '.join(linked_ids[:30])}")
    prefix = "\n".join(prefix_parts)

    max_body = max(300, chunk_size - len(prefix) - 12)
    clauses = [part.strip() for part in _CLAUSE_RE.split(content) if part.strip()]
    body_chunks = _pack_units(clauses or [content], max_body)
    return [f"{prefix}\nNỘI DUNG: {body}".strip() for body in body_chunks if body.strip()]


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split Markdown legal contexts by article, then by legal clauses."""
    del overlap
    chunk_size = chunk_size or config.CHUNK_SIZE
    text = clean_text(text)
    if not text:
        return []

    starts = [match.start() for match in _ARTICLE_RE.finditer(text)]
    chunks: list[str] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        chunks.extend(_article_chunks(text[start:end], chunk_size))

    if not chunks:
        return _pack_units([part.strip() for part in text.split("\n\n") if part.strip()], chunk_size)

    deduped: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk not in seen:
            seen.add(chunk)
            deduped.append(chunk)
    return deduped


@lru_cache(maxsize=1)
def _load_sbert():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        config.EMBEDDING_MODEL,
        device=config.DEVICE,
        local_files_only=config.MODEL_LOCAL_ONLY,
    )


def _sbert_encode(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    model = _load_sbert()
    embeddings = model.encode(
        texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    return embeddings.astype(np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    backend = config.RETRIEVER_BACKEND
    if backend == "openai":
        from llm_client import embed_openai

        return embed_openai(texts)
    if backend in {"sbert", "hybrid", "vector"}:
        return _sbert_encode(texts)
    if backend in {"bm25", "tfidf"}:
        return np.zeros((len(texts), 0), dtype=np.float32)
    raise RuntimeError(f"embed_texts not supported for RETRIEVER_BACKEND={backend!r}")


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def normalize_search_text(text: str) -> str:
    text = unicodedata.normalize("NFD", (text or "").lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return text.replace("đ", "d")


def tokenize_text(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9_./%-]+", normalize_search_text(text), flags=re.UNICODE)
    bigrams = [f"{first}_{second}" for first, second in zip(words, words[1:])]
    trigrams = [f"{first}_{second}_{third}" for first, second, third in zip(words, words[1:], words[2:])]
    return words + bigrams + trigrams


def prepare_exact_query(question: str) -> dict[str, Any]:
    normalized_question = normalize_search_text(question)
    legal_ids = {normalize_search_text(item) for item in _LEGAL_ID_RE.findall(question)}
    document_numbers = {normalize_search_text(item) for item in _DOCUMENT_NO_RE.findall(question)}
    acronyms = {
        acronym.lower()
        for acronym in re.findall(r"\b[A-ZĐ]{2,12}\b", question or "")
        if acronym not in {"QUESTION", "ANSWER"}
    }
    numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", normalized_question))
    words = normalized_question.split()
    weighted_phrases: list[tuple[str, float]] = []
    for size in (6, 5, 4, 3):
        phrases = {" ".join(words[index : index + size]) for index in range(len(words) - size + 1)}
        weighted_phrases.extend((phrase, size * 0.8) for phrase in phrases)
    return {
        "legal_ids": legal_ids,
        "document_numbers": document_numbers,
        "acronyms": acronyms,
        "numbers": numbers,
        "weighted_phrases": weighted_phrases,
    }


def prepared_exact_match_bonus(prepared: dict[str, Any], normalized_chunk: str) -> float:
    bonus = 0.0
    bonus += 45.0 * sum(item in normalized_chunk for item in prepared["legal_ids"])
    bonus += 40.0 * sum(item in normalized_chunk for item in prepared["document_numbers"])
    bonus += 20.0 * sum(item in normalized_chunk for item in prepared["acronyms"])
    bonus += 5.0 * sum(number in normalized_chunk for number in prepared["numbers"])
    bonus += sum(
        weight for phrase, weight in prepared["weighted_phrases"] if phrase in normalized_chunk
    )
    return bonus


def bm25_scores(
    question: str,
    chunks: list[str],
    term_freqs: list[Counter[str]],
    doc_freq: Counter[str],
    avg_doc_len: float,
    normalized_chunks: list[str] | None = None,
) -> list[tuple[int, float]]:
    query_terms = tokenize_text(question)
    if not query_terms or not chunks:
        return []
    normalized_chunks = normalized_chunks or [normalize_search_text(chunk) for chunk in chunks]
    prepared_exact = prepare_exact_query(question)
    doc_count = len(term_freqs)
    k1 = 1.5
    b = 0.68
    ranked: list[tuple[int, float]] = []
    for index, (term_freq, normalized_chunk) in enumerate(zip(term_freqs, normalized_chunks)):
        doc_len = sum(term_freq.values())
        score = prepared_exact_match_bonus(prepared_exact, normalized_chunk)
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
    return ranked


def normalize_scores(scores: dict[int, float], higher_is_better: bool) -> dict[int, float]:
    if not scores:
        return {}
    values = [value if higher_is_better else -value for value in scores.values()]
    low, high = min(values), max(values)
    if high == low:
        return {key: 1.0 for key in scores}
    return {
        key: ((value if higher_is_better else -value) - low) / (high - low)
        for key, value in scores.items()
    }


def load_reranker(model_name: str, device: str):
    cache_key = (model_name, device)
    if cache_key in _RERANKER_CACHE:
        return _RERANKER_CACHE[cache_key]
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=config.MODEL_LOCAL_ONLY)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=config.MODEL_LOCAL_ONLY)
    model.to(device)
    model.eval()
    reranker = {"torch": torch, "tokenizer": tokenizer, "model": model, "device": device}
    _RERANKER_CACHE[cache_key] = reranker
    return reranker


def rerank_results(question: str, results: list[tuple[str, float]], top_k: int) -> list[tuple[str, float]]:
    if not config.RERANKER_MODEL or not results:
        return results[:top_k]
    reranker = load_reranker(config.RERANKER_MODEL, config.DEVICE)
    scored: list[tuple[str, float]] = []
    for start in range(0, len(results), config.RERANKER_BATCH_SIZE):
        batch = results[start : start + config.RERANKER_BATCH_SIZE]
        encoded = reranker["tokenizer"](
            [[question, chunk] for chunk, _score in batch],
            padding=True,
            truncation=True,
            max_length=config.RERANKER_MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(reranker["device"]) for key, value in encoded.items()}
        with reranker["torch"].inference_mode():
            scores = reranker["model"](**encoded).logits.view(-1).float().cpu().tolist()
        scored.extend((chunk, float(score)) for (chunk, _base), score in zip(batch, scores))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]
