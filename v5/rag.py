"""Structure-aware retrieval helpers optimized for PTIT admissions text."""
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
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)(?:[.)])?\s+\S")
_ROMAN_HEADING_RE = re.compile(r"^(I|II|III|IV|V|VI|VII|VIII|IX|X)[.)]?\s+\S")
_LETTER_HEADING_RE = re.compile(r"^[a-zđ][.)]\s+\S", re.IGNORECASE)
_SEMESTER_RE = re.compile(r"^Học kỳ\s+\d+", re.IGNORECASE)
_KNOWN_HEADING_RE = re.compile(
    r"^(Tổng quan|Chuẩn đầu ra|Cấu trúc chương trình|Nghề nghiệp|Học phí|"
    r"Điều kiện tuyển sinh|Quy trình nhập học|Tài liệu đào tạo|"
    r"Thông tin chung|Tuyển sinh|Chính sách|Địa chỉ|Thông tin liên hệ)$",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving tabs because they encode table columns."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    lines = []
    blank = False
    for raw_line in text.split("\n"):
        cells = [re.sub(r"[ ]+", " ", cell).strip() for cell in raw_line.split("\t")]
        line = "\t".join(cells).strip()
        if line:
            lines.append(line)
            blank = False
        elif not blank:
            lines.append("")
            blank = True
    return "\n".join(lines).strip()


def retrieval_query(question: str) -> str:
    """Remove multiple-choice options so retrieval focuses on the question stem."""
    match = _OPTION_START_RE.search(question or "")
    stem = question[: match.start()].strip() if match else (question or "").strip()
    return stem or question


def _heading_level(line: str) -> int | None:
    if "\t" in line or len(line) > 180:
        return None
    if _ROMAN_HEADING_RE.match(line) or (line.isupper() and len(line) >= 8):
        return 1
    numbered = _NUMBERED_HEADING_RE.match(line)
    if numbered:
        return min(2 + numbered.group(1).count("."), 4)
    if _LETTER_HEADING_RE.match(line):
        return 4
    if _SEMESTER_RE.match(line):
        return 3
    if _KNOWN_HEADING_RE.match(line):
        return 2
    return None


def _split_long_text(text: str, max_size: int, overlap: int) -> list[str]:
    if len(text) <= max_size:
        return [text]
    sentences = re.split(r"(?<=[.!?;:])\s+", text)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_size:
            current = f"{current} {sentence}".strip()
            continue
        if current:
            pieces.append(current)
        if len(sentence) <= max_size:
            current = sentence
            continue
        start = 0
        while start < len(sentence):
            end = min(start + max_size, len(sentence))
            pieces.append(sentence[start:end])
            if end == len(sentence):
                break
            start = max(start + 1, end - overlap)
        current = ""
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Create section-aware chunks and preserve every TSV table row with its header."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    lines = clean_text(text).splitlines()
    if not lines:
        return []

    chunks: list[str] = []
    buffer: list[str] = []
    headings: dict[int, str] = {}
    table_header = ""
    table_group = ""

    def context() -> str:
        return " > ".join(headings[level] for level in sorted(headings))

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        body = "\n".join(buffer).strip()
        prefix = f"NGỮ CẢNH: {context()}\n" if context() else ""
        for piece in _split_long_text(body, max(200, chunk_size - len(prefix)), overlap):
            chunks.append(prefix + piece)
        buffer = []

    for line in lines:
        if not line:
            flush()
            continue

        level = _heading_level(line)
        if level is not None:
            flush()
            headings = {key: value for key, value in headings.items() if key < level}
            headings[level] = line
            table_header = ""
            table_group = ""
            continue

        if "\t" in line:
            flush()
            cells = [cell.strip() for cell in line.split("\t")]
            if cells and cells[0].upper() == "TT":
                table_header = line
                table_group = ""
                continue
            if len(cells) == 2 and re.fullmatch(r"[IVX]+", cells[0].upper()):
                table_group = line
                continue
            prefix_parts = []
            if context():
                prefix_parts.append(f"NGỮ CẢNH: {context()}")
            if table_group:
                prefix_parts.append(f"NHÓM BẢNG: {table_group}")
            if table_header:
                prefix_parts.append(f"CỘT: {table_header}")
            prefix_parts.append(f"HÀNG: {line}")
            # Keep the complete row and its table context together, even when it is
            # slightly longer than the prose chunk target.
            chunks.append("\n".join(prefix_parts))
            continue

        candidate = "\n".join(buffer + [line])
        prefix_size = len(context()) + 12 if context() else 0
        if buffer and len(candidate) + prefix_size > chunk_size:
            flush()
        buffer.append(line)

    flush()

    deduped: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        normalized = chunk.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
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
    words = re.findall(r"[a-z0-9_]+", normalize_search_text(text), flags=re.UNICODE)
    bigrams = [f"{first}_{second}" for first, second in zip(words, words[1:])]
    return words + bigrams


def _exact_match_bonus(question: str, chunk: str) -> float:
    normalized_question = normalize_search_text(question)
    normalized_chunk = normalize_search_text(chunk)
    bonus = 0.0
    codes = set(re.findall(r"\b[a-z]*\d[\w.-]*\b", normalized_question))
    bonus += 25.0 * sum(code in normalized_chunk for code in codes)
    acronyms = {
        acronym.lower()
        for acronym in re.findall(r"\b[A-ZĐ]{2,8}\b", question or "")
        if acronym not in {"QUESTION", "ANSWER"}
    }
    bonus += 30.0 * sum(acronym in normalized_chunk for acronym in acronyms)
    numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", normalized_question))
    bonus += 2.0 * sum(number in normalized_chunk for number in numbers)
    question_words = normalized_question.split()
    for size in (4, 3):
        phrases = {" ".join(question_words[index : index + size]) for index in range(len(question_words) - size + 1)}
        bonus += 0.8 * sum(phrase in normalized_chunk for phrase in phrases)
    intent_phrases = (
        "ma nganh",
        "ma xet tuyen",
        "chi tieu",
        "hoc phi",
        "hoc bong",
        "co so phia bac",
        "co so phia nam",
        "chuong trinh chat luong cao",
        "chuong trinh lien ket",
    )
    bonus += 8.0 * sum(
        phrase in normalized_question and phrase in normalized_chunk
        for phrase in intent_phrases
    )
    return bonus


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
    b = 0.65
    ranked: list[tuple[int, float]] = []
    for index, (chunk, tokens) in enumerate(zip(chunks, tokenized_docs)):
        term_freq = Counter(tokens)
        doc_len = len(tokens)
        score = _exact_match_bonus(question, chunk)
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
    return {
        key: ((value if higher_is_better else -value) - min_value) / (max_value - min_value)
        for key, value in scores.items()
    }


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


def rerank_results(question: str, results: list[tuple[str, float]], top_k: int | None = None) -> list[tuple[str, float]]:
    top_k = top_k or config.TOP_K
    model_name = config.RERANKER_MODEL
    if not model_name or not results:
        return results[:top_k]

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
            scores = model(**encoded).logits.view(-1).float().cpu().tolist()
        scored.extend((chunk, float(score)) for (chunk, _base_score), score in zip(batch, scores))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]
