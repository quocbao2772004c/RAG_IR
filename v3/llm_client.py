"""LLM client calling Teacher proxy (OpenAI-compatible)."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Mapping

import numpy as np
from openai import OpenAI

import config


SYSTEM_PROMPT = (
    "Trả lời trắc nghiệm từ CONTEXT. "
    "Chọn đúng 1 đáp án A/B/C/D. "
    "Chỉ in 1 ký tự."
)

USER_PROMPT_TEMPLATE = """CONTEXT
{context}

QUESTION
{question}

ANSWER: """


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.STUDENT_ID,
        timeout=config.LLM_TIMEOUT,
        max_retries=0,
    )


_LETTER_RE = re.compile(r"\b([ABCD])\b")
_OPTION_RE = re.compile(
    r"(?:^|\n|\s)([ABCD])[\).:\-]\s*(.*?)(?=(?:\n|\s)[ABCD][\).:\-]\s*|$)",
    re.IGNORECASE | re.DOTALL,
)


def parse_answer(text: str) -> str:
    """Extract a single letter A/B/C/D from raw LLM output."""
    if not text:
        return "A"
    s = text.strip().upper()
    if s in {"A", "B", "C", "D"}:
        return s
    m = _LETTER_RE.search(s)
    if m:
        return m.group(1)
    # try first alphabetic char
    for ch in s:
        if ch in "ABCD":
            return ch
    return "A"  # safe fallback


def extract_options(question: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for letter, text in _OPTION_RE.findall(question or ""):
        letter = letter.upper()
        cleaned = re.sub(r"\s+", " ", text).strip()
        if cleaned:
            options[letter] = cleaned
    return options


def heuristic_answer(question: str, context: str, options: Mapping[str, str] | None = None) -> str:
    """Cheap local fallback if proxy LLM is unavailable."""
    options = dict(options or extract_options(question))
    if not options:
        return "A"

    labels = sorted(k for k in options if k in "ABCD")
    if not labels:
        return "A"

    q_lower = question.lower()
    for label in labels:
        opt = options[label].lower()
        if any(key in q_lower for key in ("đăng ký", "dang ky", "register")):
            if "register" in opt:
                return label
        if any(key in q_lower for key in ("gửi tài liệu", "gui tai lieu", "send document", "upload")):
            if "upload" in opt:
                return label
        if any(key in q_lower for key in ("câu hỏi", "cau hoi", "question")):
            if "ask" in opt:
                return label

    stem = re.split(r"(?:^|\s)[ABCD][\).:\-]\s*", question, maxsplit=1)[0]
    q_tokens = set(re.findall(r"[\wÀ-ỹ/]+", stem.lower(), flags=re.UNICODE))
    weak = {
        "la", "là", "gi", "gì", "nao", "nào", "de", "để", "cua", "của", "tren", "trên",
        "trong", "mot", "một", "cac", "các", "server", "endpoint",
    }
    q_tokens = {tok for tok in q_tokens if tok not in weak and len(tok) > 1}

    windows = [
        part.strip()
        for part in re.split(r"[\n\r]+|(?<=[.!?])\s+", context or "")
        if part.strip()
    ]
    if not windows:
        windows = [context or question]

    scores = []
    for label in labels:
        option_text = options[label].lower()
        opt_tokens = re.findall(r"[\wÀ-ỹ/]+", option_text, flags=re.UNICODE)
        exact = option_text.strip()
        score = 0.0
        for window in windows:
            w = window.lower()
            w_tokens = set(re.findall(r"[\wÀ-ỹ/]+", w, flags=re.UNICODE))
            q_overlap = len(q_tokens & w_tokens)
            opt_overlap = sum(1 for tok in opt_tokens if tok in w_tokens)
            exact_hit = 2 if exact and exact in w else 0
            if opt_overlap or exact_hit:
                score = max(score, (q_overlap + 1) * (opt_overlap + exact_hit))
        scores.append(score)
    return labels[int(np.argmax(scores))]


def compact_context(chunks: list[str] | str, max_chars: int | None = None) -> str:
    max_chars = max_chars or config.MAX_CONTEXT_CHARS
    if isinstance(chunks, str):
        chunks = [chunks]
    picked: list[str] = []
    used = 0
    for chunk in chunks:
        c = re.sub(r"\s+", " ", chunk).strip()
        if not c:
            continue
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(c) > remaining:
            c = c[:remaining]
        picked.append(c)
        used += len(c) + 5
    return "\n---\n".join(picked)


def ask_llm(question: str, context: str, options: Mapping[str, str] | None = None) -> str:
    user = USER_PROMPT_TEMPLATE.format(context=context, question=question)
    try:
        res = _client().chat.completions.create(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=4,
        )
        raw = res.choices[0].message.content or ""
        return parse_answer(raw)
    except Exception:
        return heuristic_answer(question, context, options)


def embed_openai(texts: list[str]) -> np.ndarray:
    """Gọi OpenAI Embeddings qua proxy của Teacher Server.

    Trả về mảng (N, D) float32 (chưa chuẩn hoá; vector_store sẽ tự L2 normalize).
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    res = _client().embeddings.create(
        model=config.OPENAI_EMBEDDING_MODEL,
        input=texts,
    )
    embs = np.array([d.embedding for d in res.data], dtype=np.float32)
    return embs
