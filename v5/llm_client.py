"""LLM client and local fallback for multiple-choice RAG questions."""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Mapping

import numpy as np
from openai import OpenAI

import config


SYSTEM_PROMPT = """Bạn trả lời câu hỏi trắc nghiệm chỉ dựa trên CONTEXT.
Chọn đúng một đáp án A/B/C/D và chỉ in một ký tự.
Đọc kỹ ngữ cảnh, tiêu đề mục, cơ sở Bắc/Nam, loại chương trình và từng hàng bảng.
Ưu tiên hàng chứa đúng mã ngành, tên chương trình, con số hoặc điều kiện được hỏi.
Không gộp dữ liệu từ hai hàng bảng hay hai mục khác nhau."""

USER_PROMPT_TEMPLATE = """CONTEXT
{context}

QUESTION
{question}

ANSWER:"""

_LETTER_RE = re.compile(r"\b([ABCD])\b")
_OPTION_RE = re.compile(
    r"(?:^|\n|\s)([ABCD])[\).:\-]\s*(.*?)(?=(?:\n|\s)[ABCD][\).:\-]\s*|$)",
    re.IGNORECASE | re.DOTALL,
)


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.STUDENT_ID,
        timeout=config.LLM_TIMEOUT,
        max_retries=0,
    )


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", (text or "").lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return text.replace("đ", "d")


def parse_answer(text: str) -> str:
    if not text:
        return "A"
    normalized = text.strip().upper()
    if normalized in {"A", "B", "C", "D"}:
        return normalized
    match = _LETTER_RE.search(normalized)
    if match:
        return match.group(1)
    return next((char for char in normalized if char in "ABCD"), "A")


def extract_options(question: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for letter, text in _OPTION_RE.findall(question or ""):
        cleaned = re.sub(r"\s+", " ", text).strip()
        if cleaned:
            options[letter.upper()] = cleaned
    return options


def heuristic_answer(question: str, context: str, options: Mapping[str, str] | None = None) -> str:
    """Choose the option with the strongest exact and token overlap in retrieved context."""
    options = dict(options or extract_options(question))
    labels = sorted(label for label in options if label in "ABCD")
    if not labels:
        return "A"

    blocks = [_normalize(block) for block in context.split("\n---\n") if block.strip()]
    question_stem = re.split(r"(?:^|\s)[ABCD][\).:\-]\s+", question, maxsplit=1)[0]
    question_tokens = {
        token
        for token in re.findall(r"[a-z0-9_]+", _normalize(question_stem))
        if len(token) > 2 and token not in {"bao", "nhieu", "nhiêu", "nam", "năm"}
    }
    scores: list[float] = []
    for label in labels:
        option = _normalize(options[label])
        option_tokens = re.findall(r"[a-z0-9_]+", option)
        score = 0.0
        for rank, block in enumerate(blocks, start=1):
            weight = 1.0 / rank
            windows = [window for window in re.split(r"\n|(?<=[.;])\s+", block) if window.strip()]
            for window in windows:
                window_tokens = set(re.findall(r"[a-z0-9_]+", window))
                relevance = 1.0 + len(question_tokens & window_tokens)
                score += weight * relevance * 0.2 * sum(
                    token in window_tokens for token in option_tokens if len(token) > 1
                )
                if option and option in window:
                    score += 20.0 * weight * relevance
                for number in re.findall(r"\b\d+(?:[.,]\d+)?%?\b", option):
                    if number in window:
                        score += 10.0 * weight * relevance
        scores.append(score)
    return labels[int(np.argmax(scores))]


def compact_context(chunks: list[str] | str, max_chars: int | None = None) -> str:
    max_chars = max_chars or config.MAX_CONTEXT_CHARS
    chunks = [chunks] if isinstance(chunks, str) else chunks
    picked: list[str] = []
    used = 0
    for chunk in chunks:
        cleaned = re.sub(r"[ \t]+", " ", chunk).strip()
        if not cleaned:
            continue
        remaining = max_chars - used
        if remaining <= 0:
            break
        picked.append(cleaned[:remaining])
        used += len(picked[-1]) + 5
    return "\n---\n".join(picked)


def ask_llm(question: str, context: str, options: Mapping[str, str] | None = None) -> str:
    user = USER_PROMPT_TEMPLATE.format(context=context, question=question)
    try:
        response = _client().chat.completions.create(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=4,
        )
        return parse_answer(response.choices[0].message.content or "")
    except Exception:
        return heuristic_answer(question, context, options)


def embed_openai(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    response = _client().embeddings.create(model=config.OPENAI_EMBEDDING_MODEL, input=texts)
    return np.array([item.embedding for item in response.data], dtype=np.float32)
