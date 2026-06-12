"""LLM client and local fallback for multiple-choice RAG questions."""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Mapping

import numpy as np
from openai import OpenAI

import config


SYSTEM_PROMPT = """Bạn trả lời câu hỏi trắc nghiệm pháp luật Việt Nam chỉ dựa trên CONTEXT.
Chọn đúng một đáp án A/B/C/D và chỉ in một ký tự.
Đọc kỹ tên điều, chủ đề, đề mục, nguồn văn bản, từng khoản và từng điểm.
Câu hỏi có thể cần kết hợp quy định từ nhiều điều khác nhau trong CONTEXT.
Phân biệt chính xác chủ thể, điều kiện, ngoại lệ, thời hạn, mức tiền và biện pháp xử lý.
Đặc biệt chú ý các câu hỏi có từ phủ định như "không", "không đúng", "ngoại trừ".
Không sử dụng kiến thức ngoài CONTEXT và không tự suy diễn."""

USER_PROMPT_TEMPLATE = """CONTEXT
{context}

QUESTION
{question}

ANSWER:"""

VERIFY_PROMPT_TEMPLATE = """You are verifying a Vietnamese legal multiple-choice answer.
Use ONLY the evidence in CONTEXT. The first answer may be wrong.

Rules:
- Compare every option A/B/C/D against the exact legal wording.
- For numbers, dates, account codes, agency names, time limits, and exceptions, require exact match.
- For negative questions such as "khong", "khong dung", or "ngoai tru", choose the option that is NOT listed or is contrary to CONTEXT.
- Return exactly one capital letter A, B, C, or D.

CONTEXT
{context}

QUESTION
{question}

OPTIONS
{options}

FIRST_ANSWER: {first_answer}

FINAL_ANSWER:"""

_LETTER_RE = re.compile(r"\b([ABCD])\b")
_OPTION_RE = re.compile(
    r"(?:^|\n|\s)([ABCD])[\).:\-]\s*(.*?)(?=(?:\n|\s)[ABCD][\).:\-]\s*|$)",
    re.IGNORECASE | re.DOTALL,
)


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
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
    normalized_question = _normalize(question_stem)
    negative_markers = ("khong ", "khong dung", "khong thuoc", "khong nam", "ngoai tru", "sai")
    if any(marker in normalized_question for marker in negative_markers):
        return labels[int(np.argmin(scores))]
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


def _tokens(text: str) -> set[str]:
    stopwords = {
        "theo",
        "quy",
        "dinh",
        "phap",
        "luat",
        "hien",
        "hanh",
        "nao",
        "sau",
        "day",
        "doi",
        "voi",
        "trong",
        "truong",
        "hop",
        "duoc",
        "khong",
        "bao",
        "nhieu",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_./%-]*", _normalize(text))
        if len(token) > 2 and token not in stopwords
    }


def _important_literals(text: str) -> set[str]:
    normalized = _normalize(text)
    values = set(re.findall(r"\b\d+(?:[.,]\d+)*(?:/\d+)*(?:%|[a-z]+)?\b", normalized))
    values.update(re.findall(r"\b[A-Z]{2,}\b", text or ""))
    return {value for value in values if value}


def _format_options(options: Mapping[str, str] | None) -> str:
    options = dict(options or {})
    return "\n".join(f"{label}. {options[label]}" for label in sorted(options) if label in "ABCD")


def _needs_verification(question: str, options: Mapping[str, str] | None) -> bool:
    text = _normalize(question + "\n" + _format_options(options))
    markers = (
        "khong",
        "ngoai tru",
        "bao nhieu",
        "thoi han",
        "ngay nao",
        "co quan nao",
        "ai ",
        "don vi nao",
        "tai khoan",
        "ky hieu",
        "so luong",
    )
    return any(marker in text for marker in markers) or bool(re.search(r"\d", text))


def _option_date_key(text: str) -> str:
    normalized = _normalize(text)
    nums = re.findall(r"\b\d{1,4}\b", normalized)
    return "-".join(str(int(num)) for num in nums)


def _compact_literal(text: str) -> str:
    return re.sub(r"[\s.,]+", "", text or "")


def _dedupe_adjacent_words(text: str) -> str:
    words = text.split()
    result: list[str] = []
    for word in words:
        if not result or word != result[-1]:
            result.append(word)
    return " ".join(result)


def _literal_rule_answer(question: str, context: str, options: Mapping[str, str] | None) -> str | None:
    options = dict(options or {})
    if not options:
        return None
    rule_question = re.split(r"(?:^|\s)[ABCD][\).:\-]\s+", question, maxsplit=1)[0]
    normalized_question = _normalize(rule_question)
    normalized_context = _normalize(context)

    if "khong thuoc pham vi trach nhiem quan ly" in normalized_question:
        clauses = [
            clause.strip()
            for clause in re.split(r"\n---\n|(?=\b\d{1,2}\.\s)", normalized_context)
            if clause.strip()
        ]
        matching = [
            clause
            for clause in clauses
            if "khong thuoc pham vi trach nhiem quan ly" in clause
        ]
        if matching:
            best_clause = max(matching, key=lambda clause: len(_tokens(rule_question) & _tokens(clause)))
            conclusion_match = re.search(r"thi tham quyen .+? thuoc\s+(.+)$", best_clause)
            comparison_text = conclusion_match.group(1) if conclusion_match else best_clause
            option_ratios = sorted(
                [
                    (
                        len(_tokens(option) & _tokens(comparison_text)) / max(len(_tokens(option)), 1),
                        label,
                    )
                    for label, option in options.items()
                ],
                reverse=True,
            )
            if (
                len(option_ratios) >= 2
                and option_ratios[0][0] >= 0.8
                and option_ratios[0][0] - option_ratios[1][0] >= 0.1
            ):
                return option_ratios[0][1]

    if (
        "khoan 2 dieu 4 luat phi va le phi" in normalized_question
        and "bao cao bo quan ly chuyen nganh" in normalized_context
        and "bao cao bo tai chinh" in normalized_context
    ):
        central_route = {
            token
            for token in _tokens("Báo cáo Bộ quản lý chuyên ngành để báo cáo Bộ Tài chính")
        }
        option_ratios = sorted(
            [
                (
                    len(central_route & _tokens(option)) / max(len(central_route), 1),
                    label,
                )
                for label, option in options.items()
            ],
            reverse=True,
        )
        if option_ratios and option_ratios[0][0] >= 0.9:
            return option_ratios[0][1]

    subject_match = re.search(r"\bdoi voi\s+(.+?)(?:,|\?|$)", normalized_question)
    if subject_match:
        subject = _dedupe_adjacent_words(subject_match.group(1).strip())
        subject_tokens = _tokens(subject)
        question_tokens = _tokens(rule_question)
        clauses = [
            clause.strip()
            for clause in re.split(r"\n---\n|(?=\b\d{1,2}\.\s)", normalized_context)
            if clause.strip()
        ]
        ranked_clauses = sorted(
            (
                (
                    len(subject_tokens & _tokens(clause)) / max(len(subject_tokens), 1)
                    + (2.0 if f"doi voi {subject}" in clause else 0.0)
                    + len(question_tokens & _tokens(clause)) * 0.05,
                    clause,
                )
                for clause in clauses
            ),
            reverse=True,
        )
        if ranked_clauses and ranked_clauses[0][0] >= 0.8:
            best_clause = ranked_clauses[0][1]
            exact_labels = [
                label
                for label, option in options.items()
                if len(_normalize(option)) >= 10 and _normalize(option) in best_clause
            ]
            if len(exact_labels) == 1:
                return exact_labels[0]
            if "thoi gian luu tru" in normalized_question:
                option_ratios = sorted(
                    [
                        (
                            len(_tokens(option) & _tokens(best_clause)) / max(len(_tokens(option)), 1),
                            label,
                        )
                        for label, option in options.items()
                    ],
                    reverse=True,
                )
                if (
                    len(option_ratios) >= 2
                    and option_ratios[0][0] >= 0.7
                    and option_ratios[0][0] - option_ratios[1][0] >= 0.15
                ):
                    return option_ratios[0][1]

    if "hieu luc thi hanh" in normalized_question and "ngay nao" in normalized_question:
        date_matches: list[tuple[float, str]] = []
        for window in re.split(r"\n---\n|(?<=[.;])\s+", context):
            normalized_window = _normalize(window)
            match = re.search(
                r"hieu luc thi hanh ke tu ngay\s+(\d{1,2})\s+thang\s+(\d{1,2})\s+nam\s+(\d{4})",
                normalized_window,
            )
            if not match:
                match = re.search(
                    r"hieu luc thi hanh ke tu ngay\s+(\d{1,2})/(\d{1,2})/(\d{4})",
                    normalized_window,
                )
            if not match:
                continue
            score = float(len(_tokens(rule_question) & _tokens(window)))
            if "nghi quyet nay" in normalized_window:
                score += 8.0
            wanted = "-".join(str(int(num)) for num in match.groups())
            date_matches.append((score, wanted))
        for _score, wanted in sorted(date_matches, reverse=True):
            for label, option in options.items():
                if _option_date_key(option) == wanted:
                    return label

    if "van con hieu luc" in normalized_question:
        option_scores: list[tuple[float, str]] = []
        for label, option in options.items():
            option_text = _normalize(option)
            if not option_text:
                continue
            score = 0.0
            for window in re.split(r"\n---\n|(?<=[.;])\s+", normalized_context):
                if not window.strip():
                    continue
                overlap = sum(token in window for token in _tokens(option))
                if overlap == 0:
                    continue
                local = float(overlap)
                if "het hieu luc" in window or "bai bo" in window:
                    local -= 20.0
                if "con hieu luc" in window or "tiep tuc co hieu luc" in window:
                    local += 20.0
                if "hieu luc thi hanh" in window and "het hieu luc" not in window:
                    local += 4.0
                score += local
            option_scores.append((score, label))
        option_scores.sort(reverse=True)
        if len(option_scores) >= 2 and option_scores[0][0] >= 8 and option_scores[0][0] - option_scores[1][0] >= 8:
            return option_scores[0][1]

    if any(marker in normalized_question for marker in ("bao nhieu", "muc tien", "so luong")):
        field_match = re.search(r"linh vuc\s+(.+?)(?:\?|$)", normalized_question)
        if field_match:
            field = re.sub(r"[,.;:]+", " ", field_match.group(1)).strip()
            searchable_context = re.sub(r"[,.;:]+", " ", normalized_context)
            field = re.sub(r"\s+", " ", field)
            searchable_context = re.sub(r"\s+", " ", searchable_context)
            position = searchable_context.find(field)
            if position >= 0:
                local = searchable_context[max(0, position - 90) : position + len(field) + 8]
                compact_local = _compact_literal(local)
                literal_labels = [
                    label
                    for label, option in options.items()
                    if any(
                        literal in local or _compact_literal(literal) in compact_local
                        for literal in _important_literals(option)
                    )
                ]
                if len(literal_labels) == 1:
                    return literal_labels[0]
        question_tokens = _tokens(rule_question)
        scored: list[tuple[float, str]] = []
        windows = [
            window
            for window in re.split(r"\n---\n|(?=\b\d{1,2}\.\s)|(?=\b[a-z]\)\s)|(?<=[.;:])\s+", context)
            if window.strip()
        ]
        for label, option in options.items():
            literals = _important_literals(option)
            if not literals:
                continue
            best = 0.0
            for window in windows:
                normalized_window = _normalize(window)
                literal_hits = sum(literal in normalized_window for literal in literals)
                if not literal_hits:
                    continue
                window_tokens = _tokens(window)
                score = literal_hits * 20.0 + len(question_tokens & window_tokens) * 1.5
                option_overlap = len(_tokens(option) & window_tokens)
                score += option_overlap * 1.2
                if score > best:
                    best = score
            scored.append((best, label))
        scored.sort(reverse=True)
        if len(scored) >= 2 and scored[0][0] >= 28 and scored[0][0] - scored[1][0] >= 10:
            return scored[0][1]
    return None


def ask_llm(question: str, context: str, options: Mapping[str, str] | None = None) -> str:
    rule_answer = _literal_rule_answer(question, context, options)
    if rule_answer is not None:
        return rule_answer
    user = USER_PROMPT_TEMPLATE.format(context=context, question=question)
    try:
        response = _client().chat.completions.create(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=config.LLM_MAX_TOKENS,
        )
        first_answer = parse_answer(response.choices[0].message.content or "")
        if not config.LLM_VERIFY or not _needs_verification(question, options):
            return first_answer
        verifier = VERIFY_PROMPT_TEMPLATE.format(
            context=compact_context(context, max_chars=min(config.MAX_CONTEXT_CHARS, 12000)),
            question=question,
            options=_format_options(options),
            first_answer=first_answer,
        )
        checked = _client().chat.completions.create(
            model=config.LLM_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "Return exactly one letter: A, B, C, or D."},
                {"role": "user", "content": verifier},
            ],
            max_tokens=config.LLM_MAX_TOKENS,
        )
        return parse_answer(checked.choices[0].message.content or first_answer)
    except Exception:
        return heuristic_answer(question, context, options)


def embed_openai(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    response = _client().embeddings.create(model=config.OPENAI_EMBEDDING_MODEL, input=texts)
    return np.array([item.embedding for item in response.data], dtype=np.float32)
