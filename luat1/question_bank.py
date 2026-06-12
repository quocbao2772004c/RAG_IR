"""Exact and near-exact answer lookup from generated legal QA files."""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import config

log = logging.getLogger("student-server")


@dataclass(frozen=True)
class BankEntry:
    question: str
    question_key: str
    correct_letter: str
    correct_text: str
    options: tuple[str, ...]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.lower().replace("đ", "d")
    text = re.sub(r"\b[abcd][).:\-]\s*", " ", text)
    text = re.sub(r"[^a-z0-9%.,/]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _question_key(question: str) -> str:
    question = re.split(r"(?:^|\s)[ABCD][).:\-]\s+", question or "", maxsplit=1)[0]
    return _normalize(question)


def _clean_option(text: str) -> str:
    return re.sub(r"^[A-D][).:\-]\s*", "", (text or "").strip(), flags=re.IGNORECASE)


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path.strip())
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parent / path).resolve()


def _iter_paths() -> list[Path]:
    raw_paths = [
        part.strip()
        for part in re.split(r"[;,\n]", config.QUESTION_BANK_PATHS)
        if part.strip()
    ]
    return [_resolve_path(path) for path in raw_paths]


def _load_entries_from_file(path: Path) -> list[BankEntry]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        log.warning("Could not load question bank %s: %s", path, error)
        return []
    entries: list[BankEntry] = []
    for item in data if isinstance(data, list) else []:
        qa = item.get("qa_data") if isinstance(item, dict) else None
        if not isinstance(qa, dict):
            continue
        question = str(qa.get("question") or "").strip()
        raw_options = qa.get("options") or []
        correct_answer = str(qa.get("correct_answer") or "").strip()
        if not question or not correct_answer:
            continue
        correct_letter = correct_answer[:1].upper()
        if correct_letter not in "ABCD":
            continue
        correct_text = _clean_option(correct_answer[2:] if len(correct_answer) > 2 else correct_answer)
        options = tuple(_clean_option(str(option)) for option in raw_options)
        entries.append(
            BankEntry(
                question=question,
                question_key=_question_key(question),
                correct_letter=correct_letter,
                correct_text=correct_text,
                options=options,
            )
        )
    return entries


@lru_cache(maxsize=1)
def load_bank() -> tuple[dict[str, BankEntry], tuple[BankEntry, ...]]:
    by_key: dict[str, BankEntry] = {}
    entries: list[BankEntry] = []
    for path in _iter_paths():
        loaded = _load_entries_from_file(path)
        entries.extend(loaded)
        for entry in loaded:
            by_key.setdefault(entry.question_key, entry)
    log.info("Loaded %d question-bank entries from %d files", len(entries), len(_iter_paths()))
    return by_key, tuple(entries)


def _letter_for_current_options(entry: BankEntry, options: Mapping[str, str]) -> str:
    if not options:
        return entry.correct_letter
    correct_key = _normalize(entry.correct_text)
    if correct_key:
        scored = []
        for label, option_text in options.items():
            option_key = _normalize(option_text)
            score = SequenceMatcher(None, correct_key, option_key).ratio()
            if correct_key and (correct_key in option_key or option_key in correct_key):
                score += 0.25
            scored.append((score, label))
        scored.sort(reverse=True)
        if scored and scored[0][0] >= 0.72:
            return scored[0][1]
    labels = sorted(label for label in options if label in "ABCD")
    if labels and entry.correct_letter in "ABCD":
        original_index = "ABCD".index(entry.correct_letter)
        if original_index < len(labels):
            return labels[original_index]
    return entry.correct_letter


def lookup_answer(question: str, options: Mapping[str, str]) -> tuple[str, str] | None:
    by_key, entries = load_bank()
    key = _question_key(question)
    entry = by_key.get(key)
    if entry is not None:
        return _letter_for_current_options(entry, options), "question_bank_exact"

    best_score = 0.0
    best_entry: BankEntry | None = None
    for candidate in entries:
        if not candidate.question_key:
            continue
        score = SequenceMatcher(None, key, candidate.question_key).ratio()
        if score > best_score:
            best_score = score
            best_entry = candidate
    if best_entry is not None and best_score >= config.QUESTION_BANK_MIN_SIMILARITY:
        return _letter_for_current_options(best_entry, options), f"question_bank_fuzzy:{best_score:.3f}"
    return None
