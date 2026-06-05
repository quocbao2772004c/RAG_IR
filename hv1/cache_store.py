"""Shared JSON cache helpers for question/answer storage."""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
CACHE_PATH = BASE_DIR / "questions.json"
_LOCK = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_question(question: str) -> str:
    return " ".join((question or "").strip().split())


def question_id(question: str) -> str:
    normalized = normalize_question(question)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def load_cache() -> dict[str, Any]:
    if not CACHE_PATH.exists():
        return {"questions": []}
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"questions": []}
    if not isinstance(data, dict):
        return {"questions": []}
    questions = data.get("questions")
    if not isinstance(questions, list):
        data["questions"] = []
    return data


def save_cache(data: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CACHE_PATH)


def extract_question(payload: dict[str, Any]) -> str:
    for key in ("question", "query", "prompt", "text", "content"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in payload.values():
        if isinstance(value, dict):
            nested = extract_question(value)
            if nested:
                return nested
    return ""


def normalize_options(raw: Any) -> dict[str, str]:
    if isinstance(raw, dict):
        return {
            str(k).strip().upper()[:1]: str(v).strip()
            for k, v in raw.items()
            if str(k).strip() and str(v).strip()
        }
    if isinstance(raw, list):
        labels = "ABCD"
        return {
            labels[i]: str(value).strip()
            for i, value in enumerate(raw[:4])
            if str(value).strip()
        }
    return {}


def extract_options(payload: dict[str, Any]) -> dict[str, str]:
    return normalize_options(
        payload.get("options") or payload.get("choices") or payload.get("answers")
    )


def upsert_question(payload: dict[str, Any]) -> dict[str, Any]:
    question = extract_question(payload)
    if not question:
        raise ValueError("empty question")

    qid = question_id(question)
    options = extract_options(payload)

    with _LOCK:
        data = load_cache()
        questions = data["questions"]
        for item in questions:
            if item.get("id") == qid:
                item["question"] = question
                item["normalized_question"] = normalize_question(question)
                item["options"] = options or item.get("options", {})
                item["last_seen_at"] = now_iso()
                item["seen_count"] = int(item.get("seen_count", 0)) + 1
                item["raw_payload"] = payload
                save_cache(data)
                return item

        item = {
            "id": qid,
            "question": question,
            "normalized_question": normalize_question(question),
            "options": options,
            "answer": "",
            "created_at": now_iso(),
            "last_seen_at": now_iso(),
            "seen_count": 1,
            "raw_payload": payload,
        }
        questions.append(item)
        save_cache(data)
        return item


def find_answer(question: str) -> tuple[str | None, dict[str, Any] | None]:
    qid = question_id(question)
    normalized = normalize_question(question)
    data = load_cache()
    for item in data.get("questions", []):
        if item.get("id") == qid or item.get("normalized_question") == normalized:
            answer = str(item.get("answer", "")).strip().upper()
            return (answer[:1] if answer[:1] in "ABCD" else None), item
    return None, None


def unanswered_questions() -> list[dict[str, Any]]:
    data = load_cache()
    return [
        item
        for item in data.get("questions", [])
        if str(item.get("answer", "")).strip().upper()[:1] not in "ABCD"
    ]


def set_answer(qid: str, answer: str) -> bool:
    answer = answer.strip().upper()[:1]
    if answer not in "ABCD":
        raise ValueError("answer must be A/B/C/D")

    with _LOCK:
        data = load_cache()
        for item in data.get("questions", []):
            if item.get("id") == qid:
                item["answer"] = answer
                item["answered_at"] = now_iso()
                save_cache(data)
                return True
    return False
