from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests

from rag_engine import (
    AnswerResult,
    HybridRAG,
    clean_option_text,
    compact_source,
    focus_tokens,
    informative_tokens,
    is_negative_question,
    normalize_number_token,
    normalize_text,
    parse_options,
    unique_keep_order,
)


AVAILABLE_VERSIONS = ("v1", "v2", "v3", "v4", "v5", "v6", "v7")

WINDOW_SPLIT_RE = re.compile(
    r"(?=\b\d{1,2}\.\s)|(?=\b[a-zđ]\)\s)|(?<=[.;:])\s+|\n+",
    re.IGNORECASE,
)
NEXT_STEP_RE = re.compile(
    r"\b(buoc tiep theo|sau buoc|sau khi|truoc khi|tiep theo)\b"
)
SYMBOL_RE = re.compile(r"\b(ky hieu|ma|tai khoan|so hieu|ghi la|viet tat|trang thai)\b")
NEGATIVE_HINT_RE = re.compile(r"\b(khong|chua|cam|nghiem cam|ngoai)\b")
NEGATION_TERMS = {
    "khong",
    "chua",
    "chẳng",
    "ngoai",
    "tru",
    "cam",
    "nghiem",
}


def split_windows(text: str, limit: int = 900) -> list[str]:
    parts: list[str] = []
    for raw in WINDOW_SPLIT_RE.split(text):
        part = re.sub(r"\s+", " ", raw).strip()
        if len(part) >= 20:
            parts.append(part[:limit])
    if not parts:
        compact = re.sub(r"\s+", " ", text).strip()
        return [compact[:limit]] if compact else []

    merged: list[str] = []
    for part in parts:
        if merged and len(part) < 70:
            merged[-1] = f"{merged[-1]} {part}"[:limit]
        else:
            merged.append(part)
    return merged


def bigrams(tokens: list[str]) -> set[str]:
    return {f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)}


def number_set(text: str) -> set[str]:
    return {
        normalize_number_token(token)
        for token in informative_tokens(text)
        if re.search(r"\d", token)
    }


def weighted_recall(tokens: list[str], window_tokens: list[str], idf: dict[str, float]) -> float:
    token_set = set(tokens)
    if not token_set:
        return 0.0
    window_set = set(window_tokens)
    denom = sum(idf.get(token, 1.0) for token in token_set) or 1.0
    return sum(idf.get(token, 1.0) for token in token_set & window_set) / denom


def is_next_step_question(question: str) -> bool:
    return bool(NEXT_STEP_RE.search(normalize_text(question)))


def negation_penalty(option: str, window: str) -> float:
    option_norm = normalize_text(option)
    window_norm = normalize_text(window)
    penalty = 0.0
    for phrase in (
        "khong duoc",
        "khong co",
        "khong thuoc",
        "chua co",
        "khong phai",
        "ngoai",
    ):
        if phrase in option_norm and phrase not in window_norm:
            penalty += 0.22
        if phrase in window_norm and phrase not in option_norm:
            penalty += 0.12

    option_terms = set(option_norm.split()) & NEGATION_TERMS
    window_terms = set(window_norm.split()) & NEGATION_TERMS
    if "khong" in option_terms and "khong" not in window_terms:
        penalty += 0.10
    if "chua" in option_terms and "chua" not in window_terms:
        penalty += 0.10
    return min(0.45, penalty)


class VersionedRAG:
    def __init__(
        self,
        rag: HybridRAG,
        version: str = "v5",
        use_semantic_reranker: bool = True,
        semantic_model_name: str | None = None,
    ) -> None:
        if version not in AVAILABLE_VERSIONS:
            raise ValueError(f"Unknown RAG version: {version}")
        self.rag = rag
        self.version = version
        self.use_semantic_reranker = use_semantic_reranker
        self.semantic_model_name = (
            semantic_model_name
            or os.getenv("RAG_RERANK_MODEL")
            or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self._semantic_model: Any = None

    def answer(self, question: str, options: list[str] | None = None) -> AnswerResult:
        return self.answer_many([{"question": question, "options": options}])[0]

    def answer_many(self, items: list[dict[str, Any]]) -> list[AnswerResult]:
        if self.version == "v1":
            return [self.rag.answer(item["question"], item.get("options")) for item in items]

        candidates: list[dict[str, AnswerResult]] = []
        for item in items:
            question = item["question"]
            options = item.get("options")
            per_item = {
                "v1": self.rag.answer(question, options),
                "v2": self._answer_v2(question, options),
            }
            if self.version in {"v3", "v4", "v5", "v6", "v7"}:
                per_item["v3"] = self._answer_v3(question, options)
            candidates.append(per_item)

        if self.version == "v2":
            return [row["v2"] for row in candidates]
        if self.version == "v3":
            return [row["v3"] for row in candidates]

        if self.version in {"v5", "v6", "v7"} and self.use_semantic_reranker:
            semantic = self._semantic_answer_many(items)
            for row, sem_result in zip(candidates, semantic):
                if sem_result is not None:
                    row["v5_semantic"] = sem_result

        out: list[AnswerResult] = []
        for item, row in zip(items, candidates):
            out.append(
                self._ensemble(
                    item["question"],
                    row,
                    include_semantic=self.version in {"v5", "v6", "v7"},
                )
            )
        if self.version in {"v6", "v7"}:
            local_llm = self._local_llm_answer_many(items, verify=self.version == "v7")
            out = [llm_result or base_result for llm_result, base_result in zip(local_llm, out)]
        return out

    def _answer_v2(self, question: str, options: list[str] | None) -> AnswerResult:
        stem, parsed = parse_options(question, options)
        letters = [letter for letter in "ABCD" if letter in parsed]
        if not letters:
            return self.rag.answer(question, options)

        all_options = " ".join(parsed[letter] for letter in letters)
        hits = self.rag.retrieve(f"{stem} {all_options}".strip(), k=16)
        hits += self.rag.retrieve(stem, k=12)
        indices = unique_keep_order([hit.index for hit in hits])[:24]
        windows = self._candidate_windows(indices)

        q_tokens = focus_tokens(stem)
        q_bigrams = bigrams(q_tokens)
        q_nums = number_set(stem)

        scores: dict[str, float] = {}
        best_windows: dict[str, str] = {}
        for letter in letters:
            option = clean_option_text(parsed[letter])
            opt_tokens = informative_tokens(option)
            opt_bigrams = bigrams(opt_tokens)
            opt_nums = number_set(option)
            option_norm = normalize_text(option)
            best_score = -1.0
            best_window = ""

            for rank, window, window_norm, window_tokens in windows:
                q_recall = weighted_recall(q_tokens, window_tokens, self.rag.token_idf)
                window_bigrams = bigrams(window_tokens)
                q_bigram = len(q_bigrams & window_bigrams) / (len(q_bigrams) or 1)
                q_num = (
                    len(q_nums & number_set(window)) / len(q_nums)
                    if q_nums
                    else 0.0
                )
                relevance = 0.79 * q_recall + 0.06 * q_bigram + 0.15 * q_num

                opt_recall = weighted_recall(opt_tokens, window_tokens, self.rag.token_idf)
                opt_bigram = (
                    len(opt_bigrams & window_bigrams) / len(opt_bigrams)
                    if opt_bigrams
                    else 0.0
                )
                exact = 1.0 if option_norm and option_norm in window_norm else 0.0
                opt_num = (
                    len(opt_nums & number_set(window)) / len(opt_nums)
                    if opt_nums
                    else 0.0
                )
                support = (
                    0.50 * opt_recall
                    + 0.18 * opt_bigram
                    + 0.17 * exact
                    + 0.15 * opt_num
                )
                score = support * (0.35 + 1.25 * relevance) * (0.92**rank)
                if score > best_score:
                    best_score = score
                    best_window = window

            scores[letter] = max(0.0, best_score)
            best_windows[letter] = best_window

        answer = self._select_answer(stem, letters, scores)
        confidence = self._confidence(letters, scores, stem)
        sources = self._sources_for_answer(stem, parsed[answer])
        return AnswerResult(
            answer=answer,
            sources=sources,
            confidence=confidence,
            negative_question=is_negative_question(stem),
            details={
                "version": "v2",
                "scores": scores,
                "best_windows": best_windows,
            },
        )

    def _answer_v3(self, question: str, options: list[str] | None) -> AnswerResult:
        stem, parsed = parse_options(question, options)
        letters = [letter for letter in "ABCD" if letter in parsed]
        if not letters:
            return self.rag.answer(question, options)

        all_options = " ".join(parsed[letter] for letter in letters)
        hits = self.rag.retrieve(stem, k=22)
        hits += self.rag.retrieve(f"{stem} {all_options}".strip(), k=10)[:4]
        indices = unique_keep_order([hit.index for hit in hits])[:24]

        q_tokens = focus_tokens(stem)
        q_bigrams = bigrams(q_tokens)
        q_nums = number_set(stem)
        relevant_windows: list[tuple[float, int, str, str, list[str]]] = []
        for rank, window, window_norm, window_tokens in self._candidate_windows(indices):
            q_recall = weighted_recall(q_tokens, window_tokens, self.rag.token_idf)
            window_bigrams = bigrams(window_tokens)
            q_bigram = len(q_bigrams & window_bigrams) / (len(q_bigrams) or 1)
            q_num = (
                len(q_nums & number_set(window)) / len(q_nums)
                if q_nums
                else 0.0
            )
            relevance = (0.62 * q_recall + 0.24 * q_bigram + 0.14 * q_num) * (0.93**rank)
            if relevance > 0.03 or rank < 5:
                relevant_windows.append((relevance, rank, window, window_norm, window_tokens))
        relevant_windows.sort(key=lambda item: item[0], reverse=True)

        scores: dict[str, float] = {}
        best_windows: dict[str, str] = {}
        for letter in letters:
            option = clean_option_text(parsed[letter])
            opt_tokens = informative_tokens(option)
            opt_bigrams = bigrams(opt_tokens)
            opt_nums = number_set(option)
            option_norm = normalize_text(option)
            fragments = [
                normalize_text(fragment)
                for fragment in re.split(r",|;|\bva\b|\bhoac\b", option, flags=re.IGNORECASE)
                if len(normalize_text(fragment)) >= 12
            ]
            best_score = -1.0
            best_window = ""
            for rank, (relevance, _, window, window_norm, window_tokens) in enumerate(
                relevant_windows[:80]
            ):
                opt_recall = weighted_recall(opt_tokens, window_tokens, self.rag.token_idf)
                window_bigrams = bigrams(window_tokens)
                opt_bigram = (
                    len(opt_bigrams & window_bigrams) / len(opt_bigrams)
                    if opt_bigrams
                    else 0.0
                )
                exact = 1.0 if option_norm and option_norm in window_norm else 0.0
                fragment_hit = 1.0 if any(fragment in window_norm for fragment in fragments) else 0.0
                opt_num = (
                    len(opt_nums & number_set(window)) / len(opt_nums)
                    if opt_nums
                    else 0.0
                )
                support = (
                    0.44 * opt_recall
                    + 0.18 * opt_bigram
                    + 0.24 * exact
                    + 0.08 * fragment_hit
                    + 0.06 * opt_num
                )
                score = (0.35 + 1.15 * relevance) * support
                score -= negation_penalty(option, window)
                if exact and relevance > 0.06:
                    score += 0.18
                if opt_num and q_nums and opt_nums & q_nums:
                    score += 0.08
                score *= 0.985**rank
                if score > best_score:
                    best_score = score
                    best_window = window
            scores[letter] = max(0.0, best_score)
            best_windows[letter] = best_window

        answer = self._select_answer(stem, letters, scores)
        confidence = self._confidence(letters, scores, stem)
        sources = self._sources_for_answer(stem, parsed[answer])
        return AnswerResult(
            answer=answer,
            sources=sources,
            confidence=confidence,
            negative_question=is_negative_question(stem),
            details={
                "version": "v3",
                "scores": scores,
                "best_windows": best_windows,
            },
        )

    def _semantic_answer_many(self, items: list[dict[str, Any]]) -> list[AnswerResult | None]:
        model = self._load_semantic_model()
        if model is None:
            return [None for _ in items]

        all_windows: list[str] = []
        window_ids: dict[str, int] = {}
        per_item_windows: list[list[int]] = []
        query_texts: list[str] = []
        query_meta: list[tuple[int, str]] = []
        parsed_items: list[tuple[str, dict[str, str], list[str]]] = []

        for idx, item in enumerate(items):
            stem, parsed = parse_options(item["question"], item.get("options"))
            letters = [letter for letter in "ABCD" if letter in parsed]
            parsed_items.append((stem, parsed, letters))
            if not letters:
                per_item_windows.append([])
                continue

            all_options = " ".join(parsed[letter] for letter in letters)
            hits = self.rag.retrieve(f"{stem} {all_options}".strip(), k=14)
            hits += self.rag.retrieve(stem, k=10)
            indices = unique_keep_order([hit.index for hit in hits])[:18]
            item_windows: list[int] = []
            for chunk_index in indices:
                for window in split_windows(self.rag.chunks[chunk_index].index_text, limit=700):
                    if window not in window_ids:
                        window_ids[window] = len(all_windows)
                        all_windows.append(window)
                    item_windows.append(window_ids[window])

            capped: list[int] = []
            seen: set[int] = set()
            for window_id in item_windows:
                if window_id not in seen:
                    seen.add(window_id)
                    capped.append(window_id)
                if len(capped) >= 70:
                    break
            per_item_windows.append(capped)

            for letter in letters:
                query_meta.append((idx, letter))
                query_texts.append(f"{stem} [SEP] {parsed[letter]}")

        if not all_windows or not query_texts:
            return [None for _ in items]

        batch_size = int(os.getenv("RAG_RERANK_BATCH_SIZE", "64"))
        try:
            window_embeddings = model.encode(
                all_windows,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            query_embeddings = model.encode(
                query_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception:
            return [None for _ in items]

        window_embeddings = np.asarray(window_embeddings, dtype=np.float32)
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
        scores_by_item: list[dict[str, float]] = [defaultdict(float) for _ in items]  # type: ignore[list-item]
        for query_index, (item_index, letter) in enumerate(query_meta):
            window_indices = per_item_windows[item_index]
            if not window_indices:
                scores_by_item[item_index][letter] = 0.0
                continue
            sims = window_embeddings[window_indices] @ query_embeddings[query_index]
            scores_by_item[item_index][letter] = float(np.max(sims))

        results: list[AnswerResult | None] = []
        for idx, (stem, parsed, letters) in enumerate(parsed_items):
            if not letters:
                results.append(None)
                continue
            scores = dict(scores_by_item[idx])
            answer = self._select_answer(stem, letters, scores)
            confidence = self._confidence(letters, scores, stem)
            sources = self._sources_for_answer(stem, parsed[answer])
            results.append(
                AnswerResult(
                    answer=answer,
                    sources=sources,
                    confidence=confidence,
                    negative_question=is_negative_question(stem),
                    details={"version": "v5_semantic", "scores": scores},
                )
            )
        return results

    def _local_llm_answer_many(
        self,
        items: list[dict[str, Any]],
        verify: bool = False,
    ) -> list[AnswerResult | None]:
        base_url = os.getenv("RAG_LOCAL_LLM_BASE_URL", "").strip().rstrip("/")
        model = os.getenv("RAG_LOCAL_LLM_MODEL", "").strip()
        if not base_url or not model or not self._is_allowed_local_llm_url(base_url):
            return [None for _ in items]

        results: list[AnswerResult | None] = []
        for item in items:
            result = self._local_llm_answer_one(
                item["question"],
                item.get("options"),
                base_url=base_url,
                model=model,
                verify=verify,
            )
            results.append(result)
        return results

    def _local_llm_answer_one(
        self,
        question: str,
        options: list[str] | None,
        base_url: str,
        model: str,
        verify: bool,
    ) -> AnswerResult | None:
        stem, parsed = parse_options(question, options)
        letters = [letter for letter in "ABCD" if letter in parsed]
        if not letters:
            return None
        all_options = "\n".join(f"{letter}. {parsed[letter]}" for letter in letters)
        hits = self.rag.retrieve(f"{stem}\n{all_options}", k=int(os.getenv("RAG_LOCAL_LLM_TOP_K", "10")))
        context_parts: list[str] = []
        used = 0
        max_chars = int(os.getenv("RAG_LOCAL_LLM_CONTEXT_CHARS", "18000"))
        for hit in hits:
            text = self._format_source(hit.chunk)
            remain = max_chars - used
            if remain <= 0:
                break
            context_parts.append(text[:remain])
            used += len(context_parts[-1]) + 5
        context = "\n---\n".join(context_parts)
        prompt = (
            "Bạn là bộ chọn đáp án trắc nghiệm pháp luật Việt Nam. "
            "Chỉ dùng CONTEXT, không dùng kiến thức ngoài. "
            "So sánh từng đáp án A/B/C/D, chú ý phủ định, số tiền, ngày tháng, chủ thể, tài khoản. "
            "Trả về đúng một chữ cái A, B, C hoặc D.\n\n"
            f"CONTEXT\n{context}\n\nQUESTION\n{stem}\n\nOPTIONS\n{all_options}\n\nANSWER:"
        )
        answer = self._call_local_chat(base_url, model, prompt)
        if answer not in letters:
            return None
        if verify:
            verify_prompt = (
                "Kiểm tra lại đáp án đầu tiên bằng CONTEXT. "
                "Nếu sai hãy sửa. Trả về đúng một chữ cái A/B/C/D.\n\n"
                f"CONTEXT\n{context}\n\nQUESTION\n{stem}\n\nOPTIONS\n{all_options}\n\n"
                f"FIRST_ANSWER: {answer}\nFINAL_ANSWER:"
            )
            checked = self._call_local_chat(base_url, model, verify_prompt)
            if checked in letters:
                answer = checked
        return AnswerResult(
            answer=answer,
            sources=context_parts[:5],
            confidence=1.0,
            negative_question=is_negative_question(stem),
            details={"version": "v7_local_llm" if verify else "v6_local_llm", "model": model},
        )

    @staticmethod
    def _call_local_chat(base_url: str, model: str, prompt: str) -> str | None:
        url = f"{base_url}/chat/completions"
        headers = {}
        api_key = os.getenv("RAG_LOCAL_LLM_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            response = requests.post(
                url,
                headers=headers or None,
                json={
                    "model": model,
                    "temperature": 0,
                    "max_tokens": int(os.getenv("RAG_LOCAL_LLM_MAX_TOKENS", "8")),
                    "messages": [
                        {
                            "role": "system",
                            "content": "Return exactly one capital letter: A, B, C, or D.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=float(os.getenv("RAG_LOCAL_LLM_TIMEOUT", "60")),
            )
            response.raise_for_status()
            payload = response.json()
            text = payload["choices"][0]["message"].get("content") or ""
        except Exception:
            return None
        match = re.search(r"\b([ABCD])\b", text.upper())
        if match:
            return match.group(1)
        return next((char for char in text.upper() if char in "ABCD"), None)

    @staticmethod
    def _is_allowed_local_llm_url(base_url: str) -> bool:
        if os.getenv("RAG_ALLOW_NONLOCAL_LLM", "0") == "1":
            return True
        parsed = urlparse(base_url)
        host = (parsed.hostname or "").lower()
        return host in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}

    def _ensemble(
        self,
        question: str,
        candidates: dict[str, AnswerResult],
        include_semantic: bool,
    ) -> AnswerResult:
        if include_semantic and "v5_semantic" in candidates:
            chosen_name = self._calibrated_tree_candidate(question, candidates)
            chosen = candidates[chosen_name]
            return AnswerResult(
                answer=chosen.answer,
                sources=chosen.sources,
                confidence=chosen.confidence,
                negative_question=chosen.negative_question,
                details={
                    "version": self.version,
                    "reason": "calibrated_tree",
                    "chosen": chosen_name,
                    "candidates": {
                        name: {
                            "answer": result.answer,
                            "confidence": result.confidence,
                            "details": result.details,
                        }
                        for name, result in candidates.items()
                    },
                },
            )

        weights = {
            "v1": (0.0, 0.0, 0.8),
            "v2": (0.1, 0.3, 1.5),
            "v3": (0.1, 0.3, 0.8),
        }
        if include_semantic and "v5_semantic" in candidates:
            weights["v5_semantic"] = (0.02, 1.2, 0.3)

        votes: dict[str, float] = defaultdict(float)
        for name, (bias, multiplier, cap) in weights.items():
            result = candidates.get(name)
            if result is None:
                continue
            votes[result.answer] += bias + multiplier * min(float(result.confidence), cap)

        if not votes:
            return candidates["v1"]

        answer = max(votes, key=votes.get)
        reason = "weighted_vote"
        if is_next_step_question(question):
            answer = candidates["v1"].answer
            reason = "next_step_prefers_v1"

        chosen = next(
            (result for result in candidates.values() if result.answer == answer),
            candidates["v1"],
        )
        ordered_votes = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        confidence = 0.0
        if len(ordered_votes) >= 2:
            confidence = ordered_votes[0][1] - ordered_votes[1][1]
        elif ordered_votes:
            confidence = ordered_votes[0][1]

        return AnswerResult(
            answer=answer,
            sources=chosen.sources,
            confidence=round(float(confidence), 4),
            negative_question=chosen.negative_question,
            details={
                "version": self.version,
                "reason": reason,
                "votes": dict(votes),
                "candidates": {
                    name: {
                        "answer": result.answer,
                        "confidence": result.confidence,
                        "details": result.details,
                    }
                    for name, result in candidates.items()
                },
            },
        )

    def _calibrated_tree_candidate(
        self,
        question: str,
        candidates: dict[str, AnswerResult],
    ) -> str:
        v1 = candidates["v1"]
        v2 = candidates["v2"]
        v3 = candidates["v3"]
        v5 = candidates["v5_semantic"]

        conf_v1 = float(v1.confidence)
        conf_v2 = float(v2.confidence)
        conf_v3 = float(v3.confidence)
        conf_v5 = float(v5.confidence)
        ans_v1 = self._answer_value(v1.answer)
        ans_v2 = self._answer_value(v2.answer)
        ans_v5 = self._answer_value(v5.answer)
        v1_eq_v2 = v1.answer == v2.answer
        v1_eq_v3 = v1.answer == v3.answer
        v2_eq_v5 = v2.answer == v5.answer
        norm_question = normalize_text(question)
        symbol = bool(SYMBOL_RE.search(norm_question))
        neg = bool(NEGATIVE_HINT_RE.search(norm_question))

        if not v1_eq_v2:
            if conf_v2 <= 0.12550000101327896:
                if conf_v3 <= 0.11620000004768372:
                    if ans_v2 <= 0.5:
                        if symbol:
                            return "v2"
                        return "v1"
                    if conf_v1 <= 0.04064999893307686:
                        return "v1"
                    return "v2"
                if conf_v2 <= 0.0243500005453825:
                    return "v2"
                return "v1"

            if conf_v5 <= 0.0020819902420043945:
                if not neg:
                    if conf_v2 <= 0.2475000023841858:
                        return "v2"
                    return "v1"
                return "v2"

            if ans_v2 <= 0.5:
                if conf_v3 <= 0.2613000050187111:
                    return "v2"
                return "v1"
            if not v1_eq_v3:
                return "v2"
            return "v2"

        if not v2_eq_v5:
            if conf_v2 <= 0.128200002014637:
                if conf_v1 <= 0.3413500040769577:
                    if conf_v3 <= 0.1843000054359436:
                        return "v2"
                    return "v5_semantic"
                return "v2"

            if ans_v1 <= 0.5:
                if ans_v5 <= 1.5:
                    return "v5_semantic"
                return "v2"
            if conf_v1 <= 0.038399999029934406:
                return "v2"
            return "v2"

        if conf_v3 <= 0.004650000017136335:
            if conf_v5 <= 0.004088729619979858:
                return "v2"
            return "v2"
        return "v2"

    @staticmethod
    def _answer_value(answer: str) -> int:
        return {"A": 0, "B": 1, "C": 2, "D": 3}.get(answer, 0)

    def _candidate_windows(self, indices: list[int]) -> list[tuple[int, str, str, list[str]]]:
        windows: list[tuple[int, str, str, list[str]]] = []
        for rank, chunk_index in enumerate(indices):
            if chunk_index >= len(self.rag.chunks):
                continue
            for window in split_windows(self.rag.chunks[chunk_index].index_text):
                window_norm = normalize_text(window)
                window_tokens = informative_tokens(window)
                if window_tokens:
                    windows.append((rank, window, window_norm, window_tokens))
        return windows

    def _select_answer(self, stem: str, letters: list[str], scores: dict[str, float]) -> str:
        if is_negative_question(stem):
            return min(letters, key=lambda letter: scores.get(letter, 0.0))
        return max(letters, key=lambda letter: scores.get(letter, 0.0))

    def _confidence(self, letters: list[str], scores: dict[str, float], stem: str) -> float:
        ordered = sorted((scores.get(letter, 0.0) for letter in letters), reverse=True)
        if len(ordered) < 2:
            return round(float(ordered[0] if ordered else 0.0), 4)
        return round(float(abs(ordered[0] - ordered[1])), 4)

    def _sources_for_answer(self, stem: str, option: str, k: int = 5) -> list[str]:
        hits = self.rag.retrieve(f"{stem} {option}".strip(), k=k)
        return [self._format_source(hit.chunk) for hit in hits]

    @staticmethod
    def _format_source(chunk: Any) -> str:
        title = chunk.title.strip()
        source = f" [{chunk.source.strip()}]" if chunk.source else ""
        return compact_source(f"{title}{source}\n{chunk.text}")

    def _load_semantic_model(self) -> Any:
        if self._semantic_model is not None:
            return self._semantic_model
        try:
            from sentence_transformers import SentenceTransformer

            self._semantic_model = SentenceTransformer(
                self.semantic_model_name,
                local_files_only=True,
            )
            return self._semantic_model
        except Exception:
            return None
