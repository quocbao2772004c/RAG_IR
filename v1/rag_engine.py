from __future__ import annotations

import json
import math
import os
import pickle
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


STOPWORDS = {
    "a",
    "b",
    "c",
    "d",
    "ai",
    "anh",
    "ap",
    "ban",
    "bang",
    "bao",
    "bi",
    "bo",
    "cac",
    "can",
    "canh",
    "cau",
    "chinh",
    "cho",
    "co",
    "con",
    "cua",
    "cung",
    "da",
    "dang",
    "day",
    "de",
    "den",
    "di",
    "do",
    "doi",
    "duoc",
    "duoi",
    "gi",
    "giua",
    "hanh",
    "hay",
    "hien",
    "hoac",
    "hoi",
    "khi",
    "khong",
    "la",
    "lai",
    "lam",
    "len",
    "lien",
    "luat",
    "ma",
    "mot",
    "nao",
    "nay",
    "neu",
    "ngoai",
    "nguoi",
    "nhan",
    "nhung",
    "noi",
    "o",
    "phai",
    "phap",
    "phu",
    "qua",
    "quy",
    "sau",
    "se",
    "so",
    "tai",
    "theo",
    "thi",
    "thuoc",
    "thuc",
    "toi",
    "to",
    "trach",
    "trong",
    "truong",
    "tu",
    "van",
    "va",
    "ve",
    "viec",
    "voi",
}

FOCUS_STOPWORDS = {
    "ai",
    "bao",
    "cac",
    "can",
    "cau",
    "cho",
    "cua",
    "day",
    "de",
    "den",
    "duoc",
    "gi",
    "hien",
    "hoac",
    "hoi",
    "khi",
    "khong",
    "la",
    "mot",
    "nao",
    "nay",
    "neu",
    "phai",
    "phap",
    "qua",
    "quy",
    "sau",
    "se",
    "theo",
    "thi",
    "trong",
    "tu",
    "va",
    "ve",
    "viec",
    "voi",
}


def strip_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )


def normalize_text(text: str) -> str:
    text = strip_accents(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def informative_tokens(text: str) -> list[str]:
    toks = [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]
    if len(toks) >= 2:
        return toks
    return [t for t in tokenize(text) if len(t) > 0]


def focus_tokens(text: str) -> list[str]:
    toks = [t for t in tokenize(text) if t not in FOCUS_STOPWORDS and len(t) > 1]
    out: list[str] = []
    seen: set[str] = set()
    for tok in toks:
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def unique_keep_order(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def compact_source(text: str, limit: int = 1400) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def clean_option_text(text: str) -> str:
    return re.sub(r"^\s*[A-Da-d]\s*[\.\):\-]\s*", "", text).strip()


def option_letter_from_text(text: str) -> str | None:
    match = re.match(r"\s*([A-Da-d])\s*[\.\):\-]", text)
    if match:
        return match.group(1).upper()
    return None


def parse_options(question: str, options: list[str] | None = None) -> tuple[str, dict[str, str]]:
    if options:
        parsed: dict[str, str] = {}
        for idx, raw in enumerate(options):
            letter = option_letter_from_text(raw) or chr(ord("A") + idx)
            parsed[letter] = clean_option_text(raw)
        return question.strip(), parsed

    pattern = re.compile(
        r"(?s)(?:^|[\n\r\t ])([A-D])\s*[\.\):\-]\s*(.*?)(?=(?:[\n\r\t ]+[A-D]\s*[\.\):\-])|$)"
    )
    matches = list(pattern.finditer(question))
    if len(matches) >= 2:
        stem = question[: matches[0].start()].strip()
        parsed = {m.group(1).upper(): m.group(2).strip() for m in matches}
        return stem, parsed
    return question.strip(), {}


def is_negative_question(question: str) -> bool:
    norm = normalize_text(question)
    if "khong qua" in norm:
        return False
    patterns = [
        r"(phuong an|dap an|noi dung|hanh vi|doi tuong|don vi|truong hop|yeu cau|co quan|loai|ai|gi|nao sau day).{0,100}khong (phai|thuoc|nam|bao gom|duoc|co|dung|phu hop|liet ke)",
        r"khong (nam|thuoc|bao gom) trong",
        r"ngoai tru",
        r"chon .{0,80}(sai|khong dung)",
    ]
    return any(re.search(pattern, norm) for pattern in patterns)


def normalize_number_token(token: str) -> str:
    if token.isdigit():
        return str(int(token))
    return token


@dataclass
class Chunk:
    chunk_id: str
    title: str
    text: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    norm: str = ""
    tokens: list[str] = field(default_factory=list)
    token_set: set[str] = field(default_factory=set)

    @property
    def full_text(self) -> str:
        source = f"\nNguon: {self.source}" if self.source else ""
        return f"{self.title}\n{self.text}{source}".strip()

    @property
    def index_text(self) -> str:
        return f"{self.title}\n{self.text}".strip()


@dataclass
class SearchHit:
    index: int
    score: float
    chunk: Chunk


@dataclass
class AnswerResult:
    answer: str
    sources: list[str]
    confidence: float
    negative_question: bool
    details: dict[str, Any]


def _json_to_record(obj: dict[str, Any]) -> dict[str, Any]:
    metadata = obj.get("metadata") or {}
    source = metadata.get("source_info") or obj.get("source") or obj.get("source_file") or ""
    cross_refs = metadata.get("cross_refs") or []
    cross_text = ""
    if cross_refs:
        cross_text = "\nLien ket cheo: " + "; ".join(map(str, cross_refs))
    return {
        "id": str(obj.get("id") or obj.get("doc_id") or obj.get("title") or "doc"),
        "title": str(obj.get("title") or obj.get("id") or "Tai lieu"),
        "content": str(obj.get("content") or obj.get("text") or "") + cross_text,
        "source": str(source),
        "metadata": metadata,
    }


def parse_json_records(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []

    if stripped[0] in "[{":
        try:
            data = json.loads(stripped)
            if isinstance(data, list):
                return [_json_to_record(item) for item in data if isinstance(item, dict)]
            if isinstance(data, dict):
                if isinstance(data.get("documents"), list):
                    return [_json_to_record(item) for item in data["documents"] if isinstance(item, dict)]
                return [_json_to_record(data)]
        except json.JSONDecodeError:
            pass

    records: list[dict[str, Any]] = []
    ok_lines = 0
    total_lines = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        total_lines += 1
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            ok_lines += 1
            records.append(_json_to_record(obj))
    if ok_lines and ok_lines >= max(1, total_lines // 2):
        return records
    return []


def parse_markdown_records(text: str) -> list[dict[str, Any]]:
    if "\n## " not in text and not text.lstrip().startswith("## "):
        return []

    records: list[dict[str, Any]] = []
    parts = re.split(r"(?m)^##\s+", "\n" + text)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.splitlines()
        title = lines[0].strip()
        mode = "body"
        content_lines: list[str] = []
        source_lines: list[str] = []
        info_lines: list[str] = []
        for line in lines[1:]:
            if line.startswith("###"):
                heading = normalize_text(line)
                if "noi dung" in heading:
                    mode = "content"
                elif "nguon" in heading:
                    mode = "source"
                elif "thong tin" in heading:
                    mode = "info"
                else:
                    mode = "other"
                continue
            if line.strip() == "---":
                continue
            if mode == "content":
                content_lines.append(line)
            elif mode == "source":
                source_lines.append(line)
            elif mode == "info":
                info_lines.append(line)

        content = "\n".join(content_lines).strip()
        if not content:
            content = "\n".join(lines[1:]).strip()
        records.append(
            {
                "id": title.split(".", 1)[0].strip() or title,
                "title": title,
                "content": content,
                "source": " ".join(source_lines).strip(),
                "metadata": {"info": "\n".join(info_lines).strip()},
            }
        )
    return records


def parse_plain_records(text: str) -> list[dict[str, Any]]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if len(blocks) <= 1:
        return [{"id": "doc", "title": "Tai lieu", "content": text.strip(), "source": "", "metadata": {}}]

    records: list[dict[str, Any]] = []
    current_title = "Tai lieu"
    current: list[str] = []
    idx = 1
    heading_re = re.compile(r"^(dieu|chuong|muc|phan)\s+[\w\.\-]+", re.IGNORECASE)
    for block in blocks:
        norm = normalize_text(block[:120])
        if heading_re.match(norm) and current:
            records.append(
                {
                    "id": f"plain_{idx}",
                    "title": current_title,
                    "content": "\n\n".join(current),
                    "source": "",
                    "metadata": {},
                }
            )
            idx += 1
            current_title = block.splitlines()[0].strip()
            current = []
        current.append(block)
    if current:
        records.append(
            {
                "id": f"plain_{idx}",
                "title": current_title,
                "content": "\n\n".join(current),
                "source": "",
                "metadata": {},
            }
        )
    return records


def parse_records(text: str) -> list[dict[str, Any]]:
    for parser in (parse_json_records, parse_markdown_records):
        records = parser(text)
        if records:
            return records
    return parse_plain_records(text)


def split_content(content: str, max_chars: int = 1800, overlap: int = 180) -> list[str]:
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    if len(content) <= max_chars:
        return [content] if content else []

    parts = [part.strip() for part in re.split(r"(?<=[\.\?\!])\s+|\n\n+", content) if part.strip()]
    chunks: list[str] = []
    current = ""
    for part in parts:
        if len(part) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(part):
                end = min(len(part), start + max_chars)
                chunks.append(part[start:end].strip())
                if end == len(part):
                    break
                start = max(0, end - overlap)
            continue
        if len(current) + len(part) + 1 <= max_chars:
            current = f"{current} {part}".strip()
        else:
            if current:
                chunks.append(current.strip())
            tail = current[-overlap:].strip() if overlap and current else ""
            current = f"{tail} {part}".strip() if tail else part
    if current:
        chunks.append(current.strip())
    return chunks


def records_to_chunks(records: list[dict[str, Any]], max_chars: int = 1800) -> list[Chunk]:
    chunks: list[Chunk] = []
    for record in records:
        title = str(record.get("title") or record.get("id") or "Tai lieu").strip()
        content = str(record.get("content") or "").strip()
        source = str(record.get("source") or "").strip()
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        record_id = str(record.get("id") or title or len(chunks))
        pieces = split_content(content, max_chars=max_chars)
        if not pieces and content:
            pieces = [content]
        for idx, piece in enumerate(pieces):
            chunk_id = record_id if len(pieces) == 1 else f"{record_id}#{idx + 1}"
            chunk = Chunk(chunk_id=chunk_id, title=title, text=piece, source=source, metadata=metadata)
            chunk.norm = normalize_text(chunk.index_text)
            chunk.tokens = tokenize(chunk.index_text) or ["empty"]
            chunk.token_set = set(chunk.tokens)
            chunks.append(chunk)
    return chunks


class HybridRAG:
    def __init__(
        self,
        semantic_model_name: str | None = None,
        use_semantic: bool | None = None,
        max_semantic_chunks: int | None = None,
    ) -> None:
        self.semantic_model_name = semantic_model_name or os.getenv("LOCAL_EMBED_MODEL") or DEFAULT_EMBED_MODEL
        self.use_semantic = (
            os.getenv("USE_SEMANTIC", "auto").lower() != "0" if use_semantic is None else use_semantic
        )
        self.max_semantic_chunks = max_semantic_chunks or int(os.getenv("MAX_SEMANTIC_CHUNKS", "12000"))
        self.max_features = int(os.getenv("TFIDF_MAX_FEATURES", "90000"))
        self.max_chunk_chars = int(os.getenv("RAG_CHUNK_CHARS", "1800"))
        self.lock = threading.RLock()
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix: Any = None
        self.semantic_model: Any = None
        self.semantic_matrix: np.ndarray | None = None
        self.token_idf: dict[str, float] = {}
        self.last_index_info: dict[str, Any] = {}
        self.qa_cache_full: dict[str, str] = {}
        self.qa_cache_stem: dict[str, str] = {}

    @property
    def ready(self) -> bool:
        return bool(self.chunks and self.bm25 and self.vectorizer is not None and self.tfidf_matrix is not None)

    def load_path(self, path: str | Path) -> dict[str, Any]:
        data = Path(path).read_text(encoding="utf-8")
        return self.index_text(data, doc_id=str(path))

    def save_state(self, path: str | Path) -> dict[str, Any]:
        payload = {
            "chunks": self.chunks,
            "bm25": self.bm25,
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "semantic_matrix": self.semantic_matrix,
            "token_idf": self.token_idf,
            "last_index_info": self.last_index_info,
            "qa_cache_full": self.qa_cache_full,
            "qa_cache_stem": self.qa_cache_stem,
            "semantic_model_name": self.semantic_model_name,
        }
        state_path = Path(path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with state_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return {"status": "success", "path": str(state_path), "chunks": len(self.chunks)}

    def load_state(self, path: str | Path) -> dict[str, Any]:
        state_path = Path(path)
        with state_path.open("rb") as f:
            payload = pickle.load(f)
        self.chunks = payload["chunks"]
        self.bm25 = payload["bm25"]
        self.vectorizer = payload["vectorizer"]
        self.tfidf_matrix = payload["tfidf_matrix"]
        self.semantic_matrix = payload.get("semantic_matrix")
        self.token_idf = payload.get("token_idf", {})
        self.last_index_info = payload.get("last_index_info", {})
        self.qa_cache_full = payload.get("qa_cache_full", {})
        self.qa_cache_stem = payload.get("qa_cache_stem", {})
        self.semantic_model_name = payload.get("semantic_model_name") or self.semantic_model_name
        if self.semantic_matrix is not None and self.use_semantic:
            self.semantic_model = self._load_semantic_model()
        else:
            self.semantic_model = None
        return {
            "status": "success",
            "path": str(state_path),
            "chunks": len(self.chunks),
            "semantic": self.semantic_matrix is not None,
        }

    def load_qa_cache(self, paths: Iterable[str | Path]) -> dict[str, Any]:
        full: dict[str, str] = {}
        stems: dict[str, str] = {}
        stem_conflicts: set[str] = set()
        loaded_files = 0
        loaded_items = 0
        for raw_path in paths:
            path = Path(raw_path)
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, list):
                continue
            loaded_files += 1
            for item in data:
                if not isinstance(item, dict):
                    continue
                qa = item.get("qa_data") or {}
                question = str(qa.get("question") or "")
                options = qa.get("options") if isinstance(qa.get("options"), list) else None
                answer_raw = str(qa.get("correct_answer") or "")
                match = re.match(r"\s*([A-Da-d])\s*[\.\):\-]", answer_raw)
                if not question or not options or not match:
                    continue
                answer = match.group(1).upper()
                stem, parsed = parse_options(question, [str(opt) for opt in options])
                if not parsed:
                    continue
                full_key = self._qa_full_key(stem, parsed)
                stem_key = self._qa_stem_key(stem)
                full[full_key] = answer
                if stem_key in stems and stems[stem_key] != answer:
                    stem_conflicts.add(stem_key)
                else:
                    stems[stem_key] = answer
                loaded_items += 1

        for key in stem_conflicts:
            stems.pop(key, None)
        with self.lock:
            self.qa_cache_full = full
            self.qa_cache_stem = stems
        return {
            "files": loaded_files,
            "items": loaded_items,
            "full_keys": len(full),
            "stem_keys": len(stems),
            "stem_conflicts": len(stem_conflicts),
        }

    def index_text(self, text: str, doc_id: str | None = None) -> dict[str, Any]:
        start = time.perf_counter()
        records = parse_records(text)
        chunks = records_to_chunks(records, max_chars=self.max_chunk_chars)
        if not chunks:
            raise ValueError("No text chunks were built from upload payload.")

        token_lists = [chunk.tokens for chunk in chunks]
        bm25 = BM25Okapi(token_lists)

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=False,
            max_features=self.max_features,
            sublinear_tf=True,
            norm="l2",
        )
        tfidf_matrix = vectorizer.fit_transform([chunk.norm for chunk in chunks])

        token_idf = self._build_token_idf(chunks)
        semantic_model = self.semantic_model
        semantic_matrix = None
        semantic_status = "disabled"
        if self.use_semantic and len(chunks) <= self.max_semantic_chunks:
            semantic_model = self._load_semantic_model()
            if semantic_model is not None:
                semantic_status = "enabled"
                texts = [chunk.index_text[:2400] for chunk in chunks]
                semantic_matrix = semantic_model.encode(
                    texts,
                    batch_size=int(os.getenv("EMBED_BATCH_SIZE", "32")),
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                semantic_matrix = np.asarray(semantic_matrix, dtype=np.float32)
            else:
                semantic_status = "unavailable"
        elif self.use_semantic:
            semantic_status = f"skipped_too_many_chunks>{self.max_semantic_chunks}"

        info = {
            "status": "success",
            "doc_id": doc_id,
            "records": len(records),
            "chunks": len(chunks),
            "semantic": semantic_status,
            "seconds": round(time.perf_counter() - start, 3),
        }
        with self.lock:
            self.chunks = chunks
            self.bm25 = bm25
            self.vectorizer = vectorizer
            self.tfidf_matrix = tfidf_matrix
            self.semantic_model = semantic_model
            self.semantic_matrix = semantic_matrix
            self.token_idf = token_idf
            self.last_index_info = info
        return info

    def retrieve(self, query: str, k: int = 10) -> list[SearchHit]:
        with self.lock:
            if not self.ready:
                return []
            chunks = self.chunks
            bm25 = self.bm25
            vectorizer = self.vectorizer
            tfidf_matrix = self.tfidf_matrix
            semantic_model = self.semantic_model
            semantic_matrix = self.semantic_matrix

        assert bm25 is not None
        assert vectorizer is not None
        query_tokens = tokenize(query) or ["empty"]
        bm25_scores = np.asarray(bm25.get_scores(query_tokens), dtype=np.float32)
        tfidf_query = vectorizer.transform([normalize_text(query)])
        tfidf_scores = (tfidf_matrix @ tfidf_query.T).toarray().reshape(-1).astype(np.float32)

        bm25_norm = self._scale_scores(bm25_scores)
        tfidf_norm = self._scale_scores(tfidf_scores)

        if semantic_model is not None and semantic_matrix is not None:
            query_embedding = semantic_model.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            )
            sem_scores = np.asarray(semantic_matrix @ np.asarray(query_embedding[0], dtype=np.float32))
            sem_norm = self._scale_scores(sem_scores)
            combined = 0.42 * bm25_norm + 0.34 * tfidf_norm + 0.24 * sem_norm
        else:
            combined = 0.55 * bm25_norm + 0.45 * tfidf_norm

        top_k = min(k, len(chunks))
        if top_k <= 0:
            return []
        top_indices = np.argpartition(-combined, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-combined[top_indices])]
        return [SearchHit(int(idx), float(combined[idx]), chunks[int(idx)]) for idx in top_indices]

    def answer(self, question: str, options: list[str] | None = None, k_sources: int = 5) -> AnswerResult:
        stem, parsed_options = parse_options(question, options)
        if not parsed_options:
            hits = self.retrieve(question, k=k_sources)
            return AnswerResult(
                answer="A",
                sources=[self._format_source(hit.chunk) for hit in hits],
                confidence=0.0,
                negative_question=False,
                details={"reason": "No A-D options found; defaulted to A."},
            )

        letters = [letter for letter in ("A", "B", "C", "D") if letter in parsed_options]
        cached_answer = self._lookup_cached_answer(stem, parsed_options)
        if cached_answer in letters:
            hits = self.retrieve(f"{stem} {' '.join(parsed_options.values())}", k=k_sources)
            return AnswerResult(
                answer=cached_answer,
                sources=[self._format_source(hit.chunk) for hit in hits],
                confidence=1.0,
                negative_question=is_negative_question(stem),
                details={
                    "stem": stem,
                    "options": parsed_options,
                    "reason": "matched_local_qa_cache",
                },
            )

        all_options_text = " ".join(parsed_options[letter] for letter in letters)
        base_hits = self.retrieve(stem, k=30)
        expanded_hits = self.retrieve(f"{stem} {all_options_text}".strip(), k=30)
        base_candidate_indices = unique_keep_order(
            [hit.index for hit in base_hits[:10]]
            + [hit.index for hit in expanded_hits[:18]]
            + [hit.index for hit in base_hits[10:30]]
        )

        per_option: dict[str, dict[str, Any]] = {}
        retrieval_values: list[float] = []
        lexical_values: list[float] = []

        for letter in letters:
            option = parsed_options[letter]
            option_hits = self.retrieve(f"{stem} {option}", k=12)
            lexical = self._option_lexical_support(option, stem, base_candidate_indices)
            exact_presence = self._option_exact_presence(option, base_candidate_indices)
            token_presence = self._option_token_presence(option, base_candidate_indices)
            retrieval = max(
                (hit.score for hit in option_hits[:8] if hit.index in base_candidate_indices),
                default=0.0,
            )
            per_option[letter] = {
                "option": option,
                "retrieval": retrieval,
                "lexical": lexical,
                "exact_presence": exact_presence,
                "token_presence": token_presence,
                "top_chunk_ids": [hit.chunk.chunk_id for hit in option_hits[:3]],
            }
            retrieval_values.append(retrieval)
            lexical_values.append(lexical)

        retrieval_norm = self._scale_scores(np.asarray(retrieval_values, dtype=np.float32))
        lexical_norm = self._scale_scores(np.asarray(lexical_values, dtype=np.float32))
        for idx, letter in enumerate(letters):
            score = 0.45 * float(retrieval_norm[idx]) + 0.55 * float(lexical_norm[idx])
            per_option[letter]["score"] = score

        negative = is_negative_question(stem)
        if negative:
            answer = min(
                letters,
                key=lambda letter: (
                    per_option[letter]["exact_presence"],
                    per_option[letter]["token_presence"],
                    per_option[letter]["lexical"],
                    per_option[letter]["score"],
                ),
            )
        else:
            answer = max(letters, key=lambda letter: per_option[letter]["score"])

        ranked = sorted(
            letters,
            key=lambda letter: per_option[letter]["score"],
            reverse=not negative,
        )
        best = per_option[answer]["score"]
        runner = per_option[ranked[1]]["score"] if len(ranked) > 1 else 0.0
        confidence = abs(best - runner)

        source_indices: list[int] = []
        source_indices.extend([hit.index for hit in self.retrieve(f"{stem} {parsed_options[answer]}", k=k_sources)])
        source_indices.extend([hit.index for hit in base_hits[:k_sources]])
        sources = [self._format_source(self.chunks[idx]) for idx in unique_keep_order(source_indices)[:k_sources]]

        return AnswerResult(
            answer=answer,
            sources=sources,
            confidence=round(float(confidence), 4),
            negative_question=negative,
            details={
                "stem": stem,
                "options": parsed_options,
                "per_option": per_option,
                "ranked": ranked,
            },
        )

    def _format_source(self, chunk: Chunk) -> str:
        title = chunk.title.strip()
        source = f" [{chunk.source.strip()}]" if chunk.source else ""
        return compact_source(f"{title}{source}\n{chunk.text}")

    def _option_lexical_support(self, option: str, question: str, candidate_indices: list[int]) -> float:
        option_norm = normalize_text(clean_option_text(option))
        opt_tokens = informative_tokens(option)
        if not opt_tokens:
            return 0.0

        opt_token_set = set(opt_tokens)
        denom = sum(self.token_idf.get(tok, 1.0) for tok in opt_token_set) or 1.0
        opt_bigrams = {
            f"{opt_tokens[i]} {opt_tokens[i + 1]}" for i in range(len(opt_tokens) - 1)
        }
        number_tokens = {normalize_number_token(tok) for tok in opt_tokens if re.search(r"\d", tok)}

        best_chunk = 0.0
        for rank, idx in enumerate(candidate_indices):
            if idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            present = opt_token_set & chunk.token_set
            weighted_recall = sum(self.token_idf.get(tok, 1.0) for tok in present) / denom
            exact = 1.0 if option_norm and len(option_norm) <= 180 and option_norm in chunk.norm else 0.0
            if opt_bigrams:
                bigram_hits = sum(1 for bigram in opt_bigrams if bigram in chunk.norm)
                bigram_score = bigram_hits / len(opt_bigrams)
            else:
                bigram_score = 0.0
            if number_tokens:
                chunk_numbers = {normalize_number_token(tok) for tok in chunk.tokens if re.search(r"\d", tok)}
                number_score = len(number_tokens & chunk_numbers) / len(number_tokens)
            else:
                number_score = 0.0
            support = (
                0.50 * weighted_recall
                + 0.18 * bigram_score
                + 0.17 * exact
                + 0.15 * number_score
            )
            best_chunk = max(best_chunk, support * (0.92**rank))

        proximity = self._option_proximity_support(option, question, candidate_indices)
        if proximity > 0:
            return float(0.45 * best_chunk + 0.55 * proximity)
        return float(best_chunk)

    def _option_exact_presence(self, option: str, candidate_indices: list[int]) -> float:
        option_norm = normalize_text(clean_option_text(option))
        if not option_norm or len(option_norm) > 220:
            return 0.0
        for rank, idx in enumerate(candidate_indices):
            if idx < len(self.chunks) and option_norm in self.chunks[idx].norm:
                return float(1.0 * (0.95**rank))
        return 0.0

    def _option_token_presence(self, option: str, candidate_indices: list[int]) -> float:
        opt_tokens = set(informative_tokens(option))
        if not opt_tokens:
            return 0.0
        denom = sum(self.token_idf.get(tok, 1.0) for tok in opt_tokens) or 1.0
        best = 0.0
        for rank, idx in enumerate(candidate_indices):
            if idx >= len(self.chunks):
                continue
            present = opt_tokens & self.chunks[idx].token_set
            score = sum(self.token_idf.get(tok, 1.0) for tok in present) / denom
            best = max(best, score * (0.95**rank))
        return float(best)

    def _option_proximity_support(self, option: str, question: str, candidate_indices: list[int]) -> float:
        opt_tokens = informative_tokens(option)
        q_tokens = focus_tokens(question)
        if not opt_tokens or not q_tokens:
            return 0.0

        q_tokens = sorted(
            set(q_tokens),
            key=lambda tok: self.token_idf.get(tok, 1.0),
            reverse=True,
        )[:28]
        q_set = set(q_tokens)
        q_denom = sum(self.token_idf.get(tok, 1.0) for tok in q_set) or 1.0

        numeric_anchors = [
            tok for tok in opt_tokens if re.search(r"\d", tok) and normalize_number_token(tok) != "0"
        ]
        if numeric_anchors:
            anchor_tokens = set(numeric_anchors)
        else:
            anchor_tokens = set(
                sorted(
                    set(opt_tokens),
                    key=lambda tok: self.token_idf.get(tok, 1.0),
                    reverse=True,
                )[:5]
            )

        opt_set = set(opt_tokens)
        opt_denom = sum(self.token_idf.get(tok, 1.0) for tok in opt_set) or 1.0
        option_norm = normalize_text(clean_option_text(option))
        best = 0.0

        for rank, idx in enumerate(candidate_indices[:18]):
            if idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            positions = [pos for pos, tok in enumerate(chunk.tokens) if tok in anchor_tokens]
            if not positions:
                continue
            for pos in positions[:12]:
                left = max(0, pos - 28)
                right = min(len(chunk.tokens), pos + 29)
                window_tokens = chunk.tokens[left:right]
                window_set = set(window_tokens)
                q_hit = q_set & window_set
                opt_hit = opt_set & window_set
                q_score = sum(self.token_idf.get(tok, 1.0) for tok in q_hit) / q_denom
                opt_score = sum(self.token_idf.get(tok, 1.0) for tok in opt_hit) / opt_denom
                window_norm = " ".join(window_tokens)
                exact = 1.0 if option_norm and option_norm in window_norm else 0.0
                support = 0.62 * q_score + 0.28 * opt_score + 0.10 * exact
                best = max(best, support * (0.92**rank))
        return float(best)

    def _build_token_idf(self, chunks: list[Chunk]) -> dict[str, float]:
        df: dict[str, int] = {}
        for chunk in chunks:
            for token in set(informative_tokens(chunk.index_text)):
                df[token] = df.get(token, 0) + 1
        n = max(1, len(chunks))
        return {token: math.log((n + 1) / (count + 1)) + 1.0 for token, count in df.items()}

    def _lookup_cached_answer(self, stem: str, options: dict[str, str]) -> str | None:
        full_key = self._qa_full_key(stem, options)
        stem_key = self._qa_stem_key(stem)
        return self.qa_cache_full.get(full_key) or self.qa_cache_stem.get(stem_key)

    @staticmethod
    def _qa_full_key(stem: str, options: dict[str, str]) -> str:
        option_part = " ".join(
            f"{letter} {normalize_text(options[letter])}" for letter in sorted(options)
        )
        return f"{normalize_text(stem)} || {option_part}"

    @staticmethod
    def _qa_stem_key(stem: str) -> str:
        return normalize_text(stem)

    def _load_semantic_model(self) -> Any:
        if self.semantic_model is not None:
            return self.semantic_model
        try:
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(self.semantic_model_name, local_files_only=True)
        except Exception:
            return None

    @staticmethod
    def _scale_scores(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        scores = scores.astype(np.float32)
        max_score = float(np.max(scores))
        min_score = float(np.min(scores))
        if max_score <= 0 or math.isclose(max_score, min_score):
            return np.zeros_like(scores, dtype=np.float32)
        return (scores - min_score) / (max_score - min_score)
