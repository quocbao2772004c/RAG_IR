"""Vector store with two backends:
- tfidf : small pure-Python TF-IDF (default, offline-friendly)
- sbert : dense vectors with cosine similarity
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Optional

import numpy as np

import config


# ----------------- Dense (sbert) backend -----------------

@dataclass
class DenseStore:
    chunks: list[str] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None  # (N, D) L2-normalized

    def reset(self) -> None:
        self.chunks = []
        self.embeddings = None

    def add(self, chunks: list[str], embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = (embeddings / norms).astype(np.float32)
        self.chunks.extend(chunks)
        self.embeddings = embs if self.embeddings is None else np.vstack([self.embeddings, embs])

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        if self.embeddings is None or not self.chunks:
            return []
        q = query_emb.astype(np.float32).reshape(-1)
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        scores = self.embeddings @ q
        k = min(top_k, len(self.chunks))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.chunks[i], float(scores[i])) for i in idx]

    def __len__(self) -> int:
        return len(self.chunks)


# ----------------- TF-IDF backend -----------------

class TfidfStore:
    def __init__(self) -> None:
        self.chunks: list[str] = []
        self._doc_vecs: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}

    @staticmethod
    def _tokens(text: str) -> list[str]:
        words = re.findall(r"[\wÀ-ỹ]+", (text or "").lower(), flags=re.UNICODE)
        bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
        return words + bigrams

    @staticmethod
    def _normalize(vec: dict[str, float]) -> dict[str, float]:
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm == 0:
            return vec
        return {k: v / norm for k, v in vec.items()}

    def reset(self) -> None:
        self.chunks = []
        self._doc_vecs = []
        self._idf = {}

    def add(self, chunks: list[str], embeddings=None) -> None:
        self.chunks.extend(chunks)
        tokenized = [self._tokens(c) for c in self.chunks]
        n_docs = max(1, len(tokenized))
        df: dict[str, int] = {}
        for toks in tokenized:
            for tok in set(toks):
                df[tok] = df.get(tok, 0) + 1
        self._idf = {tok: math.log((1 + n_docs) / (1 + count)) + 1.0 for tok, count in df.items()}
        self._doc_vecs = []
        for toks in tokenized:
            tf: dict[str, float] = {}
            for tok in toks:
                tf[tok] = tf.get(tok, 0.0) + 1.0
            vec = {tok: (1.0 + math.log(freq)) * self._idf.get(tok, 1.0) for tok, freq in tf.items()}
            self._doc_vecs.append(self._normalize(vec))

    def search(self, query, top_k: int = 5) -> list[tuple[str, float]]:
        # `query` here is the question string (not a vector)
        if not self._doc_vecs or not self.chunks:
            return []
        if not isinstance(query, str):
            raise TypeError("TfidfStore.search expects a string query")
        tf: dict[str, float] = {}
        for tok in self._tokens(query):
            tf[tok] = tf.get(tok, 0.0) + 1.0
        q_vec = self._normalize(
            {tok: (1.0 + math.log(freq)) * self._idf.get(tok, 1.0) for tok, freq in tf.items()}
        )
        scores = np.array(
            [sum(weight * doc.get(tok, 0.0) for tok, weight in q_vec.items()) for doc in self._doc_vecs],
            dtype=np.float32,
        )
        k = min(top_k, len(self.chunks))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.chunks[i], float(scores[i])) for i in idx]

    def __len__(self) -> int:
        return len(self.chunks)


# ----------------- Factory singleton -----------------

def _build_store():
    backend = config.RETRIEVER_BACKEND
    if backend == "tfidf":
        return TfidfStore()
    if backend in ("sbert", "openai"):
        return DenseStore()
    raise RuntimeError(f"Unknown RETRIEVER_BACKEND={backend!r}")


store = _build_store()
