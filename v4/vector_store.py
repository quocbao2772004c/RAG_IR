"""v4 in-memory store with vector, BM25, and hybrid retrieval modes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import config
from rag import bm25_rank, normalize_scores, rerank_results


@dataclass
class HybridStore:
    chunks: list[str] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.chunks = []
        self.embeddings = None

    def add(self, chunks: list[str], embeddings: np.ndarray | None = None) -> None:
        self.chunks.extend(chunks)
        if embeddings is None or embeddings.size == 0:
            return
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = (embeddings / norms).astype(np.float32)
        self.embeddings = embs if self.embeddings is None else np.vstack([self.embeddings, embs])

    def _vector_search(self, query_emb: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self.embeddings is None or not self.chunks:
            return []
        q = query_emb.astype(np.float32).reshape(-1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        scores = self.embeddings @ q
        k = min(top_k, len(self.chunks))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]

    def search(self, query, top_k: int = 5) -> list[tuple[str, float]]:
        backend = config.RETRIEVER_BACKEND
        if not self.chunks:
            return []

        if backend in {"bm25", "tfidf"}:
            results = [(self.chunks[i], score) for i, score in bm25_rank(str(query), self.chunks, top_k)]
            return rerank_results(str(query), results)

        if backend in {"sbert", "openai", "vector"}:
            results = [(self.chunks[i], score) for i, score in self._vector_search(query, top_k)]
            return rerank_results("", results)

        if backend != "hybrid":
            raise RuntimeError(f"Unknown RETRIEVER_BACKEND={backend!r}")

        if not isinstance(query, tuple) or len(query) != 2:
            raise TypeError("HybridStore.search expects (question, query_embedding)")
        question, query_emb = query
        candidate_k = max(top_k, config.VECTOR_CANDIDATES)
        if config.RERANKER_MODEL:
            candidate_k = max(candidate_k, config.RERANK_CANDIDATES)

        vector_scores = dict(self._vector_search(query_emb, candidate_k))
        bm25_scores = dict(bm25_rank(question, self.chunks, candidate_k))
        vector_norm = normalize_scores(vector_scores, higher_is_better=True)
        bm25_norm = normalize_scores(bm25_scores, higher_is_better=True)

        bm25_weight = min(max(config.BM25_WEIGHT, 0.0), 1.0)
        vector_weight = 1.0 - bm25_weight
        keys = set(vector_norm) | set(bm25_norm)
        ranked = [
            (index, vector_weight * vector_norm.get(index, 0.0) + bm25_weight * bm25_norm.get(index, 0.0))
            for index in keys
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        results = [(self.chunks[index], score) for index, score in ranked[:candidate_k]]
        return rerank_results(question, results)[:top_k]

    def __len__(self) -> int:
        return len(self.chunks)


store = HybridStore()
