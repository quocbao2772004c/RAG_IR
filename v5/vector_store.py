"""Persistent ChromaDB store with vector, BM25, and hybrid retrieval modes."""
from __future__ import annotations

import logging

import chromadb
import numpy as np

import config
from rag import bm25_rank, normalize_scores, rerank_results

log = logging.getLogger("student-server")


class HybridStore:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
        self.collection = self._get_collection()
        self.chunks: list[str] = []
        self.embeddings: np.ndarray | None = None
        self._load()

    def _get_collection(self):
        return self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def _load(self) -> None:
        data = self.collection.get(include=["documents", "embeddings"])
        self.chunks = list(data.get("documents") or [])
        raw_embeddings = data.get("embeddings")
        if raw_embeddings is None or len(raw_embeddings) == 0:
            self.embeddings = None
        else:
            embeddings = np.asarray(raw_embeddings, dtype=np.float32)
            self.embeddings = embeddings if embeddings.ndim == 2 and embeddings.shape[1] > 1 else None
        if self.chunks:
            log.info("Loaded %d chunks from ChromaDB at %s", len(self.chunks), config.CHROMA_DB_DIR)

    def reset(self) -> None:
        self.client.delete_collection(config.CHROMA_COLLECTION)
        self.collection = self._get_collection()
        self.chunks = []
        self.embeddings = None

    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (embeddings / norms).astype(np.float32)

    def add(self, chunks: list[str], embeddings: np.ndarray | None = None) -> None:
        if not chunks:
            return
        if embeddings is not None and embeddings.size > 0:
            if len(chunks) != len(embeddings):
                raise ValueError("chunks and embeddings must have the same length")
            normalized = self._normalize_embeddings(embeddings)
        else:
            # Persist BM25-only documents without asking Chroma to create embeddings.
            normalized = np.ones((len(chunks), 1), dtype=np.float32)

        offset = len(self.chunks)
        batch_size = max(1, config.CHROMA_BATCH_SIZE)
        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            self.collection.upsert(
                ids=[f"chunk-{offset + index:08d}" for index in range(start, end)],
                documents=chunks[start:end],
                embeddings=normalized[start:end].tolist(),
            )

        self.chunks.extend(chunks)
        if normalized.shape[1] > 1:
            self.embeddings = normalized if self.embeddings is None else np.vstack([self.embeddings, normalized])
        log.info("Persisted %d chunks to ChromaDB at %s", len(chunks), config.CHROMA_DB_DIR)

    def _vector_search(self, query_emb: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self.embeddings is None or not self.chunks:
            return []
        q = query_emb.astype(np.float32).reshape(-1)
        if q.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                "Persisted embedding dimension does not match the configured model. "
                "Run evaluate without --document-received to rebuild ChromaDB."
            )
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
            return rerank_results(str(query), results, top_k)

        if backend in {"sbert", "openai", "vector"}:
            results = [(self.chunks[i], score) for i, score in self._vector_search(query, top_k)]
            return rerank_results("", results, top_k)

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
        return rerank_results(question, results, top_k)

    def __len__(self) -> int:
        return len(self.chunks)


store = HybridStore()
