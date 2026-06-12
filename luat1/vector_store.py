"""Persistent ChromaDB store with a precomputed legal BM25 index."""
from __future__ import annotations

import logging
from collections import Counter

import chromadb
import numpy as np

import config
from rag import bm25_scores, normalize_scores, normalize_search_text, rerank_results, tokenize_text

log = logging.getLogger("student-server")


class LegalStore:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
        self.collection = self._get_collection()
        self.chunks: list[str] = []
        self.embeddings: np.ndarray | None = None
        self.term_freqs: list[Counter[str]] = []
        self.normalized_chunks: list[str] = []
        self.doc_freq: Counter[str] = Counter()
        self.avg_doc_len = 0.0
        self._load()

    def _get_collection(self):
        return self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def _rebuild_lexical_index(self) -> None:
        self.term_freqs = [Counter(tokenize_text(chunk)) for chunk in self.chunks]
        self.normalized_chunks = [normalize_search_text(chunk) for chunk in self.chunks]
        self.doc_freq = Counter()
        for term_freq in self.term_freqs:
            self.doc_freq.update(term_freq.keys())
        self.avg_doc_len = sum(sum(freq.values()) for freq in self.term_freqs) / max(len(self.term_freqs), 1)

    def _load(self) -> None:
        data = self.collection.get(include=["documents", "embeddings"])
        self.chunks = list(data.get("documents") or [])
        raw_embeddings = data.get("embeddings")
        if raw_embeddings is None or len(raw_embeddings) == 0:
            self.embeddings = None
        else:
            embeddings = np.asarray(raw_embeddings, dtype=np.float32)
            self.embeddings = embeddings if embeddings.ndim == 2 and embeddings.shape[1] > 1 else None
        self._rebuild_lexical_index()
        if self.chunks:
            log.info("Loaded %d legal chunks from ChromaDB at %s", len(self.chunks), config.CHROMA_DB_DIR)

    def reset(self) -> None:
        self.client.delete_collection(config.CHROMA_COLLECTION)
        self.collection = self._get_collection()
        self.chunks = []
        self.embeddings = None
        self._rebuild_lexical_index()

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
            normalized = np.ones((len(chunks), 1), dtype=np.float32)

        offset = len(self.chunks)
        for start in range(0, len(chunks), config.CHROMA_BATCH_SIZE):
            end = min(start + config.CHROMA_BATCH_SIZE, len(chunks))
            self.collection.upsert(
                ids=[f"legal-{offset + index:08d}" for index in range(start, end)],
                documents=chunks[start:end],
                embeddings=normalized[start:end].tolist(),
            )
        self.chunks.extend(chunks)
        if normalized.shape[1] > 1:
            self.embeddings = normalized if self.embeddings is None else np.vstack([self.embeddings, normalized])
        self._rebuild_lexical_index()
        log.info("Persisted %d legal chunks to ChromaDB at %s", len(chunks), config.CHROMA_DB_DIR)

    def _bm25_search(self, question: str, top_k: int) -> list[tuple[int, float]]:
        return bm25_scores(
            question,
            self.chunks,
            self.term_freqs,
            self.doc_freq,
            self.avg_doc_len,
            self.normalized_chunks,
        )[:top_k]

    def _vector_search(self, query_emb: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self.embeddings is None or not self.chunks:
            return []
        query = query_emb.astype(np.float32).reshape(-1)
        if query.shape[0] != self.embeddings.shape[1]:
            raise ValueError("Embedding model changed. Rebuild ChromaDB by evaluating with document_received=false.")
        norm = np.linalg.norm(query)
        if norm > 0:
            query /= norm
        scores = self.embeddings @ query
        k = min(top_k, len(self.chunks))
        indexes = np.argpartition(-scores, k - 1)[:k]
        indexes = indexes[np.argsort(-scores[indexes])]
        return [(int(index), float(scores[index])) for index in indexes]

    def search(self, query, top_k: int = 5) -> list[tuple[str, float]]:
        if not self.chunks:
            return []
        backend = config.RETRIEVER_BACKEND
        candidate_k = max(top_k, config.VECTOR_CANDIDATES)

        if backend in {"bm25", "tfidf"}:
            results = [(self.chunks[index], score) for index, score in self._bm25_search(str(query), candidate_k)]
            return rerank_results(str(query), results, top_k)
        if backend in {"sbert", "openai", "vector"}:
            results = [(self.chunks[index], score) for index, score in self._vector_search(query, candidate_k)]
            return rerank_results("", results, top_k)
        if backend != "hybrid":
            raise RuntimeError(f"Unknown RETRIEVER_BACKEND={backend!r}")
        if not isinstance(query, tuple) or len(query) != 2:
            raise TypeError("LegalStore.search expects (question, query_embedding) in hybrid mode")

        question, query_emb = query
        vector = dict(self._vector_search(query_emb, candidate_k))
        lexical = dict(self._bm25_search(question, candidate_k))
        vector_norm = normalize_scores(vector, higher_is_better=True)
        lexical_norm = normalize_scores(lexical, higher_is_better=True)
        lexical_weight = min(max(config.BM25_WEIGHT, 0.0), 1.0)
        ranked = [
            (
                index,
                (1.0 - lexical_weight) * vector_norm.get(index, 0.0)
                + lexical_weight * lexical_norm.get(index, 0.0),
            )
            for index in set(vector_norm) | set(lexical_norm)
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        results = [(self.chunks[index], score) for index, score in ranked[:candidate_k]]
        return rerank_results(question, results, top_k)

    def __len__(self) -> int:
        return len(self.chunks)


store = LegalStore()
