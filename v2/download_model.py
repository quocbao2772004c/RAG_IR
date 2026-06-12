from __future__ import annotations

import os

from sentence_transformers import SentenceTransformer


MODEL_NAME = os.getenv(
    "RAG_RERANK_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)


def main() -> None:
    SentenceTransformer(MODEL_NAME)
    print(f"Downloaded/cached model: {MODEL_NAME}")


if __name__ == "__main__":
    main()
