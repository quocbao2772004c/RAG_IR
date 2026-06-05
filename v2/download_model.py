"""Download the local embedding model before entering the offline LAN."""
import os

from sentence_transformers import SentenceTransformer


if __name__ == "__main__":
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    print(f"Downloading/caching: {model_name}")
    SentenceTransformer(model_name)
    print("Done. This variant can now load the embedding model from local cache.")
