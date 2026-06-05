# Student RAG Server - v4 Hybrid

Variant theo format `v1/v2/v3`, nhung retrieval dung hybrid BM25 + dense vector dua tren logic tu `v4/rag.py` ban dau.

## Cai dat nhanh bang uv

```bash
uv sync
uv run python download_model.py
cp .env.example .env
```

Neu may chua co `uv`: `pip install uv`.

Fallback bang pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_model.py
cp .env.example .env
```

Sua `.env`: `STUDENT_ID`, `STUDENT_SERVER_URL`, neu can thi sua `TEACHER_BASE_URL`.

## Chay trong LAN

```bash
uv run python main.py
```

Terminal khac:

```bash
uv run python client.py register
uv run python client.py evaluate
uv run python client.py result
```

Backend mac dinh la `hybrid`. Co the doi:

```env
RETRIEVER_BACKEND=bm25
RETRIEVER_BACKEND=sbert
RETRIEVER_BACKEND=openai
```

Neu muon dung reranker, dien model vao `RERANKER_MODEL`, vi du `BAAI/bge-reranker-v2-m3`. De trong se chay nhanh hon.
