# Student RAG Server - v1 Local Ensemble

V1 la ban remap tu folder cu `IR/final/v5`: lexical ensemble + local semantic
reranker, chay offline/local, khong bat QA cache mac dinh.

Huong dan setup va xu ly loi Windows: [WINDOWS_SETUP.md](../WINDOWS_SETUP.md).

## Diem chinh

- Mac dinh `RAG_VERSION=v5` du folder ten la `v1`.
- Hybrid retrieval BM25 + TF-IDF.
- Semantic reranker `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Khi Teacher goi `/upload`, server build index moi tu document Teacher gui.
- Khong dung answer cache mac dinh: `RAG_USE_QA_CACHE=0`.

## Cai dat

Linux/macOS:

```bash
uv sync
uv run python download_model.py
cp .env.example .env
```

Windows PowerShell:

```powershell
uv sync
uv run python download_model.py
Copy-Item .env.example .env
```

Sua `.env`: `STUDENT_ID`, `STUDENT_SERVER_URL`, `TEACHER_BASE_URL` neu can.

## Chay

```bash
uv run python main.py
```

Terminal khac:

```bash
uv run python client.py register
uv run python client.py evaluate --document-received false
uv run python client.py evaluate --document-received true
uv run python client.py result
```

Lan dau dung `evaluate --document-received false` de Teacher upload tai lieu.
Sau khi index da co trong `rag_state`, dung `evaluate --document-received true`.
