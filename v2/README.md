# Student RAG Server - v2 Local LLM Fallback

V2 la ban remap tu folder cu `IR/final/v6`: local ensemble nhu v1, them
optional OpenAI-compatible local/private LLM fallback neu cau hinh trong `.env`.

Huong dan setup va xu ly loi Windows: [WINDOWS_SETUP.md](../WINDOWS_SETUP.md).

## Diem chinh

- Mac dinh `RAG_VERSION=v6` du folder ten la `v2`.
- Hybrid retrieval BM25 + TF-IDF.
- Semantic reranker `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Co the dat `RAG_LOCAL_LLM_BASE_URL`, `RAG_LOCAL_LLM_MODEL`,
  `RAG_LOCAL_LLM_API_KEY` de goi model local/private OpenAI-compatible.
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
Neu co LLM local/private, sua them cac bien `RAG_LOCAL_LLM_*`; neu endpoint khong nam tren localhost thi dat `RAG_ALLOW_NONLOCAL_LLM=1`.

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
