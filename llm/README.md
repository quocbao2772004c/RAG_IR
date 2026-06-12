# LLM Chat and PDF RAG

Thu muc nay dung de chat voi OpenAI-compatible LLM va thu nghiem RAG tren PDF.

Huong dan setup va xu ly loi Windows: [WINDOWS_SETUP.md](../WINDOWS_SETUP.md).

## Format

- `chat_cli.py`: chat voi LLM trong terminal.
- `chat_streamlit.py`: giao dien chat bang Streamlit.
- `rag.py`: ingest PDF vao ChromaDB, query context, answer/chat bang RAG.
- `.env.example`: cau hinh mau, copy thanh `.env` truoc khi chay.

## Cai dat

Linux/macOS:

```bash
uv sync
cp .env.example .env
```

Windows PowerShell:

```powershell
uv sync
Copy-Item .env.example .env
```

Sua `.env`: `STUDENT_ID`, `TEACHER_BASE_URL`, `LLM_BASE_URL`, `LLM_MODEL` neu can.
Mac dinh `LLM_BASE_URL=https://mba.ptit.edu.vn/v1`, `LLM_MODEL=Qwen/Qwen3-4B-AWQ`,
`LLM_API_KEY=EMPTY`.

## Chat LLM

Chay chat trong terminal:

```bash
uv run python chat_cli.py
```

Gui mot cau roi thoat:

```bash
uv run python chat_cli.py -m "Xin chao"
```

Mo giao dien web:

```bash
uv run streamlit run chat_streamlit.py
```

## RAG voi PDF

Tao index tu PDF:

```bash
uv run python rag.py --pdf /duong/dan/tai-lieu.pdf --db-dir chroma_db ingest
```

Kiem tra index:

```bash
uv run python rag.py --db-dir chroma_db info
```

Tim cac doan lien quan, chua goi LLM:

```bash
uv run python rag.py --db-dir chroma_db query "quy dinh hoc phi la gi?"
```

Tra loi bang RAG co goi LLM:

```bash
uv run python rag.py --db-dir chroma_db answer "quy dinh hoc phi la gi?"
```

Chat RAG tuong tac:

```bash
uv run python rag.py --db-dir chroma_db chat
```

Neu muon xem context da truy xuat:

```bash
uv run python rag.py --db-dir chroma_db answer "cau hoi..." --show-context
```

## Lenh huu ich

Kiem tra bien moi truong:

```bash
uv run python rag.py check-env
```

Xoa ChromaDB local:

```bash
uv run python rag.py --db-dir chroma_db reset
```
