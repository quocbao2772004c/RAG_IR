# Student RAG Server - v1 TF-IDF

Variant chac chan nhat: khong can tai embedding model, chay offline bang TF-IDF.

## Cai dat nhanh bang uv

```bash
uv sync
cp .env.example .env
```

Neu may chua co `uv`: `pip install uv`.

Fallback bang pip:

```bash
uv venv
uv pip install -r requirements.txt
cp .env.example .env
```

Sua `.env`: `STUDENT_ID`, `STUDENT_SERVER_URL`, neu can thi sua `TEACHER_BASE_URL`.

## Chay

```bash
uv run python main.py
```

Terminal khac:

```bash
uv run python client.py register
uv run python client.py evaluate
uv run python client.py result
```

Endpoint Student bat buoc: `POST /upload`, `POST /ask`. Response `/ask` luon tra `answer` la `A/B/C/D`.
