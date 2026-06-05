# Student RAG Server - v2 all-MiniLM-L6-v2

Variant embedding local rat nhe, nhanh, nhung thien tieng Anh hon tieng Viet.

## Chuan bi nhanh bang uv khi con Internet

```bash
uv sync
uv run python download_model.py
cp .env.example .env
```

Neu may chua co `uv`: `pip install uv`.

Fallback bang pip:

```bash
uv venv
uv pip install -r requirements.txt
uv run python download_model.py
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

Neu model chua nam trong cache local, ban nay co the loi khi da bi ngat Internet. Khi do chuyen ngay sang `v1`.
