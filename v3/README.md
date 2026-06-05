# Student RAG Server - v3 Multilingual MiniLM

Variant embedding local nhe-vua, hop tieng Viet hon `v2`, nen thu neu da cache model.

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

Neu da cache model, day la ban nen so diem voi `v1` vi retrieval tieng Viet thuong tot hon TF-IDF o cac cau hoi dien dat khac tai lieu.
