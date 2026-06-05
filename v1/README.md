# Student RAG Server - v1 TF-IDF

Variant chắc chắn nhất: không cần tải embedding model, chạy offline bằng TF-IDF.

## Cài đặt

```bash
uv venv
uv pip install -r requirements.txt
cp .env.example .env
```

Sửa `.env`: `STUDENT_ID`, `STUDENT_SERVER_URL`, nếu cần thì sửa `TEACHER_BASE_URL`.

## Chạy

```bash
uv run python main.py
```

Terminal khác:

```bash
uv run python client.py register
uv run python client.py evaluate
uv run python client.py result
```

Endpoint Student bắt buộc: `POST /upload`, `POST /ask`. Response `/ask` luôn trả `answer` là `A/B/C/D`.
