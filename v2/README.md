# Student RAG Server - v2 all-MiniLM-L6-v2

Variant embedding local rất nhẹ, nhanh, nhưng thiên tiếng Anh hơn tiếng Việt.

## Chuẩn bị khi còn Internet

```bash
uv venv
uv pip install -r requirements.txt
uv run python download_model.py
cp .env.example .env
```

Sửa `.env`: `STUDENT_ID`, `STUDENT_SERVER_URL`, nếu cần thì sửa `TEACHER_BASE_URL`.

## Chạy trong LAN

```bash
uv run python main.py
```

Terminal khác:

```bash
uv run python client.py register
uv run python client.py evaluate
uv run python client.py result
```

Nếu model chưa nằm trong cache local, bản này có thể lỗi khi đã bị ngắt Internet. Khi đó chuyển ngay sang `v1`.
