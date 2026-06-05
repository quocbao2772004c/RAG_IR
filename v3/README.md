# Student RAG Server - v3 Multilingual MiniLM

Variant embedding local nhẹ-vừa, hợp tiếng Việt hơn `v2`, nên thử nếu đã cache model.

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

Nếu đã cache model, đây là bản nên so điểm với `v1` vì retrieval tiếng Việt thường tốt hơn TF-IDF ở các câu hỏi diễn đạt khác tài liệu.
