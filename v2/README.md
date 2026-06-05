# Student RAG Server - v2 all-MiniLM-L6-v2

Variant embedding local rất nhẹ, nhanh, nhưng thiên tiếng Anh hơn tiếng Việt.

## Chuẩn bị khi còn Internet

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_model.py
cp .env.example .env
```

Sửa `.env`: `STUDENT_ID`, `STUDENT_SERVER_URL`, nếu cần thì sửa `TEACHER_BASE_URL`.

## Chạy trong LAN

```bash
python main.py
```

Terminal khác:

```bash
python client.py register
python client.py evaluate
python client.py result
```

Nếu model chưa nằm trong cache local, bản này có thể lỗi khi đã bị ngắt Internet. Khi đó chuyển ngay sang `v1`.
