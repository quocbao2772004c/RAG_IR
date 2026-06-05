# Student RAG Server - v1 TF-IDF

Variant chắc chắn nhất: không cần tải embedding model, chạy offline bằng TF-IDF.

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Sửa `.env`: `STUDENT_ID`, `STUDENT_SERVER_URL`, nếu cần thì sửa `TEACHER_BASE_URL`.

## Chạy

```bash
python main.py
```

Terminal khác:

```bash
python client.py register
python client.py evaluate
python client.py result
```

Endpoint Student bắt buộc: `POST /upload`, `POST /ask`. Response `/ask` luôn trả `answer` là `A/B/C/D`.
