# RUN

Các lệnh dưới đây giả sử đang ở thư mục repo:

```bash
cd /home/anonymous/code/IR/test_bai_thi
```

Nếu vừa clone từ GitHub:

```bash
git clone <LINK_GITHUB_CUA_BAN>
cd test_bai_thi
```

Lấy IP LAN của máy:

```bash
ip addr
```

Tìm IP dạng `192.168.x.x` hoặc `10.x.x.x`, rồi điền vào `STUDENT_SERVER_URL`.

## v1 - chắc nhất, không cần embedding model

### Cài lần đầu

```bash
cd /home/anonymous/code/IR/test_bai_thi/v1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Sửa `.env`:

```bash
nano .env
```

Điền tối thiểu:

```env
STUDENT_ID=MSSV_CUA_BAN
STUDENT_SERVER_URL=http://IP_LAN_CUA_MAY:5000
PORT=5000
RETRIEVER_BACKEND=tfidf
```

### Chạy server

Terminal 1:

```bash
cd /home/anonymous/code/IR/test_bai_thi/v1
source .venv/bin/activate
python main.py
```

Terminal 2:

```bash
cd /home/anonymous/code/IR/test_bai_thi/v1
source .venv/bin/activate
python client.py register
python client.py evaluate
python client.py result
```

Nếu cần làm lại:

```bash
python client.py reset
python client.py register
python client.py evaluate
python client.py result
```

## v2 - embedding local nhẹ

Model: `sentence-transformers/all-MiniLM-L6-v2`.

### Cài và tải model khi còn Internet

```bash
cd /home/anonymous/code/IR/test_bai_thi/v2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_model.py
cp .env.example .env
```

Sửa `.env`:

```bash
nano .env
```

Điền tối thiểu:

```env
STUDENT_ID=MSSV_CUA_BAN
STUDENT_SERVER_URL=http://IP_LAN_CUA_MAY:5001
PORT=5001
RETRIEVER_BACKEND=sbert
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Chạy server

Terminal 1:

```bash
cd /home/anonymous/code/IR/test_bai_thi/v2
source .venv/bin/activate
python main.py
```

Terminal 2:

```bash
cd /home/anonymous/code/IR/test_bai_thi/v2
source .venv/bin/activate
python client.py register
python client.py evaluate
python client.py result
```

## v3 - embedding local hợp tiếng Việt hơn

Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

### Cài và tải model khi còn Internet

```bash
cd /home/anonymous/code/IR/test_bai_thi/v3
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_model.py
cp .env.example .env
```

Sửa `.env`:

```bash
nano .env
```

Điền tối thiểu:

```env
STUDENT_ID=MSSV_CUA_BAN
STUDENT_SERVER_URL=http://IP_LAN_CUA_MAY:5002
PORT=5002
RETRIEVER_BACKEND=sbert
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### Chạy server

Terminal 1:

```bash
cd /home/anonymous/code/IR/test_bai_thi/v3
source .venv/bin/activate
python main.py
```

Terminal 2:

```bash
cd /home/anonymous/code/IR/test_bai_thi/v3
source .venv/bin/activate
python client.py register
python client.py evaluate
python client.py result
```

## Đổi version khi đang thi

Dừng server cũ ở Terminal 1 bằng:

```bash
Ctrl+C
```

Chạy server version mới, ví dụ chuyển sang `v3`:

```bash
cd /home/anonymous/code/IR/test_bai_thi/v3
source .venv/bin/activate
python main.py
```

Terminal 2 gọi lại:

```bash
cd /home/anonymous/code/IR/test_bai_thi/v3
source .venv/bin/activate
python client.py register
python client.py evaluate
python client.py result
```

## Test ở nhà

Test API nhanh bằng tay với `v1`.

```bash
cd /home/anonymous/code/IR/test_bai_thi/v1
source .venv/bin/activate
python main.py
```

Terminal khác:

```bash
curl -s -X POST http://127.0.0.1:5000/upload \
  -H 'Content-Type: application/json' \
  -d '{"doc_id":"demo","text":"RAG gồm retrieval và generation. Endpoint gửi tài liệu là /upload."}'

curl -s -X POST http://127.0.0.1:5000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Endpoint gửi tài liệu là gì? A. /ask B. /upload C. /result D. /proxy"}'
```

## Cache câu hỏi và đáp án

Mỗi version tự lưu cache trong thư mục của version đó:

```text
v1/cache/questions_seen.jsonl
v1/cache/answer_cache.jsonl
v2/cache/questions_seen.jsonl
v2/cache/answer_cache.jsonl
v3/cache/questions_seen.jsonl
v3/cache/answer_cache.jsonl
```

Xem cache:

```bash
tail -n 20 cache/questions_seen.jsonl
tail -n 20 cache/answer_cache.jsonl
```

Xóa cache của version hiện tại:

```bash
rm -rf cache
```

## Khuyến nghị lúc thi

Chạy `v1` trước để có điểm nền vì không cần model embedding. Nếu còn thời gian và `v3` đã tải model xong, thử `v3` để kéo điểm. `v2` dùng khi `v3` tải chậm hoặc máy yếu.
