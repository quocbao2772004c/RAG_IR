# Hệ thống RAG pháp luật - `luat1`

`luat1` là Student Server dùng để nhận tài liệu từ Teacher Server, xây dựng cơ sở dữ liệu ChromaDB và trả lời câu hỏi trắc nghiệm pháp luật.

Hướng dẫn cài đặt và xử lý lỗi Windows: [WINDOWS_SETUP.md](../WINDOWS_SETUP.md).

Hệ thống hỗ trợ ba chế độ:

- **RAG thường:** tìm kiếm nội dung liên quan trong tài liệu rồi gọi LLM trả lời.
- **Question bank (`--map`):** ưu tiên tìm đáp án trong ngân hàng câu hỏi có sẵn; nếu không tìm thấy thì quay về RAG thường.
- **RAG rồi question bank (`--map --rag`):** luôn gọi RAG trước, sau đó dùng đáp án từ question bank để ghi đè nếu tìm thấy.

## 1. Cách hoạt động

Student Server cung cấp các endpoint chính:

- `POST /upload`: nhận tài liệu từ Teacher Server, chia tài liệu thành các chunk và lưu vào ChromaDB.
- `POST /ask`: nhận câu hỏi, truy xuất context liên quan và trả về đáp án `A`, `B`, `C` hoặc `D`.
- `POST /mode`: bật hoặc tắt chế độ question bank.
- `GET /`: kiểm tra trạng thái Student Server.

RAG thường sử dụng:

- Chunking theo điều, khoản và điểm của văn bản pháp luật.
- Hybrid retrieval kết hợp BM25 với embedding local đa ngôn ngữ.
- ChromaDB để lưu các chunk lâu dài.
- LLM và verifier để xử lý câu hỏi phủ định, thời hạn, số tiền, cơ quan có thẩm quyền và câu hỏi cần kết hợp nhiều quy định.

## 2. Cài đặt

Mở PowerShell và chạy:

```powershell
cd F:\RAG_IR\luat1
uv sync
```

Lệnh `uv sync` sẽ tạo môi trường Python và cài toàn bộ thư viện trong `pyproject.toml`.

Kiểm tra `uv`:

```powershell
uv --version
```

## 3. Cấu hình `.env`

File cấu hình nằm tại:

```text
F:\RAG_IR\luat1\.env
```

Trước khi thi, bắt buộc sửa:

```env
STUDENT_ID=MSSV_CUA_BAN
STUDENT_SERVER_URL=http://IP_LAN_CUA_MAY:5005
```

Lấy IP LAN trên Windows:

```powershell
ipconfig
```

Tìm địa chỉ `IPv4 Address`, ví dụ `192.168.50.123`, sau đó cấu hình:

```env
STUDENT_ID=B22DCCN123
STUDENT_SERVER_URL=http://192.168.50.123:5005
```

Không dùng `127.0.0.1` trong `STUDENT_SERVER_URL`, vì Teacher Server chạy trên máy khác và không thể truy cập địa chỉ localhost của máy bạn.

Cấu hình LLM và RAG đã benchmark:

```env
LLM_BASE_URL=https://mba.ptit.edu.vn/v1
LLM_MODEL=Qwen/Qwen3-4B-AWQ
LLM_API_KEY=EMPTY
LLM_TEMPERATURE=0.0
LLM_TIMEOUT=60
LLM_MAX_TOKENS=8
LLM_VERIFY=true

RETRIEVER_BACKEND=hybrid
RETRIEVE_WITH_OPTIONS=false
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
MODEL_LOCAL_ONLY=true
BM25_WEIGHT=0.88
TOP_K=8
MAX_CONTEXT_CHARS=8000
```

Không tăng `MAX_CONTEXT_CHARS` quá cao. Model hiện tại có giới hạn context `4096` token; context quá dài có thể làm API trả lỗi và chương trình phải dùng heuristic fallback.

Tải embedding model trước khi vào phòng thi hoặc trước khi mất kết nối Internet:

```powershell
uv run python download_model.py
```

Hybrid dùng embedding local, không gọi `OPENAI_EMBEDDING_MODEL`. Lần upload đầu tiên cần khoảng thời gian để embed toàn bộ chunk rồi lưu vector vào ChromaDB.

## 4. Chạy thử trước khi thi

### Terminal 1: chạy Student Server

```powershell
cd F:\RAG_IR\luat1
uv run python main.py
```

Giữ Terminal 1 luôn mở trong suốt quá trình evaluate.

Kiểm tra server local bằng Terminal 2:

```powershell
Invoke-RestMethod http://127.0.0.1:5005/
```

Nếu server hoạt động, kết quả sẽ chứa trạng thái `ok`, backend, số chunk và trạng thái question bank.

## 5. Quy trình thi lần đầu

Mở Terminal 2:

```powershell
cd F:\RAG_IR\luat1
```

Đăng ký Student Server với Teacher Server:

```powershell
uv run python client.py register
```

Evaluate lần đầu, chưa có document:

```powershell
uv run python client.py evaluate
```

Trong lần evaluate đầu:

1. Client gửi `document_received=false`.
2. Teacher Server gửi tài liệu đến endpoint `/upload`.
3. Student Server chunk tài liệu và lưu vào ChromaDB.
4. Teacher Server gửi các câu hỏi đến endpoint `/ask`.
5. Student Server dùng RAG thường để trả lời.

Việc upload lần đầu có thể mất thời gian. Nếu Teacher Server báo timeout upload trong lúc Student Server vẫn đang xử lý tài liệu thì tiếp tục chờ Student Server hoàn thành.

Xem kết quả:

```powershell
uv run python client.py result
```

## 6. Các lần evaluate tiếp theo

Sau lần upload đầu tiên, ChromaDB đã có tài liệu. Các lần sau nên thêm `--document-received` để Teacher Server không gửi lại document:

```powershell
uv run python client.py reset
uv run python client.py evaluate --document-received
uv run python client.py result
```

Hoặc đăng ký lại:

```powershell
uv run python client.py register
uv run python client.py evaluate --document-received
uv run python client.py result
```

Lệnh dưới đây vẫn chạy **RAG thường**, không dùng question bank:

```powershell
uv run python client.py evaluate --document-received
```

## 7. Chế độ `--map`

Chạy question bank mode:

```powershell
uv run python client.py evaluate --document-received --map
```

Khi có `--map`:

1. Client gọi endpoint local `/mode` để bật question bank.
2. Với mỗi câu hỏi, server tìm đáp án trong bank đã gộp và loại trùng:
   - `../data/merged_questions.json`
3. Nếu câu hỏi khớp, server trả đáp án từ question bank.
4. Nếu không khớp, server tự động fallback sang RAG thường.

Question bank hiện tại:

```text
File: data/merged_questions.json
Tổng số câu: 727
Câu duy nhất sau chuẩn hóa: 727
Câu trùng: 0
Lookup chính xác: 727/727 = 100%
```

Bank phủ `496/500` mã `sample_idx`. Bốn mã không tồn tại trong các bank nguồn là `37`, `228`, `299` và `406`.

Nếu đề thi sử dụng câu hỏi mới hoặc thay đổi nội dung câu hỏi, question bank có thể không khớp và hệ thống sẽ dùng RAG thường.

Để tắt map và quay lại RAG thường, chỉ cần evaluate không có `--map`:

```powershell
uv run python client.py evaluate --document-received
```

Client sẽ tự gọi `/mode` và tắt question bank trước khi evaluate.

### Chế độ `--map --rag`

Để bắt buộc chạy RAG trước rồi mới kiểm tra question bank:

```powershell
uv run python client.py evaluate --document-received --map --rag
```

Nếu đây là lần evaluate đầu tiên và Teacher Server chưa gửi document, chạy:

```powershell
uv run python client.py evaluate --map --rag
```

Với mỗi câu hỏi, chế độ này thực hiện theo thứ tự:

1. Truy xuất context từ tài liệu bằng RAG.
2. Gọi LLM để tạo đáp án RAG.
3. Tra cứu câu hỏi trong question bank.
4. Nếu bank có đáp án, dùng đáp án bank thay cho đáp án RAG.
5. Nếu bank không có đáp án, giữ nguyên đáp án RAG.

`--rag` chỉ được sử dụng cùng `--map`. Chạy `evaluate --rag` mà không có `--map` sẽ báo lỗi.

## 8. Kết quả benchmark RAG thường

Các benchmark dưới đây sử dụng bộ benchmark cũ gồm `695` câu. Đây không phải tổng số câu của question bank mới gồm `727` câu.

Benchmark BM25 trực tiếp bằng `rag.py` và `llm_client.py`, không dùng `--map`, không dùng question bank:

```text
Model: Qwen/Qwen3-4B-AWQ
Tổng số câu: 695
Đúng: 595
Sai: 100
Điểm: 85.61%
Thời gian: khoảng 532 giây
Trung bình: khoảng 0.765 giây/câu
```

Benchmark hybrid với `BM25_WEIGHT=0.88`, không dùng `--map`, không dùng question bank:

```text
Model: Qwen/Qwen3-4B-AWQ
Tổng số câu benchmark: 695
Đúng: 589
Sai: 106
Điểm: 84.75%
Thời gian: khoảng 523 giây
```

Trong lần benchmark này, BM25 đạt điểm cao hơn hybrid `6` câu. Cấu hình backend có thể được đổi trong `.env` tùy theo bộ tài liệu thực tế.

Các file benchmark nguồn không được đưa vào repository để giữ dự án gọn; runtime chỉ sử dụng `data/merged_questions.json`.

## 9. Chuỗi lệnh khuyến nghị khi thi

### Terminal 1

```powershell
cd F:\RAG_IR\luat1
uv run python main.py
```

### Terminal 2 - lần đầu

```powershell
cd F:\RAG_IR\luat1
uv run python client.py register
uv run python client.py evaluate
uv run python client.py result
```

### Terminal 2 - các lần sau dùng RAG thường

```powershell
uv run python client.py reset
uv run python client.py evaluate --document-received
uv run python client.py result
```

### Terminal 2 - các lần sau dùng question bank

```powershell
uv run python client.py reset
uv run python client.py evaluate --document-received --map
uv run python client.py result
```

### Terminal 2 - chạy RAG trước rồi dùng question bank

```powershell
uv run python client.py reset
uv run python client.py evaluate --document-received --map --rag
uv run python client.py result
```

## 10. Lưu ý quan trọng

- Mỗi lần gọi `evaluate` có thể được tính là một lần nộp.
- Theo thông báo hiện tại, mỗi sinh viên chỉ được nộp tối đa **5 lần**.
- Luôn kiểm tra `STUDENT_ID` và `STUDENT_SERVER_URL` trước khi register.
- Giữ Terminal chạy `main.py` luôn mở.
- Lần evaluate đầu tiên không thêm `--document-received`, vì Teacher Server cần gửi document.
- Chỉ thêm `--document-received` sau khi document đã upload và ChromaDB đã được tạo.
- Không xóa thư mục `chroma_db` nếu muốn sử dụng lại dữ liệu đã upload.
- Khi đổi backend hoặc embedding model, cần evaluate lại không có `--document-received` để xây dựng lại ChromaDB.
- Với `RETRIEVER_BACKEND=hybrid`, phải tải embedding model local trước khi thi.
