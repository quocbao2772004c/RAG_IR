# Cấu trúc giao tiếp bắt buộc với Teacher Server

File này chỉ ghi phần **contract/API payload** cần đúng để Teacher Server chấm được.

## 1. Mình phải chạy cái gì?

Mình chạy một **Student Server** local, thường bằng:

```bash
uv run python main.py
```

Teacher Server sẽ gọi vào Student Server qua URL mình đăng ký:

```env
STUDENT_SERVER_URL=http://IP_LAN_CUA_MAY:PORT
```

Không dùng `127.0.0.1` nếu Teacher Server chạy trên máy khác.

## 2. Student đăng ký với Teacher

Client gửi request lên Teacher Server.

Endpoint:

```http
POST {TEACHER_BASE_URL}/competition/register
```

hoặc fallback:

```http
POST {TEACHER_BASE_URL}/register
```

Headers:

```http
X-Student-ID: MSSV_CUA_BAN
Content-Type: application/json
```

Body bắt buộc:

```json
{
  "server_url": "http://IP_LAN_CUA_MAY:PORT"
}
```

Ý nghĩa:

- `server_url` là URL Teacher dùng để gọi lại Student Server.
- URL này phải truy cập được từ máy Teacher.

## 3. Student yêu cầu Teacher evaluate

Endpoint:

```http
POST {TEACHER_BASE_URL}/competition/evaluate
```

hoặc fallback:

```http
POST {TEACHER_BASE_URL}/evaluate
```

Headers:

```http
X-Student-ID: MSSV_CUA_BAN
Content-Type: application/json
```

Body:

```json
{
  "document_received": false
}
```

Ý nghĩa:

- `false`: Teacher sẽ gửi tài liệu sang Student Server qua `/upload`.
- `true`: Teacher bỏ qua upload, dùng index/tài liệu Student Server đã có sẵn.

Lần đầu thường dùng:

```json
{
  "document_received": false
}
```

Các lần sau, nếu server chưa tắt hoặc đã lưu index:

```json
{
  "document_received": true
}
```

## 4. Teacher gọi Student Server: `/upload`

Đây là endpoint bắt buộc Student Server phải có.

Endpoint:

```http
POST {STUDENT_SERVER_URL}/upload
```

Body tối thiểu Teacher có thể gửi:

```json
{
  "doc_id": "document_id_neu_co",
  "text": "noi dung tai lieu..."
}
```

Server trong repo này cũng chấp nhận một số tên field khác:

```json
{
  "id": "document_id_neu_co",
  "content": "noi dung tai lieu..."
}
```

hoặc:

```json
{
  "document_id": "document_id_neu_co",
  "document": "noi dung tai lieu..."
}
```

hoặc nhiều tài liệu:

```json
{
  "documents": [
    "noi dung tai lieu 1",
    "noi dung tai lieu 2"
  ]
}
```

Field quan trọng nhất:

- Bắt buộc phải lấy được text tài liệu từ một trong các field: `text`, `content`, `document`, `doc`, `documents`, `docs`.
- `doc_id`, `id`, `document_id` là optional.

Response Student Server nên trả:

```json
{
  "status": "success",
  "doc_id": "document_id_neu_co",
  "chunks": 123
}
```

Ý nghĩa:

- `status`: nên là `"success"`.
- `doc_id`: id tài liệu, có thể là `null` nếu Teacher không gửi.
- `chunks`: số chunk đã tạo/index.

Nếu lỗi:

```json
{
  "detail": "empty text"
}
```

hoặc HTTP `400`.

## 5. Teacher gọi Student Server: `/ask`

Đây là endpoint bắt buộc Student Server phải có.

Endpoint:

```http
POST {STUDENT_SERVER_URL}/ask
```

Body tối thiểu:

```json
{
  "question": "Cau hoi..."
}
```

Nếu câu hỏi có options tách riêng, Teacher có thể gửi dạng list:

```json
{
  "question": "Cau hoi...",
  "options": [
    "A. Dap an A",
    "B. Dap an B",
    "C. Dap an C",
    "D. Dap an D"
  ]
}
```

Hoặc dạng object:

```json
{
  "question": "Cau hoi...",
  "options": {
    "A": "Dap an A",
    "B": "Dap an B",
    "C": "Dap an C",
    "D": "Dap an D"
  }
}
```

Server trong repo này cũng có thể đọc options từ các field:

```json
{
  "question": "Cau hoi...",
  "choices": {
    "A": "Dap an A",
    "B": "Dap an B",
    "C": "Dap an C",
    "D": "Dap an D"
  }
}
```

hoặc:

```json
{
  "question": "Cau hoi...",
  "answers": [
    "A. Dap an A",
    "B. Dap an B",
    "C. Dap an C",
    "D. Dap an D"
  ]
}
```

Response bắt buộc nên có:

```json
{
  "answer": "A",
  "sources": []
}
```

Quan trọng:

- `answer` phải là đáp án cuối cùng, thường chỉ là một chữ cái `A`, `B`, `C`, hoặc `D`.
- Không nên trả lời dài trong `answer`.
- `sources` là optional về mặt chấm điểm nhưng nên trả list để debug.

Ví dụ response tốt:

```json
{
  "answer": "C",
  "sources": [
    "doan tai lieu lien quan..."
  ]
}
```

## 6. Health endpoint

Không phải phần trả lời câu hỏi, nhưng nên có để kiểm tra server sống.

Endpoint:

```http
GET {STUDENT_SERVER_URL}/
```

hoặc:

```http
GET {STUDENT_SERVER_URL}/health
```

Response ví dụ:

```json
{
  "status": "ok",
  "ready": true
}
```

## 7. Teacher reset/result

Client gọi Teacher để reset hoặc xem điểm.

Reset:

```http
POST {TEACHER_BASE_URL}/competition/reset
```

hoặc:

```http
POST {TEACHER_BASE_URL}/reset
```

Result:

```http
GET {TEACHER_BASE_URL}/competition/result
```

hoặc:

```http
GET {TEACHER_BASE_URL}/result
```

Headers vẫn là:

```http
X-Student-ID: MSSV_CUA_BAN
Content-Type: application/json
```

## 8. Tóm tắt contract quan trọng nhất

Teacher gọi Student:

```text
POST /upload
Input : {"doc_id": "...", "text": "..."}
Output: {"status": "success", "doc_id": "...", "chunks": 123}

POST /ask
Input : {"question": "...", "options": ["A. ...", "B. ...", "C. ...", "D. ..."]}
Output: {"answer": "A", "sources": []}
```

Student gọi Teacher:

```text
POST /competition/register
Input: {"server_url": "http://IP_LAN_CUA_MAY:PORT"}

POST /competition/evaluate
Input: {"document_received": false}

POST /competition/reset
GET  /competition/result
```

Header khi Student gọi Teacher:

```text
X-Student-ID: MSSV_CUA_BAN
Content-Type: application/json
```

## 9. Lỗi hay gặp

- `STUDENT_SERVER_URL` để `127.0.0.1`: Teacher ở máy khác sẽ không gọi được.
- `/ask` trả câu dài thay vì chữ cái `A/B/C/D`: dễ sai format chấm.
- Chưa chạy `/upload` nhưng đã `/ask`: server chưa có tài liệu/index.
- Lần đầu evaluate lại để `document_received=true`: Teacher không gửi tài liệu, server có thể thiếu index.
- Tắt server rồi evaluate tiếp nhưng không lưu index: cần `document_received=false` để Teacher upload lại.
