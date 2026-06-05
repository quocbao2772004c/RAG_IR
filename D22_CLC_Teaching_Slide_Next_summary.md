# Ghi chú: Xây dựng API Endpoint truy vấn tài liệu bằng LLM + RAG với FastAPI

## 1. Mục tiêu bài tập

Xây dựng hệ thống cung cấp API endpoint để truy vấn tài liệu sử dụng **LLM + RAG** với framework **FastAPI**.

Sinh viên cần triển khai một **Student Server** chạy local, nhận tài liệu từ Teacher Server, xử lý chunking, embedding, lưu vào VectorDB, sau đó trả lời câu hỏi trắc nghiệm thông qua RAG.

---

## 2. Luồng thi tổng quan

Hệ thống gồm 3 thành phần chính:

1. **Student Servers**
   - Server local do sinh viên tự triển khai.
   - Nằm trong LAN không có kết nối WAN.
   - Nhận request từ Teacher Proxy Server.

2. **Teacher Proxy Server**
   - Đóng vai trò trung gian điều phối cuộc thi.
   - Gọi tới Student Server thông qua các endpoint `/upload`, `/ask`.
   - Proxy request tới PTIT LLM Server.

3. **PTIT LLM Server**
   - Server LLM có Internet.
   - Được truy cập thông qua proxy server.

Các endpoint chính trong luồng thi:

- Student gọi tới Teacher Server:
  - `POST /competition/register`
  - `POST /competition/evaluate`
  - `POST /competition/reset`
  - `GET /competition/result`

- Teacher Server gọi ngược về Student Server:
  - `POST /upload`
  - `POST /ask`

- Student Server gọi LLM thông qua proxy:
  - `POST /proxy`

---

## 3. Proxy Competition Server

Proxy Competition Server là backend phục vụ tổ chức thi cuối kỳ Offline RAG Competition.

Hệ thống đóng vai trò:

- API Gateway trung gian cho toàn bộ cuộc thi.
- Proxy điều phối request tới Public LLM.
- Hệ thống chấm điểm tự động.
- Bộ điều phối timeout và queue chống rate-limit.

### Thành phần chính

#### Teacher Server

Chức năng:

- Quản lý cuộc thi.
- Gửi document.
- Gửi câu hỏi.
- Chấm điểm.
- Proxy LLM API.

#### Student Server

Chức năng:

- Nhận tài liệu.
- Chunking + VectorDB.
- Thực hiện RAG.
- Trả lời câu hỏi.

---

## 4. Teacher Server APIs

Base URL:

```text
http://192.168.50.218:8000/api/v1
```

Header bắt buộc để định danh sinh viên:

```http
X-Student-ID: <Mã Sinh viên viết hoa>
```

---

### 4.1. Đăng ký Student Server

```http
POST /competition/register
```

Chức năng: Sinh viên đăng ký địa chỉ Student Server.

#### Request Schema

```python
class RegisterPayload(BaseModel):
    server_url: str
```

#### Response Schema

```python
class RegisterResponse(BaseModel):
    status: str
    student_id: str
    server_url: str
```

#### Ví dụ Request

```json
{
  "server_url": "http://192.168.1.15:5000"
}
```

#### Ví dụ Response

```json
{
  "message": "Đăng ký thành công!",
  "student_id": "B21DCCN629",
  "server_url": "http://192.168.1.15:5000"
}
```

---

### 4.2. Bắt đầu đánh giá

```http
POST /competition/evaluate
```

Chức năng: Bắt đầu quá trình thi. Teacher Server sẽ tự động gọi `/upload`, sau đó gọi 10 lần `/ask`.

Request body: Không cần content body.

#### Response Schema

```python
class EvaluateResponse(BaseModel):
    student_id: str
    score: float
    status: str
    detail: list
```

#### Ví dụ Response

```json
{
  "student_id": "B21DCCN629",
  "score": 8.0,
  "status": "completed",
  "detail": ["..."]
}
```

---

### 4.3. Reset trạng thái thi

```http
POST /competition/reset
```

Chức năng: Xóa bỏ điểm cũ, reset trạng thái thi để bắt đầu lại nếu code bị crash.

Request body: Không cần content body.

#### Response Schema

```python
class ResetResponse(BaseModel):
    status: str
    message: str
```

#### Ví dụ Response

```json
{
  "status": "success",
  "message": "Đã reset trạng thái cho sinh viên <Mã Sinh viên viết hoa>"
}
```

---

### 4.4. Kiểm tra kết quả hiện tại

```http
GET /competition/result
```

Chức năng: Kiểm tra trạng thái và điểm hiện tại của sinh viên.

Request body: Không cần content body.

#### Response Schema

```python
class ResultResponse(BaseModel):
    student_id: str
    score: float
    status: str
    current_question: int
```

#### Ví dụ Response

```json
{
  "student_id": "B21DCCN629",
  "score": 5.0,
  "status": "evaluating",
  "current_question": 6
}
```

---

## 5. Gọi LLM khi không có mạng

Sinh viên không gọi trực tiếp Public LLM, mà gọi qua proxy server.

Endpoint:

```http
POST /proxy
```

Ví dụ dùng OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.50.218:8000/api/v1/proxy",
    api_key="B21DCCN629"  # Dùng MSSV trên lớp làm API KEY
)

res = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "..."}]
)
```

---

## 6. Teacher Server như một Client

Teacher Server không chỉ ngồi chờ request. Sau khi Student gọi `/evaluate`, Teacher Server sẽ chủ động gửi request ngược về Student Server.

Quy trình:

1. Gửi dữ liệu tài liệu RAG gốc xuống máy sinh viên.
   - Endpoint gọi đến Student Server: `/upload`
   - Timeout tối đa: 120 giây

2. Gửi lần lượt 10 câu hỏi.
   - Endpoint gọi đến Student Server: `/ask`
   - Timeout mỗi câu: 60 giây

### Ví dụ request đến `/upload`

```json
{
  "doc_id": "none",
  "text": "Nội dung tài liệu RAG..."
}
```

### Ví dụ request đến `/ask`

```json
{
  "question": "RAG là gì? A.xxx B.xxx C.xxx D.xxx"
}
```

---

## 7. Student Server API Endpoints

Sinh viên bắt buộc phải viết đúng 2 endpoint trên server local:

- `POST /upload`
- `POST /ask`

Schema phải đúng như đã thống nhất.

---

### 7.1. Endpoint nhận tài liệu

```http
POST /upload
```

Chức năng: Nhận document từ Teacher Server, thực hiện chunking, embedding và lưu vào VectorDB.

#### Request Schema

```python
class UploadRequest(BaseModel):
    doc_id: Optional[str] = None
    text: str
```

#### Response Schema

```python
class UploadResponse(BaseModel):
    status: str
    doc_id: Optional[str] = None
    chunks: int
```

#### Ví dụ Response

```json
{
  "status": "success",
  "doc_id": "abc_doc",
  "chunks": 42
}
```

---

### 7.2. Endpoint trả lời câu hỏi

```http
POST /ask
```

Chức năng: Nhận truy vấn, retrieve context từ VectorDB, gửi cho LLM và trả ra đáp án trắc nghiệm.

#### Request Schema

```python
class AskRequest(BaseModel):
    question: str
```

#### Response Schema

```python
class AskResponse(BaseModel):
    answer: str
    sources: List[str] = []
```

#### Ví dụ Response

```json
{
  "answer": "B",
  "sources": [
    "chunk_1_content_...",
    "chunk_2_content_..."
  ]
}
```

---

## 8. Quy định quan trọng

Trường `answer` trong response của `/ask` bắt buộc chỉ được là **1 ký tự duy nhất**:

```text
A / B / C / D
```

Không trả lời dạng câu dài, không giải thích trong trường `answer`.

Ví dụ đúng:

```json
{
  "answer": "B",
  "sources": [
    "chunk_1_content_...",
    "chunk_2_content_..."
  ]
}
```

---

## 9. Checklist triển khai nhanh

### Student Server cần có

- FastAPI app chạy local.
- Endpoint `POST /upload`.
- Endpoint `POST /ask`.
- Cơ chế chunking tài liệu.
- Embedding model local.
- VectorDB hoặc lưu vector đơn giản bằng FAISS/Chroma.
- Hàm retrieve top-k context.
- Hàm gọi LLM qua proxy.
- Response `/ask` chỉ trả `A`, `B`, `C` hoặc `D`.

### Thứ tự test đề xuất

1. Chạy Student Server local.
2. Test `/upload` bằng Postman hoặc curl.
3. Test `/ask` bằng câu hỏi mẫu.
4. Gọi `/competition/register` để đăng ký server.
5. Gọi `/competition/evaluate` để bắt đầu thi.
6. Gọi `/competition/result` để xem trạng thái và điểm.
7. Nếu code crash, gọi `/competition/reset`.

---

## 10. Tóm tắt ngắn

Bài tập yêu cầu sinh viên xây dựng một FastAPI server local có 2 endpoint chính: `/upload` để nhận tài liệu và tạo VectorDB, `/ask` để nhận câu hỏi và trả đáp án trắc nghiệm. Teacher Server sẽ gọi ngược vào Student Server khi đánh giá. Sinh viên đăng ký server bằng `/competition/register`, bắt đầu chấm bằng `/competition/evaluate`, xem kết quả bằng `/competition/result`, và có thể reset bằng `/competition/reset`.
