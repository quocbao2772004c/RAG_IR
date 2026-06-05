# Bài tập: Xây dựng hệ thống cung cấp API Endpoint để truy vấn tài liệu sử dụng LLM + RAG với FastAPI

## 1. Mục tiêu bài tập

Bài tập yêu cầu sinh viên xây dựng một **Student Server** chạy trên máy cá nhân, cung cấp các API endpoint để tham gia một cuộc thi/đánh giá offline về **RAG - Retrieval-Augmented Generation**.

Hệ thống cần có khả năng:

- Nhận tài liệu từ **Teacher Server**.
- Xử lý tài liệu bằng các bước RAG:
  - chia nhỏ văn bản thành các đoạn nhỏ (*chunking*);
  - tạo embedding cho từng đoạn;
  - lưu embedding vào Vector Database hoặc cấu trúc tương đương;
  - truy xuất các đoạn liên quan khi nhận câu hỏi.
- Gửi ngữ cảnh truy xuất được cho LLM.
- Trả về đáp án trắc nghiệm đúng định dạng.
- Tương tác đúng với các API mà Teacher Server yêu cầu.

Bối cảnh bài thi là hệ thống **offline trong mạng LAN**, trong đó máy sinh viên không gọi trực tiếp Internet mà gọi LLM thông qua proxy do Teacher Server cung cấp.

---

## 2. Kiến trúc tổng thể hệ thống

### 2.1. Các thành phần chính

Hệ thống gồm 3 thành phần chính:

1. **Teacher Proxy Server**
2. **PTIT LLM Server**
3. **Student Server**

### 2.2. Luồng thi tổng quát

Trong sơ đồ ở slide 2:

- **Student Server** nằm trong mạng LAN, không có kết nối WAN/Internet trực tiếp.
- **Teacher Proxy Server** đóng vai trò trung gian giữa sinh viên và hệ thống LLM.
- **PTIT LLM Server** là nơi xử lý yêu cầu sinh văn bản từ mô hình ngôn ngữ lớn.
- Teacher Proxy Server có thể truy cập Internet hoặc các dịch vụ LLM bên ngoài thông qua mạng Wi-Fi ngoài.

Các API chính trong luồng thi gồm:

- Sinh viên gọi tới Teacher Server:
  - `POST /register`
  - `POST /evaluate`
- Teacher Server gọi ngược về Student Server:
  - `POST /upload`
  - `POST /ask`
- Student Server hoặc logic RAG gọi LLM thông qua proxy:
  - `POST /chat/completions`
  - `POST /v1/completions`

Ý tưởng chính là sinh viên không cần gọi LLM trực tiếp ra Internet. Thay vào đó, mọi request đến LLM được đi qua proxy do Teacher Server cung cấp.

---

## 3. Proxy Competition Server

### 3.1. Vai trò của Proxy Competition Server

**Proxy Competition Server** là backend phục vụ tổ chức bài thi cuối kỳ dạng Offline RAG Competition.

Hệ thống này đảm nhiệm nhiều vai trò:

- Là **API Gateway trung gian** cho toàn bộ cuộc thi.
- Là **proxy điều phối request tới Public LLM**.
- Là **hệ thống chấm điểm tự động**.
- Là **bộ điều phối timeout và queue** nhằm hạn chế lỗi do rate limit hoặc quá tải.

### 3.2. Teacher Server

Teacher Server là thành phần quản lý cuộc thi.

Các nhiệm vụ chính:

- Quản lý quá trình thi.
- Gửi tài liệu cho Student Server.
- Gửi lần lượt các câu hỏi đến Student Server.
- Chấm điểm dựa trên câu trả lời.
- Cung cấp proxy LLM API để sinh viên gọi mô hình ngôn ngữ.

### 3.3. Student Server

Student Server là server do sinh viên tự xây dựng và chạy trên máy cá nhân.

Student Server cần:

- Nhận tài liệu từ Teacher Server.
- Thực hiện chia nhỏ tài liệu thành chunk.
- Tạo embedding.
- Lưu vào VectorDB.
- Khi nhận câu hỏi, thực hiện RAG để tìm context phù hợp.
- Gửi context và câu hỏi tới LLM.
- Trả về đáp án trắc nghiệm.

---

## 4. Teacher Server APIs

Teacher Server có base URL:

```text
http://192.168.50.218:8000/api/v1
```

Tất cả các API liên quan đến sinh viên đều yêu cầu header định danh:

```http
X-Student-ID: <Mã Sinh viên viết hoa>
```

Ví dụ:

```http
X-Student-ID: B21DCCN629
```

---

## 5. API đăng ký Student Server

### 5.1. Endpoint

```http
POST /competition/register
```

### 5.2. Chức năng

API này dùng để sinh viên đăng ký địa chỉ Student Server của mình với Teacher Server.

Teacher Server cần biết URL của Student Server để sau đó có thể gọi ngược lại các endpoint `/upload` và `/ask`.

### 5.3. Header bắt buộc

```http
X-Student-ID: <Mã Sinh viên viết hoa>
```

### 5.4. Request Schema

```python
class RegisterPayload(BaseModel):
    server_url: str
```

Trong đó:

- `server_url`: địa chỉ server local của sinh viên, ví dụ `http://192.168.1.15:5000`.

### 5.5. Ví dụ Request Body

```json
{
  "server_url": "http://192.168.1.15:5000"
}
```

### 5.6. Response Schema

```python
class RegisterResponse(BaseModel):
    status: str
    student_id: str
    server_url: str
```

### 5.7. Ví dụ Response

```json
{
  "message": "Đăng ký thành công!",
  "student_id": "B21DCCN629",
  "server_url": "http://192.168.1.15:5000"
}
```

### 5.8. Lưu ý triển khai

Trước khi gọi `/competition/register`, sinh viên cần đảm bảo:

- Student Server đã chạy.
- Server có thể truy cập được từ Teacher Server trong cùng mạng LAN.
- Địa chỉ IP và port khai báo trong `server_url` là đúng.
- Firewall không chặn kết nối từ Teacher Server đến Student Server.

---

## 6. API bắt đầu đánh giá

### 6.1. Endpoint

```http
POST /competition/evaluate
```

### 6.2. Chức năng

API này dùng để bắt đầu quá trình thi/đánh giá.

Khi sinh viên gọi endpoint này, Teacher Server sẽ tự động:

1. Gọi endpoint `/upload` trên Student Server để gửi tài liệu RAG.
2. Sau đó gọi endpoint `/ask` trên Student Server 10 lần, tương ứng với 10 câu hỏi.
3. Thu thập đáp án.
4. Chấm điểm.
5. Trả về kết quả sau khi quá trình kết thúc.

### 6.3. Header bắt buộc

```http
X-Student-ID: <Mã Sinh viên viết hoa>
```

### 6.4. Request Body

Không cần content body.

### 6.5. Response Schema

```python
class EvaluateResponse(BaseModel):
    student_id: str
    score: float
    status: str
    detail: list
```

Trong đó:

- `student_id`: mã sinh viên.
- `score`: điểm hiện tại hoặc điểm cuối cùng.
- `status`: trạng thái quá trình đánh giá.
- `detail`: thông tin chi tiết từng câu hỏi/câu trả lời.

### 6.6. Ví dụ Response

```json
{
  "student_id": "B21DCCN629",
  "score": 8.0,
  "status": "completed",
  "detail": [ ... ]
}
```

---

## 7. API reset trạng thái thi

### 7.1. Endpoint

```http
POST /competition/reset
```

### 7.2. Chức năng

API này dùng để xóa bỏ điểm cũ và reset trạng thái thi để bắt đầu lại.

Trường hợp sử dụng:

- Code Student Server bị crash.
- Sinh viên cần chạy lại quá trình evaluate.
- Trạng thái cũ bị kẹt.
- Muốn làm lại từ đầu sau khi sửa lỗi.

### 7.3. Header bắt buộc

```http
X-Student-ID: <Mã Sinh viên viết hoa>
```

### 7.4. Request Body

Không cần content body.

### 7.5. Response Schema

```python
class ResetResponse(BaseModel):
    status: str
    message: str
```

### 7.6. Ví dụ Response

```json
{
  "status": "success",
  "message": "Đã reset trạng thái cho sinh viên <Mã Sinh viên viết hoa>"
}
```

---

## 8. API xem kết quả hiện tại

### 8.1. Endpoint

```http
GET /competition/result
```

### 8.2. Chức năng

API này dùng để kiểm tra trạng thái và điểm hiện tại của sinh viên.

Có thể dùng API này trong lúc quá trình evaluate đang chạy để xem tiến độ.

### 8.3. Header bắt buộc

```http
X-Student-ID: <Mã Sinh viên viết hoa>
```

### 8.4. Request Body

Không cần content body.

### 8.5. Response Schema

```python
class ResultResponse(BaseModel):
    student_id: str
    score: float
    status: str
    current_question: int
```

Trong đó:

- `student_id`: mã sinh viên.
- `score`: điểm hiện tại.
- `status`: trạng thái hiện tại, ví dụ `evaluating`, `completed`.
- `current_question`: câu hỏi đang được xử lý hoặc số thứ tự câu hiện tại.

### 8.6. Ví dụ Response

```json
{
  "student_id": "B21DCCN629",
  "score": 5.0,
  "status": "evaluating",
  "current_question": 6
}
```

---

## 9. Gọi LLM khi không có mạng Internet trực tiếp

### 9.1. Vấn đề

Trong bài thi, máy sinh viên nằm trong mạng LAN và không có kết nối WAN/Internet trực tiếp. Vì vậy, sinh viên không thể gọi trực tiếp API của OpenAI hoặc các Public LLM từ máy mình.

### 9.2. Cách giải quyết

Sinh viên gọi LLM thông qua **Proxy LLM** được cung cấp bởi Teacher Server.

Endpoint proxy:

```text
http://192.168.50.218:8000/api/v1/proxy
```

### 9.3. Ví dụ sử dụng OpenAI client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.50.218:8000/api/v1/proxy",
    api_key="B21DCCN629"  # Dùng MSSV trên lớp làm API KEY
)

res = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "..."
        }
    ]
)
```

### 9.4. Ý nghĩa các tham số

- `base_url`: trỏ đến proxy server của Teacher Server.
- `api_key`: dùng mã sinh viên trên lớp làm API key.
- `model`: model được proxy hỗ trợ, ví dụ `gpt-4o-mini`.
- `messages`: danh sách message theo định dạng chat completion.

---

## 10. Teacher Server hoạt động như một client

### 10.1. Ý tưởng chính

Teacher Server không chỉ ngồi chờ sinh viên gửi request.

Sau khi sinh viên gọi `/competition/evaluate`, Teacher Server sẽ chủ động gửi request ngược về Student Server đã đăng ký trước đó.

### 10.2. Các bước Teacher Server thực hiện

Teacher Server sẽ thực hiện:

1. **Gửi dữ liệu tài liệu RAG gốc xuống máy sinh viên**
   - Gọi endpoint `/upload` của Student Server.
   - Thời gian chờ tối đa: 120 giây.

2. **Bơm 10 câu hỏi lần lượt**
   - Gọi endpoint `/ask` của Student Server.
   - Mỗi câu hỏi có thời gian chờ tối đa: 60 giây.

### 10.3. Ví dụ request đến `/upload`

```json
{
  "doc_id": "none",
  "text": "Nội dung tài liệu RAG..."
}
```

### 10.4. Ví dụ request đến `/ask`

```json
{
  "question": "RAG là gì? A.xxx B.xxx C.xxx D.xxx"
}
```

### 10.5. Lưu ý quan trọng

Vì Teacher Server gọi ngược về Student Server nên Student Server phải:

- Chạy ổn định trong suốt quá trình thi.
- Mở đúng port đã đăng ký.
- Có địa chỉ IP nội bộ đúng.
- Trả response đúng schema.
- Không xử lý quá thời gian timeout.
- Không bị lỗi khi nhận nhiều request liên tiếp.

---

## 11. Student Server API Endpoints

Sinh viên bắt buộc phải viết đúng 2 endpoint:

```http
POST /upload
POST /ask
```

Hai endpoint này phải tuân thủ đúng schema đã thống nhất trước trên server local của sinh viên.

---

## 12. Endpoint `/upload`

### 12.1. Endpoint

```http
POST /upload
```

### 12.2. Chức năng

Endpoint này nhận document từ Teacher Server.

Sau khi nhận tài liệu, Student Server cần thực hiện:

1. Lấy nội dung văn bản từ trường `text`.
2. Chia văn bản thành nhiều chunk.
3. Tạo embedding cho từng chunk.
4. Lưu chunk và embedding vào VectorDB.
5. Trả về số lượng chunk đã tạo.

### 12.3. Request Schema

```python
class UploadRequest(BaseModel):
    doc_id: Optional[str] = None
    text: str
```

Trong đó:

- `doc_id`: mã tài liệu, có thể không có.
- `text`: nội dung tài liệu RAG gốc.

### 12.4. Response Schema

```python
class UploadResponse(BaseModel):
    status: str
    doc_id: Optional[str] = None
    chunks: int
```

Trong đó:

- `status`: trạng thái xử lý, ví dụ `success`.
- `doc_id`: mã tài liệu trả về.
- `chunks`: số lượng chunk sau khi chia tài liệu.

### 12.5. Ví dụ Response

```json
{
  "status": "success",
  "doc_id": "abc_doc",
  "chunks": 42
}
```

### 12.6. Gợi ý xử lý trong `/upload`

Một pipeline cơ bản có thể gồm:

```text
Nhận text
→ Làm sạch văn bản
→ Chia chunk
→ Tạo embedding
→ Lưu vào VectorDB
→ Trả về số lượng chunk
```

Các lưu ý:

- Chunk không nên quá ngắn vì dễ mất ngữ cảnh.
- Chunk không nên quá dài vì khó retrieve chính xác và tốn token.
- Nên dùng overlap giữa các chunk để tránh mất thông tin ở ranh giới.
- Có thể lưu `chunk_id`, `content`, `embedding`, `doc_id`.

---

## 13. Endpoint `/ask`

### 13.1. Endpoint

```http
POST /ask
```

### 13.2. Chức năng

Endpoint này nhận câu hỏi từ Teacher Server, retrieve context từ VectorDB, gửi cho LLM và trả ra đáp án trắc nghiệm.

### 13.3. Request Schema

```python
class AskRequest(BaseModel):
    question: str
```

Trong đó:

- `question`: câu hỏi trắc nghiệm, thường gồm nội dung câu hỏi và các lựa chọn A/B/C/D.

### 13.4. Response Schema

```python
class AskResponse(BaseModel):
    answer: str
    sources: List[str] = []
```

Trong đó:

- `answer`: đáp án trắc nghiệm.
- `sources`: danh sách các chunk được dùng làm nguồn tham khảo.

### 13.5. Ví dụ Response

```json
{
  "answer": "B",
  "sources": [
    "chunk_1_content_...",
    "chunk_2_content_..."
  ]
}
```

### 13.6. Pipeline xử lý trong `/ask`

Một pipeline RAG cơ bản:

```text
Nhận question
→ Tạo embedding cho question
→ Tìm top-k chunk liên quan trong VectorDB
→ Ghép context + question + options thành prompt
→ Gọi LLM qua proxy
→ Parse đáp án
→ Trả về JSON đúng schema
```

### 13.7. Gợi ý prompt cho LLM

Có thể thiết kế prompt theo hướng ép LLM chỉ trả về một ký tự:

```text
Bạn là hệ thống trả lời trắc nghiệm dựa trên tài liệu được cung cấp.

Chỉ sử dụng CONTEXT bên dưới để trả lời.
Chọn duy nhất một đáp án trong A, B, C hoặc D.
Chỉ trả về đúng một ký tự: A, B, C hoặc D.
Không giải thích.

CONTEXT:
{context}

QUESTION:
{question}
```

### 13.8. Xử lý output từ LLM

Vì yêu cầu đáp án chỉ được là một ký tự, cần kiểm tra output:

- Nếu LLM trả `"B"` thì giữ nguyên.
- Nếu LLM trả `"Đáp án là B"` thì parse lấy ký tự `B`.
- Nếu output không có A/B/C/D thì có thể fallback bằng luật mặc định hoặc gọi lại prompt ngắn hơn.
- Luôn đảm bảo response cuối cùng có `answer` là một trong bốn giá trị: `A`, `B`, `C`, `D`.

---

## 14. Quy định quan trọng về đáp án

Giống như điền đáp án trắc nghiệm, trường `answer` bắt buộc chỉ được là **1 ký tự duy nhất**:

```text
A / B / C / D
```

Không được trả:

```json
{
  "answer": "Đáp án là B"
}
```

Không được trả:

```json
{
  "answer": "B. Vì ..."
}
```

Không được trả:

```json
{
  "answer": "B hoặc C"
}
```

Response hợp lệ phải có dạng:

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

## 15. Checklist triển khai cho sinh viên

### 15.1. Trước khi chạy bài thi

Cần chuẩn bị:

- Cài FastAPI.
- Cài Uvicorn.
- Cài thư viện gọi OpenAI-compatible API.
- Có module chunking.
- Có module embedding.
- Có VectorDB hoặc cách lưu vector tương đương.
- Student Server chạy được trong LAN.
- Đã biết IP LAN của máy mình.
- Đã đăng ký server URL với Teacher Server.

### 15.2. Các endpoint bắt buộc

Student Server phải có:

```http
POST /upload
POST /ask
```

Teacher Server có các endpoint sinh viên sẽ gọi:

```http
POST /competition/register
POST /competition/evaluate
POST /competition/reset
GET  /competition/result
```

### 15.3. Các timeout cần lưu ý

- `/upload`: Teacher Server đợi tối đa 120 giây.
- `/ask`: Teacher Server đợi tối đa 60 giây cho mỗi câu hỏi.

Do đó, cần tối ưu:

- thời gian chunking;
- thời gian embedding;
- thời gian retrieve;
- thời gian gọi LLM;
- thời gian parse đáp án.

---

## 16. Gợi ý cấu trúc FastAPI cho Student Server

Một cấu trúc đơn giản:

```text
student_server/
├── main.py
├── rag.py
├── llm_client.py
├── vector_store.py
└── requirements.txt
```

Trong đó:

- `main.py`: định nghĩa FastAPI app và 2 endpoint `/upload`, `/ask`.
- `rag.py`: xử lý chunking, retrieve, build prompt.
- `llm_client.py`: gọi LLM qua proxy.
- `vector_store.py`: lưu và truy xuất vector.
- `requirements.txt`: danh sách thư viện cần cài.

---

## 17. Gợi ý code khung FastAPI

```python
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class UploadRequest(BaseModel):
    doc_id: Optional[str] = None
    text: str

class UploadResponse(BaseModel):
    status: str
    doc_id: Optional[str] = None
    chunks: int

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: List[str] = []

@app.post("/upload", response_model=UploadResponse)
def upload(req: UploadRequest):
    # 1. chunking
    # 2. embedding
    # 3. save vector db
    # 4. return number of chunks
    return UploadResponse(
        status="success",
        doc_id=req.doc_id,
        chunks=0
    )

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # 1. retrieve context
    # 2. call LLM
    # 3. parse answer
    return AskResponse(
        answer="B",
        sources=[]
    )
```

---

## 18. Gợi ý lệnh chạy server

Ví dụ chạy Student Server tại host `0.0.0.0` để Teacher Server trong LAN có thể truy cập:

```bash
uvicorn main:app --host 0.0.0.0 --port 5000
```

Sau đó đăng ký với Teacher Server bằng URL dạng:

```text
http://<IP-LAN-của-máy-sinh-viên>:5000
```

Ví dụ:

```text
http://192.168.1.15:5000
```

---

## 19. Gợi ý gọi API đăng ký

Ví dụ dùng `curl`:

```bash
curl -X POST "http://192.168.50.218:8000/api/v1/competition/register" \
  -H "Content-Type: application/json" \
  -H "X-Student-ID: B21DCCN629" \
  -d '{"server_url": "http://192.168.1.15:5000"}'
```

---

## 20. Gợi ý gọi API evaluate

```bash
curl -X POST "http://192.168.50.218:8000/api/v1/competition/evaluate" \
  -H "X-Student-ID: B21DCCN629"
```

---

## 21. Gợi ý gọi API reset

```bash
curl -X POST "http://192.168.50.218:8000/api/v1/competition/reset" \
  -H "X-Student-ID: B21DCCN629"
```

---

## 22. Gợi ý gọi API result

```bash
curl -X GET "http://192.168.50.218:8000/api/v1/competition/result" \
  -H "X-Student-ID: B21DCCN629"
```

---

## 23. Các lỗi thường gặp

### 23.1. Teacher Server không gọi được Student Server

Nguyên nhân có thể:

- Sai IP LAN.
- Sai port.
- Student Server chưa chạy.
- Chạy server với `127.0.0.1` thay vì `0.0.0.0`.
- Firewall chặn kết nối.
- Máy không cùng mạng LAN.

Cách kiểm tra:

- Từ máy khác trong LAN thử truy cập `http://<ip>:<port>/docs`.
- Kiểm tra IP bằng `ipconfig` trên Windows hoặc `ip addr` trên Linux.

### 23.2. Response sai schema

Ví dụ sai:

```json
{
  "result": "B"
}
```

Trong khi đúng phải là:

```json
{
  "answer": "B",
  "sources": []
}
```

### 23.3. Đáp án sai định dạng

Sai:

```json
{
  "answer": "Đáp án là B"
}
```

Đúng:

```json
{
  "answer": "B"
}
```

### 23.4. Xử lý quá lâu

Nếu `/upload` quá 120 giây hoặc `/ask` quá 60 giây, Teacher Server có thể tính là lỗi timeout.

Cần tối ưu:

- Không gọi embedding quá nhiều lần không cần thiết.
- Lưu cache vector sau `/upload`.
- Khi `/ask`, chỉ retrieve và gọi LLM.
- Prompt cần ngắn gọn.
- Số lượng chunk đưa vào context nên vừa đủ.

---

## 24. Tóm tắt yêu cầu bắt buộc

Sinh viên cần xây dựng Student Server bằng FastAPI có:

1. Endpoint `POST /upload`
   - Nhận document.
   - Chunking.
   - Embedding.
   - Lưu VectorDB.
   - Trả về số chunk.

2. Endpoint `POST /ask`
   - Nhận câu hỏi.
   - Retrieve context.
   - Gọi LLM qua proxy.
   - Trả về đáp án trắc nghiệm.
   - `answer` chỉ được là một ký tự `A`, `B`, `C` hoặc `D`.

3. Đăng ký server với Teacher Server qua:

```http
POST /competition/register
```

4. Bắt đầu chấm điểm qua:

```http
POST /competition/evaluate
```

5. Có thể reset hoặc xem kết quả bằng:

```http
POST /competition/reset
GET /competition/result
```

---

## 25. Kết luận

Slide mô tả một bài tập xây dựng hệ thống RAG phục vụ thi offline. Điểm quan trọng nhất không chỉ là tạo được API FastAPI, mà còn phải đảm bảo Student Server tuân thủ đúng schema, đúng endpoint, xử lý được tài liệu, gọi được LLM qua proxy và trả lời đúng định dạng trắc nghiệm.

Trong bài thi, Teacher Server đóng vai trò điều phối và chấm điểm tự động, còn Student Server là phần sinh viên phải tự xây dựng để nhận tài liệu, thực hiện RAG và trả lời câu hỏi.
