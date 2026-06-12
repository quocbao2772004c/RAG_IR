# Cài đặt và xử lý lỗi trên Windows

Hướng dẫn này áp dụng cho `v4`, `v5` và `luat1`.

## Cách chạy khuyến nghị

Mở PowerShell tại đúng thư mục phiên bản muốn chạy:

```powershell
cd C:\RAG_IR-main\RAG_IR-main\luat1
uv sync
uv run python download_model.py
Copy-Item .env.example .env
uv run python main.py
```

`uv run` tự sử dụng môi trường `.venv`, vì vậy thông thường không cần chạy `Activate.ps1`.

## Lỗi PowerShell chặn `Activate.ps1`

Thông báo thường gặp:

```text
running scripts is disabled on this system
Activate.ps1 cannot be loaded
```

Cho phép chạy script trong riêng cửa sổ PowerShell hiện tại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& "C:\RAG_IR-main\RAG_IR-main\v4\.venv\Scripts\Activate.ps1"
```

Nếu đang dùng phiên bản khác, thay `v4` bằng `v5` hoặc `luat1`.

Thiết lập `-Scope Process` chỉ có hiệu lực với cửa sổ PowerShell hiện tại và tự mất khi đóng cửa sổ.

Có thể không activate và chạy trực tiếp:

```powershell
uv run python main.py
```

## Lỗi `Application Control policy has blocked this file` hoặc `os error 4551`

Lỗi này khác với lỗi Execution Policy:

```text
Failed to query Python interpreter
An Application Control policy has blocked this file. (os error 4551)
```

`Set-ExecutionPolicy ... Bypass` không sửa được lỗi `4551`, vì Windows đang chặn chính file `python.exe` mới tạo trong `.venv`.

Kiểm tra Python nào đã được máy cho phép:

```powershell
py -0p
uv python list
```

Thử dùng bản Python hệ thống đã được allowlist:

```powershell
uv sync --python "C:\duong-dan-python-duoc-phep\python.exe"
uv run --python "C:\duong-dan-python-duoc-phep\python.exe" python main.py
```

Nếu Python trong `.venv` vẫn bị chặn, cần nhờ quản trị viên phòng máy cho phép `python.exe`, hoặc sử dụng môi trường `.venv` đã được tạo và kiểm tra trước đó. Không thể bỏ qua Windows Application Control chỉ bằng lệnh PowerShell thông thường.

## Không tìm thấy lệnh `uv`

Cài bằng PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Đóng rồi mở lại PowerShell, sau đó kiểm tra:

```powershell
uv --version
```

Nếu vẫn không nhận `uv`, có thể cài bằng Python:

```powershell
py -m pip install uv
py -m uv --version
py -m uv sync
```

## Không tìm thấy Python hoặc mở Microsoft Store

Kiểm tra Python:

```powershell
py --version
python --version
```

Nếu `python` mở Microsoft Store nhưng `py` hoạt động, dùng:

```powershell
py -m uv sync
```

Hoặc tắt alias Microsoft Store:

```text
Settings > Apps > Advanced app settings > App execution aliases
```

Tắt `python.exe` và `python3.exe`, sau đó mở lại PowerShell.

## Tạo lại `.venv` khi cài thư viện lỗi

Đóng các tiến trình Python đang chạy. Tại đúng thư mục phiên bản:

```powershell
Remove-Item -Recurse -Force .venv
uv sync
```

Nếu gặp lỗi hardlink hoặc máy sử dụng nhiều ổ đĩa:

```powershell
$env:UV_LINK_MODE="copy"
uv sync
```

Nếu mạng chậm hoặc lần cài trước bị ngắt:

```powershell
uv sync --refresh
```

## Tải embedding model trước khi thi

Khi dùng `RETRIEVER_BACKEND=hybrid` hoặc `sbert`:

```powershell
uv run python download_model.py
```

Phải tải model khi còn Internet. Trong phòng thi, cấu hình:

```env
MODEL_LOCAL_ONLY=true
```

Nếu báo không tìm thấy model local, kết nối Internet rồi chạy lại `download_model.py`.

## ChromaDB không khớp embedding model

Thông báo thường gặp:

```text
Embedding model changed. Rebuild ChromaDB
```

Xóa ChromaDB cũ rồi chạy evaluate lần đầu để nhận lại tài liệu:

```powershell
Remove-Item -Recurse -Force chroma_db
uv run python main.py
```

Trong Terminal khác:

```powershell
uv run python client.py register
uv run python client.py evaluate
```

Không dùng `--document-received` trong lần xây lại ChromaDB.

## Teacher Server không gọi được Student Server

Lấy địa chỉ IPv4:

```powershell
ipconfig
```

Điền IP LAN thật vào `.env`, không dùng `127.0.0.1`:

```env
STUDENT_SERVER_URL=http://192.168.50.123:5005
HOST=0.0.0.0
PORT=5005
```

Kiểm tra server local:

```powershell
Invoke-RestMethod http://127.0.0.1:5005/
```

Mở firewall cho port `5005` nếu máy khác không truy cập được:

```powershell
New-NetFirewallRule -DisplayName "RAG Student Server 5005" -Direction Inbound -Protocol TCP -LocalPort 5005 -Action Allow
```

Lệnh firewall có thể yêu cầu mở PowerShell bằng quyền Administrator.

## Port đã được sử dụng

Kiểm tra tiến trình đang giữ port:

```powershell
Get-NetTCPConnection -LocalPort 5005
```

Đổi port trong `.env` nếu cần:

```env
PORT=5006
STUDENT_SERVER_URL=http://IP_LAN_CUA_MAY:5006
```

Hai giá trị port phải giống nhau.

## Kiểm tra nhanh trước khi thi

```powershell
uv --version
uv sync
uv run python download_model.py
uv run python -m py_compile main.py client.py config.py rag.py llm_client.py vector_store.py
uv run python main.py
```

Terminal khác:

```powershell
Invoke-RestMethod http://127.0.0.1:5005/
uv run python client.py register
```
