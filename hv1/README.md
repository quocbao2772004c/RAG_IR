# hv1 - cache cau hoi va tra loi thu cong

Folder nay dung de Teacher ban cau hoi ve thi luu vao `questions.json`.
Ban co the tu dien dap an tren terminal, sau do server se doc JSON de tra loi.

## Cai goi can thiet

```powershell
cd F:\RAG_IR\hv1
pip install fastapi uvicorn
```

## Cach 1: chi luu cau hoi

```powershell
python server_save.py
```

Server chay o port `5003`. Moi cau hoi Teacher gui vao `/ask` se duoc luu trong
`questions.json`. Server tam tra ve `A`.

## Tu tra loi tren terminal

Mo terminal khac:

```powershell
cd F:\RAG_IR\hv1
python terminal_answer.py
```

Nhap `A`, `B`, `C`, hoac `D` cho tung cau. Bam Enter de bo qua, `q` de thoat.

## Cach 2: tra loi tu JSON

Sau khi da dien dap an:

```powershell
python server_answer.py
```

Khi Teacher gui cau hoi vao `/ask`, server se tim cau hoi trong `questions.json`
va tra ve dap an da luu. Neu chua co dap an, server luu cau hoi va tam tra ve `A`.

## Test nhanh

```powershell
curl -Method POST http://127.0.0.1:5003/ask `
  -ContentType "application/json" `
  -Body '{"question":"Thu do Viet Nam la gi?","options":{"A":"Ha Noi","B":"Hue","C":"Da Nang","D":"TPHCM"}}'
```
