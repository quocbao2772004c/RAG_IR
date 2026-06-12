# Student RAG Server - v5 Structured Admissions

V5 toi uu cho tai lieu tuyen sinh giong `document.txt`: van ban dai, de muc danh so,
danh sach hoc ky va bang chi tieu co nhieu hang gan giong nhau.

## Diem khac v4

- Chunk theo cau truc, giu tieu de muc trong tung chunk.
- Moi hang bang duoc luu cung ten cot va ngu canh co so/loai chuong trinh.
- BM25 ho tro tieng Viet khong dau, boost ma nganh, con so va cum tu chinh xac.
- Hybrid mac dinh nghieng ve BM25 de tranh tron hang bang.
- ChromaDB persistent, restart server van dung lai vector DB.
- Model embedding load local-only khi chay server, khong doi HuggingFace trong LAN.

## Cai dat

```bash
uv sync
uv run python download_model.py
cp .env.example .env
```

Sua `.env`: `STUDENT_ID`, `STUDENT_SERVER_URL`, va `TEACHER_BASE_URL` neu can.

## Chay

```bash
uv run python main.py
```

Terminal khac:

```bash
uv run python client.py register
uv run python client.py evaluate
uv run python client.py evaluate --document-received
uv run python client.py result
```

Lan dau dung `evaluate` de nhan tai lieu. Sau khi ChromaDB da tao xong, dung
`evaluate --document-received` de Teacher Server chi gui 100 cau hoi.
