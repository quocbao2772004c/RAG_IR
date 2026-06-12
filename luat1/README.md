# Student Legal RAG Server - luat1

Ban RAG theo format `v4`, toi uu cho du lieu phap luat giong `data/all_contexts.md`.

## Toi uu

- Tach theo tung Dieu, sau do tach theo khoan/diem neu noi dung dai.
- Moi chunk luon kem ten Dieu, Chu de, De muc, Nguon va ma lien ket cheo.
- BM25 index duoc dung mot lan khi upload/load, khong tokenize lai cho moi cau hoi.
- Boost ma Dieu, so van ban, acronym, thoi han, muc phat va cum tu chinh xac.
- Top-k rong de ho tro cau hoi can ket hop nhieu dieu.
- ChromaDB persistent va cache dap an cho cac lan evaluate lai.
- RAG thuong co literal/legal-rule verifier cho ngay, so tien, chu the, dieu kien phu dinh va cau multi-hop.
- Question bank tu `generated_questions*.json`, chi bat khi evaluate co flag `--map`.

Mac dinh dung `bm25` de upload bo du lieu luat lon nhanh hon timeout 2 phut.
Co the doi sang `hybrid` trong `.env` neu muon ket hop embedding.
Sau khi doi backend/model, chay evaluate khong co `--document-received` de tao lai ChromaDB.
Neu LLM server dung API key rieng, dat `LLM_API_KEY` thay vi dung chung `STUDENT_ID`.

## Cai dat

```bash
uv sync
cp .env.example .env
```

Neu dung `hybrid`, tai model truoc khi vao LAN:

```bash
uv run python download_model.py
```

## Chay

```bash
uv run python main.py
```

Terminal khac:

```bash
uv run python client.py register
uv run python client.py evaluate
uv run python client.py evaluate --document-received
uv run python client.py evaluate --document-received --map
uv run python client.py result
```

`evaluate --document-received` chay RAG thuong. Them `--map` thi client se bat mode map dap an tren server truoc khi goi evaluate.

Benchmark RAG thuong voi Qwen3.5-35B-A3B-GPTQ-Int4: `100/100` tren mau 100 cau tu bo du lieu hien tai.
