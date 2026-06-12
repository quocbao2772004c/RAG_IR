# Offline Hybrid RAG Student Server

Server nay khong goi cloud API. No tra loi `/ask` bang:

- v5 mac dinh: ensemble v1-v4 + local reranker `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`;
- hybrid retrieval BM25 + TF-IDF;
- khong load QA cache mac dinh de tranh trick.

## Chay server

```bash
cd /home/anonymous/code/IR/final
python -m uvicorn main:app --host 0.0.0.0 --port 5000
```

Kiem tra:

```bash
curl http://127.0.0.1:5000/health
```

Mac dinh server preload `data_ir/all_contexts.md`, dung `RAG_VERSION=v5`, tat semantic index nang, nhung van bat local semantic reranker.

Khi Teacher goi `/upload`, index se duoc build lai tu payload Teacher gui.

## Tuy chon version

```bash
RAG_VERSION=v1 python -m uvicorn main:app --host 0.0.0.0 --port 5000
RAG_VERSION=v5 python -m uvicorn main:app --host 0.0.0.0 --port 5000
```

Version:

- `v1`: baseline hybrid BM25 + TF-IDF selector.
- `v2`: option window scorer.
- `v3`: question-focused window scorer.
- `v4`: lexical ensemble.
- `v5`: calibrated local ensemble + sentence-transformers reranker.
- `v6`: v5 fallback + optional local OpenAI-compatible LLM.
- `v7`: v6 + local LLM self-verification pass.

Neu co Qwen local dang chay OpenAI-compatible:

```bash
RAG_VERSION=v7 \
RAG_LOCAL_LLM_BASE_URL=http://127.0.0.1:8000/v1 \
RAG_LOCAL_LLM_MODEL=Qwen3.5-35B-A3B-GPTQ-Int4 \
python -m uvicorn main:app --host 0.0.0.0 --port 5000
```

## QA cache

Mac dinh cache tat. Chi bat khi can debug/trick tren file mau:

```bash
RAG_USE_QA_CACHE=1 python -m uvicorn main:app --host 0.0.0.0 --port 5000
```

## Semantic index

```bash
RAG_USE_INDEX_SEMANTIC=1 python -m uvicorn main:app --host 0.0.0.0 --port 5000
```

## Test offline

Test that, khong dung answer cache:

```bash
python test.py --questions data_ir/generated_questions_2.json --no-answer-cache --version v5 --no-index-semantic
```

Ket qua benchmark hien tai tren 490 cau:

- `v5`: 354/490 = 72.245%
- Level 1: 191/248 = 77.016%
- Level 2: 96/145 = 66.207%
- Level 3: 67/97 = 69.072%

## API dung cho Teacher

`POST /upload`

```json
{
  "doc_id": "none",
  "text": "noi dung tai lieu..."
}
```

Response:

```json
{
  "status": "success",
  "doc_id": "none",
  "chunks": 42
}
```

`POST /ask`

```json
{
  "question": "Cau hoi... A. ... B. ... C. ... D. ..."
}
```

Response luon co `answer` la mot chu cai:

```json
{
  "answer": "B",
  "sources": ["..."]
}
```

Neu Teacher gui options tach rieng, server cung nhan duoc:

```json
{
  "question": "Cau hoi...",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."]
}
```

## Goi Teacher bang client.py

Dang ky Student Server:

```bash
python client.py register --student-id B21DCCN629 --server-url http://<IP-LAN>:5000
```

Evaluate lan dau, de Teacher upload document:

```bash
python client.py evaluate --student-id B21DCCN629 --document-received false
```

Evaluate cac lan sau neu server da luu index:

```bash
python client.py evaluate --student-id B21DCCN629 --document-received true
```

Register roi evaluate mot lenh:

```bash
python client.py run --student-id B21DCCN629 --server-url http://<IP-LAN>:5000
```

Xem diem/reset:

```bash
python client.py result --student-id B21DCCN629
python client.py reset --student-id B21DCCN629
```
