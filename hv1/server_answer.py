"""FastAPI server that answers Teacher by reading answers from questions.json."""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request

from cache_store import CACHE_PATH, extract_question, find_answer, upsert_question


app = FastAPI(title="HV1 Cached Answer Server", version="1.0.0")


async def json_or_text(request: Request) -> dict[str, Any]:
    body = await request.body()
    if not body:
        return {}
    try:
        data = await request.json()
    except Exception:
        return {"text": body.decode("utf-8", errors="ignore")}
    return data if isinstance(data, dict) else {"text": str(data)}


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "mode": "answer-from-json", "cache": str(CACHE_PATH)}


@app.post("/ask")
async def ask(request: Request) -> dict[str, Any]:
    payload = await json_or_text(request)
    question = extract_question(payload)
    if not question:
        raise HTTPException(status_code=400, detail="empty question")

    item = upsert_question(payload)
    answer, cached = find_answer(question)
    if not answer:
        return {
            "answer": "A",
            "cached": False,
            "id": item["id"],
            "message": "No manual answer yet in questions.json",
        }
    return {"answer": answer, "cached": True, "id": cached["id"] if cached else item["id"]}


@app.post("/upload")
async def upload() -> dict[str, str]:
    return {"status": "ignored"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server_answer:app", host="0.0.0.0", port=5003, reload=False)
