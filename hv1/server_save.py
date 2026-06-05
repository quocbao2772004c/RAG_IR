"""FastAPI server that only receives Teacher questions and saves them to JSON."""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request

from cache_store import CACHE_PATH, upsert_question


app = FastAPI(title="HV1 Question Saver", version="1.0.0")


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
    return {"status": "ok", "mode": "save-only", "cache": str(CACHE_PATH)}


@app.post("/ask")
async def ask(request: Request) -> dict[str, str]:
    payload = await json_or_text(request)
    try:
        item = upsert_question(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"answer": "A", "saved": "true", "id": item["id"]}


@app.post("/upload")
async def upload() -> dict[str, str]:
    return {"status": "ignored"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server_save:app", host="0.0.0.0", port=5003, reload=False)
