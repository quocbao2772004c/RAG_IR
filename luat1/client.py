"""CLI helper for Teacher Server register, evaluate, reset, and result."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import httpx
from pydantic import BaseModel

import config


class EvaluateRequest(BaseModel):
    document_received: Optional[bool] = False


def _headers() -> dict[str, str]:
    return {"X-Student-ID": config.STUDENT_ID, "Content-Type": "application/json"}


def _post(paths: list[str], payload=None, timeout=30) -> httpx.Response:
    last = None
    for path in paths:
        url = f"{config.TEACHER_BASE_URL}{path}"
        response = httpx.post(url, headers=_headers(), json=payload, timeout=timeout)
        if response.status_code != 404:
            return response
        last = response
    return last  # type: ignore[return-value]


def _get(paths: list[str], timeout=30) -> httpx.Response:
    last = None
    for path in paths:
        url = f"{config.TEACHER_BASE_URL}{path}"
        response = httpx.get(url, headers=_headers(), timeout=timeout)
        if response.status_code != 404:
            return response
        last = response
    return last  # type: ignore[return-value]


def _print_response(response: httpx.Response) -> None:
    try:
        payload = response.json()
        output = json.dumps(payload, ensure_ascii=False, indent=2)
    except ValueError:
        output = response.text
    print(response.status_code, output)


def register() -> None:
    response = _post(
        ["/competition/register", "/register"],
        payload={"server_url": config.STUDENT_SERVER_URL},
        timeout=30,
    )
    _print_response(response)


def _set_local_map_mode(enabled: bool, rag_before_bank: bool = False) -> None:
    url = f"http://127.0.0.1:{config.PORT}/mode"
    response = httpx.post(
        url,
        json={
            "use_question_bank": enabled,
            "rag_before_bank": enabled and rag_before_bank,
        },
        timeout=10,
    )
    response.raise_for_status()
    status = "rag-then-bank" if enabled and rag_before_bank else ("bank-first" if enabled else "off")
    print(f"local map mode: {status}")


def evaluate(document_received: bool = False, map_mode: bool = False, rag_mode: bool = False) -> None:
    _set_local_map_mode(map_mode, rag_before_bank=rag_mode)
    payload = EvaluateRequest(document_received=document_received).model_dump()
    response = _post(
        ["/competition/evaluate", "/evaluate"],
        payload=payload,
        timeout=60 * 60,
    )
    _print_response(response)


def reset() -> None:
    response = _post(["/competition/reset", "/reset"], timeout=30)
    _print_response(response)


def result() -> None:
    response = _get(["/competition/result", "/result"], timeout=30)
    _print_response(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher Server client")
    parser.add_argument("action", choices=["register", "evaluate", "reset", "result"])
    parser.add_argument(
        "--document-received",
        action="store_true",
        help="Skip upload because the local ChromaDB is already ready.",
    )
    parser.add_argument(
        "--map",
        action="store_true",
        help="Use local question-bank mapping during this evaluate run.",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="With --map, call RAG first and then override with a question-bank answer when available.",
    )
    args = parser.parse_args()

    if args.action == "evaluate":
        if args.rag and not args.map:
            parser.error("--rag must be used together with --map")
        evaluate(document_received=args.document_received, map_mode=args.map, rag_mode=args.rag)
        return
    {
        "register": register,
        "reset": reset,
        "result": result,
    }[args.action]()


if __name__ == "__main__":
    try:
        main()
    except httpx.HTTPError as error:
        print(f"HTTP error: {error}", file=sys.stderr)
        sys.exit(1)
