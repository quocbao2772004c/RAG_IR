"""CLI helper for Teacher Server register, evaluate, reset, and result."""
from __future__ import annotations

import argparse
import json
import os
import sys

import requests
from dotenv import load_dotenv


load_dotenv()


TEACHER_BASE_URL = os.getenv("TEACHER_BASE_URL", "http://192.168.50.218:8000/api/v1").rstrip("/")
STUDENT_ID = os.getenv("STUDENT_ID", "").strip().upper()
STUDENT_SERVER_URL = os.getenv("STUDENT_SERVER_URL", "").strip().rstrip("/")


def _require_env() -> None:
    missing = []
    if not STUDENT_ID:
        missing.append("STUDENT_ID")
    if not STUDENT_SERVER_URL:
        missing.append("STUDENT_SERVER_URL")
    if missing:
        raise SystemExit(f"Missing env var(s): {', '.join(missing)}. Copy .env.example to .env and edit it.")


def _headers() -> dict[str, str]:
    return {"X-Student-ID": STUDENT_ID, "Content-Type": "application/json"}


def _post(paths: list[str], payload=None, timeout=30) -> requests.Response:
    last = None
    for path in paths:
        url = f"{TEACHER_BASE_URL}{path}"
        response = requests.post(url, headers=_headers(), json=payload, timeout=timeout)
        if response.status_code != 404:
            return response
        last = response
    return last  # type: ignore[return-value]


def _get(paths: list[str], timeout=30) -> requests.Response:
    last = None
    for path in paths:
        url = f"{TEACHER_BASE_URL}{path}"
        response = requests.get(url, headers=_headers(), timeout=timeout)
        if response.status_code != 404:
            return response
        last = response
    return last  # type: ignore[return-value]


def _print_response(response: requests.Response) -> None:
    try:
        payload = response.json()
        output = json.dumps(payload, ensure_ascii=False, indent=2)
    except ValueError:
        output = response.text
    print(response.status_code, output)


def register() -> None:
    response = _post(
        ["/competition/register", "/register"],
        payload={"server_url": STUDENT_SERVER_URL},
        timeout=30,
    )
    _print_response(response)


def evaluate(document_received: bool = False) -> None:
    response = _post(
        ["/competition/evaluate", "/evaluate"],
        payload={"document_received": document_received},
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
        help="Skip upload because the local index is already ready.",
    )
    args = parser.parse_args()

    _require_env()
    if args.action == "evaluate":
        evaluate(document_received=args.document_received)
        return
    {
        "register": register,
        "reset": reset,
        "result": result,
    }[args.action]()


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as error:
        print(f"HTTP error: {error}", file=sys.stderr)
        sys.exit(1)
