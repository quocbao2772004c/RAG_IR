from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from typing import Any

from dotenv import load_dotenv
import requests


load_dotenv()


DEFAULT_TEACHER_BASE_URL = "http://192.168.50.218:8000/api/v1"
DEFAULT_PORT = 5000


def guess_lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        sock.close()


def bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "co", "có"}:
        return True
    if lowered in {"0", "false", "no", "n", "khong", "không"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false.")


class TeacherClient:
    def __init__(self, base_url: str, student_id: str, timeout: int = 300) -> None:
        self.base_url = base_url.rstrip("/")
        self.student_id = student_id.upper().strip()
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "X-Student-ID": self.student_id,
            }
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        response = self.session.request(method, url, timeout=self.timeout, **kwargs)
        try:
            payload = response.json()
        except Exception:
            payload = response.text
        if not response.ok:
            raise SystemExit(
                f"{method} {url} failed: HTTP {response.status_code}\n"
                f"{json.dumps(payload, ensure_ascii=False, indent=2) if isinstance(payload, (dict, list)) else payload}"
            )
        return payload

    def register(self, server_url: str) -> Any:
        return self._request(
            "POST",
            "/competition/register",
            json={"server_url": server_url.rstrip("/")},
        )

    def evaluate(self, document_received: bool = False) -> Any:
        return self._request(
            "POST",
            "/competition/evaluate",
            json={"document_received": document_received},
        )

    def reset(self) -> Any:
        return self._request("POST", "/competition/reset")

    def result(self) -> Any:
        return self._request("GET", "/competition/result")


def print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Client gọi Teacher Server cho bài RAG competition."
    )
    parser.add_argument(
        "command",
        choices=["register", "evaluate", "reset", "result", "run"],
        help="run = register rồi evaluate.",
    )
    parser.add_argument(
        "--teacher",
        default=os.getenv("TEACHER_BASE_URL", DEFAULT_TEACHER_BASE_URL),
        help=f"Teacher base URL. Default: {DEFAULT_TEACHER_BASE_URL}",
    )
    parser.add_argument(
        "--student-id",
        default=os.getenv("STUDENT_ID", ""),
        help="Mã sinh viên viết hoa, hoặc set env STUDENT_ID.",
    )
    parser.add_argument(
        "--server-url",
        default=os.getenv("SERVER_URL", os.getenv("STUDENT_SERVER_URL", "")),
        help="URL Student Server, ví dụ http://192.168.1.15:5000.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--document-received",
        type=bool_arg,
        default=False,
        help="True nếu Teacher đã upload document trước đó và server đã lưu index.",
    )
    parser.add_argument("--timeout", type=int, default=300)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.student_id:
        raise SystemExit("Thiếu --student-id hoặc env STUDENT_ID.")

    server_url = args.server_url.strip()
    if not server_url:
        server_url = f"http://{guess_lan_ip()}:{args.port}"

    client = TeacherClient(args.teacher, args.student_id, timeout=args.timeout)

    if args.command == "register":
        print_json(client.register(server_url))
    elif args.command == "evaluate":
        print_json(client.evaluate(document_received=args.document_received))
    elif args.command == "reset":
        print_json(client.reset())
    elif args.command == "result":
        print_json(client.result())
    elif args.command == "run":
        print("Registering student server...")
        print_json(client.register(server_url))
        print("Evaluating...")
        print_json(client.evaluate(document_received=args.document_received))
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"Request error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
