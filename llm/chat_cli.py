"""Simple CLI chat using the OpenAI-compatible model configured in .env."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ENV_PATH = Path(__file__).with_name(".env")


def load_settings() -> dict[str, str | float | int | None]:
    load_dotenv(ENV_PATH)

    teacher_base_url = os.getenv("TEACHER_BASE_URL", "http://192.168.50.218:8000/api/v1").rstrip("/")
    base_url = os.getenv("LLM_BASE_URL", f"{teacher_base_url}/proxy").rstrip("/")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("STUDENT_ID") or os.getenv("API_KEY")
    model = os.getenv("LLM_MODEL") or os.getenv("MODEL") or "gpt-4o-mini"

    return {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
        "timeout": float(os.getenv("LLM_TIMEOUT", "45")),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "1024")),
        "system_prompt": os.getenv(
            "LLM_SYSTEM_PROMPT",
            "You are a helpful assistant. Answer clearly and concisely.",
        ),
    }


def build_client(settings: dict[str, str | float | int | None]) -> OpenAI:
    if not settings["api_key"]:
        raise RuntimeError(
            "Missing API key. Put STUDENT_ID=..., OPENAI_API_KEY=..., or API_KEY=... in llm/.env"
        )
    return OpenAI(
        base_url=str(settings["base_url"]),
        api_key=str(settings["api_key"]),
        timeout=float(settings["timeout"]),
        max_retries=0,
    )


def chat_once(
    client: OpenAI,
    settings: dict[str, str | float | int | None],
    messages: list[dict[str, str]],
) -> str:
    response = client.chat.completions.create(
        model=str(settings["model"]),
        temperature=float(settings["temperature"]),
        max_tokens=int(settings["max_tokens"]),
        messages=messages,
    )
    return response.choices[0].message.content or ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat with the model configured in llm/.env")
    parser.add_argument("--message", "-m", help="Send one message and exit.")
    args = parser.parse_args()

    settings = load_settings()
    try:
        client = build_client(settings)
    except Exception as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    messages = [{"role": "system", "content": str(settings["system_prompt"])}]

    if args.message:
        messages.append({"role": "user", "content": args.message})
        try:
            print(chat_once(client, settings, messages))
            return 0
        except Exception as exc:
            print(f"LLM error: {exc}", file=sys.stderr)
            return 1

    print(f"Model: {settings['model']}")
    print("Type /exit to quit, /clear to reset history.")

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "exit", "quit", "q"}:
            return 0
        if user_text.lower() == "/clear":
            messages = [{"role": "system", "content": str(settings["system_prompt"])}]
            print("History cleared.")
            continue

        messages.append({"role": "user", "content": user_text})
        try:
            answer = chat_once(client, settings, messages)
        except Exception as exc:
            messages.pop()
            print(f"LLM error: {exc}", file=sys.stderr)
            continue

        messages.append({"role": "assistant", "content": answer})
        print(f"\nAssistant: {answer}")


if __name__ == "__main__":
    raise SystemExit(main())
