"""Streamlit chat UI using the OpenAI-compatible model configured in .env."""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


ENV_PATH = Path(__file__).with_name(".env")


def load_settings() -> dict[str, str | float | int | None]:
    load_dotenv(ENV_PATH)

    teacher_base_url = os.getenv("TEACHER_BASE_URL", "http://192.168.50.218:8000/api/v1").rstrip("/")
    base_url = os.getenv("LLM_BASE_URL", f"{teacher_base_url}/proxy").rstrip("/")
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("STUDENT_ID")
        or os.getenv("API_KEY")
    )
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


@st.cache_resource
def build_client(base_url: str, api_key: str, timeout: float) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)


def main() -> None:
    st.set_page_config(page_title="LLM Chat", page_icon="💬", layout="centered")
    settings = load_settings()

    st.title("LLM Chat")
    st.caption(f"Model: `{settings['model']}`")

    with st.sidebar:
        st.subheader("Settings")
        st.text_input("Base URL", value=str(settings["base_url"]), disabled=True)
        st.text_input("Model", value=str(settings["model"]), disabled=True)
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = [
                {"role": "system", "content": str(settings["system_prompt"])}
            ]
            st.rerun()

    if not settings["api_key"]:
        st.error("Missing API key. Put LLM_API_KEY=..., STUDENT_ID=..., OPENAI_API_KEY=..., or API_KEY=... in llm/.env")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": str(settings["system_prompt"])}
        ]

    for message in st.session_state.messages:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask something")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = build_client(
        str(settings["base_url"]),
        str(settings["api_key"]),
        float(settings["timeout"]),
    )

    with st.chat_message("assistant"):
        try:
            response = client.chat.completions.create(
                model=str(settings["model"]),
                temperature=float(settings["temperature"]),
                max_tokens=int(settings["max_tokens"]),
                messages=st.session_state.messages,
            )
            answer = response.choices[0].message.content or ""
        except Exception as exc:
            answer = f"LLM error: {exc}"
            st.error(answer)
        else:
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
