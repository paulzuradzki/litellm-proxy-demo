#!/usr/bin/env python3
"""LiteLLM demo — script version of notebooks/demo.ipynb.

Usage:
    uv run scripts/demo.py
"""

import os
import textwrap

import httpx
import litellm
from dotenv import load_dotenv

# repo root, regardless of where this script is invoked from
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

API_KEY = os.environ["LITELLM_API_KEY"]
BASE_URL = os.environ["LITELLM_BASE_URL"]
MODEL = "local/qwen3.8-27b-5090"

# litellm needs a provider prefix for custom proxy model names
LITELLM_MODEL = f"openai/{MODEL}"


def main() -> None:
    print(f"base_url : {BASE_URL}")
    print(f"model    : {MODEL}\n")

    # List models exposed by the proxy
    r = httpx.get(
        BASE_URL + "/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=30,
    )
    r.raise_for_status()
    models = r.json()["data"]
    print(f"{len(models)} model(s) available:\n")
    for m in models:
        print(f"  {m.get('id'):40s}  (owned_by={m.get('owned_by', '?')})")

    # Hello world
    user_query = (
        "Write me a 2 paragraph intro to getting started with local AI. "
        "What's an iGPU vs dGPU? Pros, cons? MoE vs Dense?"
    )
    print()
    print("user>")
    print(textwrap.fill(user_query, width=80, initial_indent="  ", subsequent_indent="  "))

    resp = litellm.completion(
        model=LITELLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a concise, friendly assistant."},
            {"role": "user", "content": user_query},
        ],
        api_base=BASE_URL,
        api_key=API_KEY,
    )

    print()
    print(f"assistant> (model {MODEL})")
    print(textwrap.fill(resp.choices[0].message.content, width=80, initial_indent="  ", subsequent_indent="  "))
    print(
        f"\n[usage] {resp.usage.prompt_tokens} prompt / "
        f"{resp.usage.completion_tokens} completion tokens"
    )


if __name__ == "__main__":
    main()
