"""
Cloud compatibility adapter.

The original RAG project used Ollama locally.
This module keeps the existing ollama.chat(),
ollama.embeddings(), and ollama.list() interfaces
while using Google's Gemini API in the cloud.
"""

import os
import streamlit as st

from google import genai
from google.genai import types


CHAT_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"


@st.cache_resource
def get_client():
    api_key = None

    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

    api_key = api_key or os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. "
            "Add it to Streamlit Cloud Secrets."
        )

    return genai.Client(api_key=api_key)


def _build_prompt(messages):
    parts = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        parts.append(
            f"{role.upper()}:\n{content}"
        )

    return "\n\n".join(parts)


def chat(model, messages, stream=False):
    client = get_client()

    prompt = _build_prompt(messages)

    if stream:
        for chunk in client.models.generate_content_stream(
            model=CHAT_MODEL,
            contents=prompt,
        ):
            if chunk.text:
                yield {
                    "message": {
                        "content": chunk.text
                    }
                }

    else:
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt,
        )

        return {
            "message": {
                "content": response.text or ""
            }
        }


def embeddings(model, prompt):
    client = get_client()

    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=prompt,
        config=types.EmbedContentConfig(
            output_dimensionality=768
        ),
    )

    return {
        "embedding": response.embeddings[0].values
    }


def list():
    return {
        "models": [
            {
                "name": CHAT_MODEL
            },
            {
                "name": EMBED_MODEL
            }
        ]
    }
