"""Shared helpers for DTUMOS LLM agents."""
from __future__ import annotations

from typing import Any

import ollama
from langchain.prompts import ChatPromptTemplate

DEFAULT_MODEL = "llama3.1"


def stream_agent_response(
    prompt_template: str,
    *,
    model: str = DEFAULT_MODEL,
    **prompt_variables: Any,
) -> None:
    """Render the prompt with the provided variables and stream the model output."""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    rendered = prompt.format(**prompt_variables)
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": rendered}],
        stream=True,
    )
    for chunk in response:
        content = chunk.get("message", {}).get("content", "")
        if content:
            print(content, end="", flush=True)


def preview_prompt(prompt_template: str, **prompt_variables: Any) -> str:
    """Utility to inspect the final prompt before sending it to the model."""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return prompt.format(**prompt_variables)
