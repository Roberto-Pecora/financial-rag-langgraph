"""LLM access via LangChain ChatOpenAI pointed at OpenRouter.

`chat_model(role)` returns a configured ChatOpenAI for a role's model. Graph nodes
that need typed output call `.with_structured_output(Model)`; ported pure helpers
that expect a `generate(prompt) -> str` surface use `GenerateAdapter`.
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from frag.utils.settings import settings

_MODEL_FOR = {
    "actor": lambda: settings.actor_model,
    "critic": lambda: settings.critic_model,
    "grader": lambda: settings.grader_model,
    "agent": lambda: settings.agent_model,
}


def chat_model(role: str, **overrides: Any) -> ChatOpenAI:
    """A ChatOpenAI bound to the role's model and the OpenRouter endpoint."""
    model = _MODEL_FOR.get(role, lambda: settings.actor_model)()
    params: dict[str, Any] = {
        "model": model,
        "base_url": settings.openrouter_base_url,
        "api_key": settings.openrouter_api_key or "missing",
        "temperature": settings.llm_temperature,
        "timeout": settings.llm_timeout_s,
        "max_retries": settings.llm_max_retries,
    }
    params.update(overrides)
    return ChatOpenAI(**params)


class GenerateAdapter:
    """Expose a chat model as `generate(prompt) -> str` for framework-free helpers."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def generate(self, prompt: str) -> str:
        return self._model.invoke(prompt).content
