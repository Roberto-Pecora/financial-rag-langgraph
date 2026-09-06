"""LLM access via LangChain ChatOpenAI pointed at OpenRouter.

`chat_model(role)` returns a configured ChatOpenAI for a role's model. Typed
output goes through `structured_model`; ported pure helpers that expect a
`generate(prompt) -> str` surface use `GenerateAdapter`.
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


def structured_model(role: str, schema: type) -> Any:
    """Typed output via json_mode.

    The default function_calling method is unreliable on open OpenRouter models,
    which often echo the schema instead of an instance; json_mode is steadier and
    matches what the prompts already ask for ("Return ONLY valid JSON").
    """
    return chat_model(role).with_structured_output(schema, method="json_mode")


class GenerateAdapter:
    """Expose a chat model as `generate(prompt) -> str` for framework-free helpers."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def generate(self, prompt: str) -> str:
        return self._model.invoke(prompt).content
