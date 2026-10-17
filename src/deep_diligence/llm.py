"""The provider boundary: one function returns the chat model.

offline (default) is the labelled deterministic stub. Real providers
need the `llm` extra (langchain-openai / langchain-anthropic) and one
provider's environment set — see .env.example. Any OpenAI-compatible
endpoint works, including NVIDIA NIM (hosted or self-hosted) and
Ollama's OpenAI-compatible surface.
"""

from __future__ import annotations

from .config import Settings
from .stub_model import ScriptedDiligenceModel


def build_model(settings: Settings):
    backend = settings.llm_backend
    if backend == "offline":
        return ScriptedDiligenceModel()
    try:
        if backend == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
        if backend == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=settings.anthropic_model,
                api_key=settings.anthropic_api_key,
            )
        if backend == "ollama":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                api_key="ollama",
            )
    except ImportError as exc:
        raise RuntimeError(
            "The real-provider backends need the llm extra: "
            "uv sync --extra llm (or pip install -e '.[llm]')"
        ) from exc
    raise ValueError(f"unknown LLM_BACKEND: {backend!r}")
