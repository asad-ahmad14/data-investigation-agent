"""Build the configured LLM client and per-investigation service from Settings."""

from app.core.config import Settings
from app.llm.base import LLMClient
from app.llm.errors import LLMConfigError
from app.llm.gemini import GeminiClient
from app.llm.mock import MockLLM
from app.llm.service import LLMService


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "mock":
        return MockLLM()
    if settings.llm_provider == "gemini":
        return GeminiClient(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            max_output_tokens=settings.llm_max_output_tokens,
        )
    raise LLMConfigError(f"Unknown LLM provider '{settings.llm_provider}'.")


def build_llm_service(settings: Settings, client: LLMClient | None = None) -> LLMService:
    return LLMService(
        client or build_llm_client(settings),
        max_calls=settings.max_llm_calls,
        max_repairs=settings.llm_max_repairs,
        max_question_length=settings.max_question_length,
        max_steps=settings.max_steps_per_plan,
        max_total_queries=settings.max_total_queries,
    )
