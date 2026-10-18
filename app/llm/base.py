"""Provider-neutral request/response types and the client interface."""

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel


@dataclass(frozen=True)
class LLMUsage:
    tokens_in: int | None = None
    tokens_out: int | None = None


@dataclass(frozen=True)
class LLMRequest:
    """One model call.

    `system` and `user` are the text a real model sees. `context` is the same information in
    structured form, used only by the deterministic mock provider; real providers ignore it.
    """

    task: str
    system: str
    user: str
    schema: type[BaseModel]
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMResponse:
    data: dict[str, Any]  # parsed JSON object, NOT yet validated against the schema
    model: str
    usage: LLMUsage = field(default_factory=LLMUsage)


class LLMClient(Protocol):
    provider: str
    model: str

    def generate_json(self, request: LLMRequest) -> LLMResponse:
        """Return the model's JSON reply. Raises an LLMError subclass on failure."""
        ...
