"""LLMService: the only way the rest of the application talks to a model.

Responsibilities
- clean and bound the user's question before it reaches a prompt
- count model calls against a per-investigation budget
- parse every reply into a strict schema; on failure re-ask with the reasons (bounded repairs)
- for plans, also run the policy validator so the model sees WHY a plan was refused
- keep token totals and log one structured line per call (no prompts, replies or keys)

Create one LLMService per investigation; its counters are that investigation's totals.
"""

import logging
import re
import time
from dataclasses import dataclass, replace
from datetime import date

from pydantic import BaseModel, ValidationError

from app.analytics.validator import validate_plan
from app.llm.base import LLMClient, LLMRequest
from app.llm.errors import InvalidQuestionError, LLMBudgetExceededError, LLMOutputError
from app.llm.prompts import planner_system_prompt, planner_user_prompt, repair_suffix
from app.schemas.llm_outputs import PlannerResponse

logger = logging.getLogger(__name__)

_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")
_WS = re.compile(r"\s+")


def clean_question(question: str, *, max_length: int) -> str:
    """Normalise whitespace/control characters and enforce length. Raises InvalidQuestionError."""
    cleaned = _WS.sub(" ", _CONTROL.sub(" ", question or "")).strip()
    if not cleaned or len(cleaned) > max_length:
        raise InvalidQuestionError(f"The question must be between 1 and {max_length} characters.")
    return cleaned


@dataclass
class LLMUsageTotals:
    calls: int = 0
    tokens_in: int | None = None
    tokens_out: int | None = None
    model: str | None = None

    def add(self, tokens_in: int | None, tokens_out: int | None, model: str | None) -> None:
        self.calls += 1
        if tokens_in is not None:
            self.tokens_in = (self.tokens_in or 0) + tokens_in
        if tokens_out is not None:
            self.tokens_out = (self.tokens_out or 0) + tokens_out
        self.model = model or self.model


class LLMService:
    def __init__(
        self,
        client: LLMClient,
        *,
        max_calls: int,
        max_repairs: int,
        max_question_length: int,
        max_steps: int,
        max_total_queries: int,
    ) -> None:
        self._client = client
        self._max_calls = max_calls
        self._max_repairs = max_repairs
        self._max_question_length = max_question_length
        self._max_steps = max_steps
        self._max_total_queries = max_total_queries
        self.usage = LLMUsageTotals()

    # ------------------------------------------------------------------ planner
    def plan(
        self,
        question: str,
        *,
        data_start: date,
        data_end_exclusive: date,
        prior_findings: list[dict] | None = None,
    ) -> PlannerResponse:
        """Ask the model for a plan. Returns only a structurally valid reply; a returned plan
        has also passed policy validation. Raises LLMError subclasses otherwise."""
        cleaned = clean_question(question, max_length=self._max_question_length)
        request = LLMRequest(
            task="plan",
            system=planner_system_prompt(
                data_start=data_start,
                data_end_exclusive=data_end_exclusive,
                max_steps=self._max_steps,
                max_total_queries=self._max_total_queries,
            ),
            user=planner_user_prompt(cleaned, prior_findings),
            schema=PlannerResponse,
            context={
                "question": cleaned,
                "data_start": data_start,
                "data_end_exclusive": data_end_exclusive,
                "max_steps": self._max_steps,
                "max_total_queries": self._max_total_queries,
                "prior_findings": prior_findings or [],
            },
        )
        return self._ask(request, PlannerResponse, extra_check=self._check_plan)

    def _check_plan(self, reply: PlannerResponse) -> list[str]:
        if reply.outcome != "plan" or reply.plan is None:
            return []
        issues = validate_plan(
            reply.plan, max_steps=self._max_steps, max_total_queries=self._max_total_queries
        )
        return [f"{i.step_id or 'plan'}: {i.message}" for i in issues]

    # ------------------------------------------------------------------- engine
    def _ask[T: BaseModel](self, request: LLMRequest, schema: type[T], extra_check=None) -> T:
        base_user = request.user
        problems: list[str] = []
        for attempt in range(self._max_repairs + 1):
            if self.usage.calls >= self._max_calls:
                raise LLMBudgetExceededError()
            started = time.perf_counter()
            response = self._client.generate_json(request)  # may raise a typed LLMError
            self.usage.add(response.usage.tokens_in, response.usage.tokens_out, response.model)
            logger.info(
                "llm_call",
                extra={
                    "task": request.task,
                    "provider": self._client.provider,
                    "model": response.model or self._client.model,
                    "attempt": attempt + 1,
                    "duration_ms": round((time.perf_counter() - started) * 1000),
                    "tokens_in": response.usage.tokens_in,
                    "tokens_out": response.usage.tokens_out,
                },
            )
            try:
                parsed = schema.model_validate(response.data)
            except ValidationError as exc:
                problems = [
                    f"{'.'.join(str(p) for p in e['loc']) or 'reply'}: {e['msg']}"
                    for e in exc.errors(include_input=False, include_url=False)
                ]
            else:
                problems = extra_check(parsed) if extra_check else []
                if not problems:
                    return parsed
            logger.warning(
                "llm_reply_rejected",
                extra={
                    "task": request.task,
                    "attempt": attempt + 1,
                    "problem_count": len(problems),
                },
            )
            request = replace(request, user=base_user + repair_suffix(problems))
        raise LLMOutputError("The language model could not produce a valid answer.")
