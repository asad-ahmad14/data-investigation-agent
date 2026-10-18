"""Ask the configured LLM (mock by default) to plan an investigation and print the result.

    python scripts/try_planner.py "Why did net revenue drop in June 2025?"

No database is needed. The plan is validated against the registry exactly as the real
workflow will do it. With LLM_PROVIDER=gemini this makes ONE real API call using your key.
"""

import json
import sys
from datetime import date

from app.core.config import Settings
from app.llm.errors import LLMError
from app.llm.factory import build_llm_service
from app.retail.generate import FIRST_MONTH, N_MONTHS, add_months


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python scripts/try_planner.py "your question"')
        return 2
    settings = Settings()
    service = build_llm_service(settings)
    start: date = FIRST_MONTH
    end: date = add_months(FIRST_MONTH, N_MONTHS)
    print(f"provider={settings.llm_provider} model={service._client.model}")
    try:
        reply = service.plan(sys.argv[1], data_start=start, data_end_exclusive=end)
    except LLMError as exc:
        print(f"LLM error ({type(exc).__name__}): {exc.public_message}")
        return 1
    print(json.dumps(reply.model_dump(mode="json", exclude_none=True), indent=2))
    u = service.usage
    print(f"calls={u.calls} tokens_in={u.tokens_in} tokens_out={u.tokens_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
