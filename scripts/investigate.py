"""Run one investigation from the command line and print the report.

    python scripts/investigate.py "Why did net revenue drop?"
    python scripts/investigate.py "Why did the refund rate rise in October 2025?" --trace

Needs the database running and seeded (see README). Uses LLM_PROVIDER from .env (mock by
default, which needs no API key). Add --json to print the full report as JSON.
"""

import json
import sys

from app.agent.runner import run_investigation
from app.core.config import Settings
from app.db.engines import build_analyst_engine


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if not args:
        print('Usage: python scripts/investigate.py "your question" [--trace] [--json]')
        return 2
    settings = Settings()
    engine = build_analyst_engine(settings)
    try:
        report, events = run_investigation(args[0], settings=settings, analyst_engine=engine)
    finally:
        engine.dispose()
    if "--json" in flags:
        print(json.dumps(report.model_dump(mode="json"), indent=2, default=str))
    else:
        print(report.to_text())
    if "--trace" in flags:
        print("\nTRACE")
        for e in events:
            print(f"- {e['node']:<9} {e['status']:<10} {e['duration_ms']:>5} ms  {e['detail']}")
    return 0 if report.status in ("completed", "insufficient_evidence") else 1


if __name__ == "__main__":
    raise SystemExit(main())
