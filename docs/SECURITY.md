# Security notes

This is a portfolio project over a synthetic dataset, run locally. The notes below say what is
protected, how that is enforced, what was found during the Phase 8 audit, and what is **not**
covered. Nothing here is a claim of production readiness.

## Threat model

| Threat | Where it could enter | Main defence |
| --- | --- | --- |
| SQL injection | the question, model output, filter values | The model never writes SQL. It produces a structured plan that is validated against a registry; values are bound parameters; the database role is read-only. |
| Prompt injection | the question text | The question is fenced as data in the prompt, but the real protection is downstream: strict schema, registry validation, and a role that can only `SELECT` approved tables. A fully "jailbroken" model still cannot name a table or run a statement. |
| Data theft or tampering through the database | a bug or bypass in the app layer | Two separate database roles with least privilege (below). |
| Cross-site scripting | model or user text rendered in the page | The page only uses `textContent`; a strict Content-Security-Policy forbids inline and third-party scripts. A test fails the build if `innerHTML` appears. |
| Resource exhaustion | many or huge requests, slow queries, runaway model loops | Body size cap, per-client rate limit, cap on concurrent investigations, statement timeout, row cap, and per-investigation limits on rounds, queries, model calls and wall-clock time. |
| Secret leakage | logs, errors, repr, git | Secrets are `SecretStr`; the API key is sent in a header (never a URL); error bodies are generic; `.env` is git- and docker-ignored. |

## Database roles

| Role | Can | Cannot |
| --- | --- | --- |
| `analyst_ro` (runs analysis queries) | `SELECT` on seven approved `retail` tables | write, create or drop anything, read the `app` schema, read server files, run programs, switch role, create roles |
| `app_rw` (saves history) | read/write the `app` schema | read `retail`, alter or drop tables, create objects, read server files, switch role |
| admin (migrations, seeding) | everything | used only by migration and seed scripts, never while serving requests |

Neither runtime role is superuser, can create databases or roles, replicate, or bypass row security
(tested). Read-only is also set as a session default, but a role can switch that off for its own
session, so the tests prove the privileges block writes even when it is off.

## Controls added in the Phase 8 audit

| # | Finding (how it was found) | Fix | Test |
| --- | --- | --- | --- |
| F1 | No security headers (curl on the live server) | `X-Content-Type-Options`, `Referrer-Policy`, `Cross-Origin-Resource-Policy`; strict CSP and `X-Frame-Options: DENY` on the page; `Cache-Control: no-store` on investigation data | `test_security.py` |
| F2 | A 3 MB request body was accepted and parsed | 16 KB body cap checked from `Content-Length` and while streaming; answers 413 | `test_security.py` |
| F3 | A burst of 60 requests created 33 investigations (history could be flooded) | per-client sliding-window rate limit (default 20 per minute) with `Retry-After` | `test_security.py` |
| F4 | API schema and docs exposed in production mode | `/docs` and `/openapi.json` disabled when `APP_ENV=production` | `test_security.py` |
| F5 | Streamed oversized bodies returned 400 instead of 413 (caught by the new test) | respond 413 even when the framework wraps the read error | `test_security.py` |
| F6 | The database port and API port were published on all network interfaces | bound to `127.0.0.1` in `docker-compose.yml` | `test_security.py` |
| F7 | A NUL character in model-written text would be refused by PostgreSQL and fail the save | control characters stripped from model-produced text | `test_security.py` |
| F8 | No `.dockerignore` | added, excluding `.env`, tests and virtualenv | `test_security.py` |

Also checked and found fine: path traversal under `/ui/` (404), cross-site `text/plain` form posts
(rejected, JSON required), dependency audit of the pinned versions with `pip-audit` (no known
vulnerabilities on the date it was run), SQL in filter values (bound as data), model naming
tables or columns (rejected before any query runs).

## Not covered (known limitations)

- **No authentication or authorization.** Anyone who can reach the server can run investigations
  and read history. Fine on `localhost`; anything public needs an auth layer first.
- **Rate limiting is in memory and per process**, keyed on the connection address. Behind a
  reverse proxy every client looks the same; limit at the proxy instead. `X-Forwarded-For` is
  deliberately not trusted.
- **No TLS** is terminated here; use a proxy for HTTPS.
- **Prompt-injection resistance was exercised with the mock model and scripted hostile replies**,
  not a live model. The structural defences hold regardless of the model, but the model's own
  behaviour under attack has not been measured.
- **Real LLM provider code is tested with mocked HTTP only.** It has not been run against the live
  API from the development sandbox.
- Database passwords live in `.env` (fine locally); use a secret manager in a real deployment.
- PostgreSQL lets every role read parts of `pg_catalog`. That cannot be revoked safely; the query
  compiler never emits catalog queries.
- The dependency audit is a point-in-time check. Re-run `pip-audit -r requirements.lock`
  regularly.
- Logs can contain the exact SQL text and bound values of failed queries (server side only; never
  returned to clients).
