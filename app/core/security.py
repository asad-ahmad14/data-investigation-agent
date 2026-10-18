"""HTTP hardening: security headers, request body cap and a per-client rate limiter.

These are pure ASGI middlewares (not BaseHTTPMiddleware) so they work with streaming bodies and
background tasks.

Known limits, stated plainly:
- The rate limiter is in-memory and per process. With several workers each has its own counters,
  and it keys on the TCP peer address. Behind a reverse proxy every client appears as the proxy,
  so a deployment behind one should rate limit at the proxy (or add trusted X-Forwarded-For
  handling). We do not trust X-Forwarded-For here because anyone can forge it.
- There is no authentication. This is a local demo over synthetic data; anything public needs an
  auth layer in front of it.
"""

import time
from collections import deque
from collections.abc import Callable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

# The page is plain HTML/CSS/JS from the same origin: no inline scripts or styles, no CDNs.
PAGE_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)

BASE_HEADERS = {
    b"x-content-type-options": b"nosniff",
    b"referrer-policy": b"no-referrer",
    b"cross-origin-resource-policy": b"same-origin",
}


class SecurityHeadersMiddleware:
    """Adds standard headers; a strict CSP on the web page; no-store on investigation data."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path: str = scope["path"]

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {k.lower() for k, _ in headers}
                for key, value in BASE_HEADERS.items():
                    if key not in present:
                        headers.append((key, value))
                if path == "/" or path.startswith("/ui"):
                    headers.append((b"content-security-policy", PAGE_CSP.encode()))
                    headers.append((b"x-frame-options", b"DENY"))
                if path.startswith("/investigations"):
                    headers.append((b"cache-control", b"no-store"))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


class BodySizeLimitMiddleware:
    """Reject request bodies over `max_bytes` with 413, before they are parsed.

    Checks Content-Length up front and also counts bytes as they stream, so a chunked upload
    without a Content-Length cannot get around the limit.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None:
            try:
                too_big = int(declared) > self.max_bytes
            except ValueError:
                too_big = True
            if too_big:
                await self._reject(send)
                return

        received = 0
        exceeded = False
        responded = False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    exceeded = True
                    raise _BodyTooLargeError
            return message

        async def tracking_send(message: Message) -> None:
            # FastAPI turns a failed body read into its own 400. If the failure was our size
            # limit, answer 413 instead and drop whatever the application tries to send.
            nonlocal responded
            if exceeded:
                if not responded:
                    responded = True
                    await self._reject(send)
                return
            if message["type"] == "http.response.start":
                responded = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLargeError:
            if not responded:
                await self._reject(send)

    @staticmethod
    async def _reject(send: Send) -> None:
        body = b'{"error":{"code":"payload_too_large","message":"The request body is too large."}}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class _BodyTooLargeError(Exception):
    pass


class SlidingWindowLimiter:
    """At most `limit` events per `window` seconds per key. limit == 0 disables it."""

    def __init__(
        self,
        limit: int,
        window: float = 60.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ) -> None:
        self.limit = limit
        self.window = window
        self._clock = clock
        self._max_keys = max_keys
        self._events: dict[str, deque[float]] = {}

    def check(self, key: str) -> float | None:
        """Record an event. Returns None if allowed, else seconds to wait before retrying."""
        if self.limit <= 0:
            return None
        now = self._clock()
        events = self._events.get(key)
        if events is None:
            if len(self._events) >= self._max_keys:
                self._evict(now)
            events = self._events[key] = deque()
        while events and now - events[0] >= self.window:
            events.popleft()
        if len(events) >= self.limit:
            return max(0.1, self.window - (now - events[0]))
        events.append(now)
        return None

    def _evict(self, now: float) -> None:
        """Bound memory: drop idle keys, and if still full, the oldest half."""
        for key in [k for k, v in self._events.items() if not v or now - v[-1] >= self.window]:
            del self._events[key]
        if len(self._events) >= self._max_keys:
            for key in list(self._events)[: self._max_keys // 2]:
                del self._events[key]
