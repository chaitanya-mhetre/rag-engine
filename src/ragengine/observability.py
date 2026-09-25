"""Prometheus metrics and structured JSON logging with request ids."""

from __future__ import annotations

import contextvars
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from prometheus_client import Counter, Histogram
from starlette.requests import Request
from starlette.responses import Response

from ragengine.pipeline import QueryLog, QueryLogSink

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "Query pipeline latency per stage",
    ["stage"],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
LLM_TOKENS = Counter("rag_llm_tokens_total", "LLM tokens", ["model", "type"])
LLM_COST = Counter(
    "rag_llm_cost_usd_total", "Estimated LLM cost (USD, known prices only)", ["model"]
)
REFUSALS = Counter("rag_refusals_total", "Refused queries", ["reason"])
INVALID_CITATIONS = Counter("rag_invalid_citations_total", "Citations stripped by validation")
INJECTION_FLAGS = Counter(
    "rag_injection_flags_total", "Retrieved paragraphs dropped by promptguard"
)
INGESTION_JOBS = Counter("rag_ingestion_jobs_total", "Ingestion outcomes", ["status"])
HTTP_REQUESTS = Histogram(
    "rag_http_request_seconds", "HTTP request latency", ["method", "route", "status"]
)


class MetricsSink:
    """Wraps a QueryLogSink: persists the log, then records Prometheus metrics from it."""

    def __init__(self, inner: QueryLogSink) -> None:
        self.inner = inner

    async def write(self, log: QueryLog) -> None:
        await self.inner.write(log)
        for stage, ms in log.latency_ms.items():
            QUERY_LATENCY.labels(stage).observe(ms / 1000)
        LLM_TOKENS.labels(log.model, "prompt").inc(log.prompt_tokens)
        LLM_TOKENS.labels(log.model, "completion").inc(log.completion_tokens)
        if log.est_cost_usd:
            LLM_COST.labels(log.model).inc(log.est_cost_usd)
        if log.refused:
            REFUSALS.labels(log.refusal_reason or "unknown").inc()
        INVALID_CITATIONS.inc(len(log.invalid_citations))
        INJECTION_FLAGS.inc(log.injection_flags)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        for key in ("tenant_id", "query_log_id", "route", "status", "duration_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


access_log = logging.getLogger("ragengine.access")


async def request_context(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Middleware: request id (propagated from X-Request-ID if sent), timing, access log."""
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex
    token = request_id_var.set(rid)
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        route = getattr(request.scope.get("route"), "path", request.url.path)
        elapsed = time.perf_counter() - start
        HTTP_REQUESTS.labels(request.method, route, str(status)).observe(elapsed)
        access_log.info(
            "%s %s",
            request.method,
            route,
            extra={"route": route, "status": status, "duration_ms": round(elapsed * 1000, 2)},
        )
        request_id_var.reset(token)
