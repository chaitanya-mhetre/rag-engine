"""LLM providers behind one interface.

`FakeExtractiveLLM` is the offline default: it reads the prompt we generated, picks the source
sentences with the best overlap with the question, and answers with citations in the requested
format (or says INSUFFICIENT_CONTEXT). It is deterministic and free, so the whole pipeline and
evaluation run in CI. It does not reason. Treat its outputs as plumbing tests, not quality.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from ragengine.generation.prompts import CONDENSE, INSUFFICIENT
from ragengine.ingestion.tokenizer import RegexTokenizer
from ragengine.retrieval.bm25 import analyze

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    usage_estimated: bool = False  # True when token counts are approximations


class LLM(Protocol):
    model: str

    async def complete(
        self, messages: list[Message], *, json_mode: bool = False
    ) -> LLMResponse: ...

    def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Yield text deltas. After exhaustion, `last_usage` holds the final LLMResponse."""
        ...

    @property
    def last_usage(self) -> LLMResponse | None: ...


_SOURCE_RE = re.compile(r'<source id="(S\d+)"[^>]*>\n(.*?)\n</source>', re.DOTALL)
_QUESTION_RE = re.compile(r"<question>(.*?)</question>", re.DOTALL)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


class FakeExtractiveLLM:
    def __init__(self, model: str = "fake-extractive-v1", min_coverage: float = 0.34) -> None:
        self.model = model
        self.min_coverage = min_coverage
        self._tok = RegexTokenizer()
        self._last: LLMResponse | None = None

    @property
    def last_usage(self) -> LLMResponse | None:
        return self._last

    def _condense(self, messages: list[Message]) -> str:
        convo = messages[-1]["content"]
        lines = [ln for ln in convo.splitlines() if ln.startswith("user:")]
        follow_up = convo.rsplit("Follow-up:", 1)[-1].strip()
        if not lines:
            return follow_up
        previous = lines[-1].removeprefix("user:").strip()
        return f"{follow_up} ({previous})"

    def _answer(self, prompt: str) -> tuple[str, list[tuple[str, str]], float]:
        match = _QUESTION_RE.search(prompt)
        question = match.group(1) if match else ""
        q_terms = set(analyze(question))
        scored: list[tuple[float, int, str, str, set[str]]] = []
        for order, (label, text) in enumerate(_SOURCE_RE.findall(prompt)):
            for sentence in _SENTENCE_RE.split(text):
                s_terms = set(analyze(sentence))
                if not s_terms or sentence.lstrip().startswith("#"):
                    continue
                cov = len(q_terms & s_terms) / len(q_terms) if q_terms else 0.0
                scored.append((cov, -order, label, sentence.strip(), q_terms & s_terms))
        if not scored:
            return INSUFFICIENT, [], 0.0
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        best = scored[0]
        if best[0] < self.min_coverage:
            return INSUFFICIENT, [], best[0]
        picked = [best]
        covered = set(best[4])
        # multi-hop: add one more sentence if it covers question terms the first one missed
        for cand in scored[1:]:
            if cand[4] - covered and cand[3] != best[3]:
                picked.append(cand)
                covered |= cand[4]
                break
        answer = " ".join(f"{s[3].rstrip('.')} [{s[2]}]." for s in picked)
        confidence = round(len(covered) / len(q_terms), 3) if q_terms else 0.0
        return answer, [(s[2], s[3][:160]) for s in picked], confidence

    def _render(self, messages: list[Message], json_mode: bool) -> str:
        if messages and messages[0]["content"] == CONDENSE:
            return self._condense(messages)
        answer, citations, confidence = self._answer(messages[-1]["content"])
        if not json_mode:
            return answer
        insufficient = answer == INSUFFICIENT
        return json.dumps(
            {
                "answer": answer,
                "citations": [{"source": s, "quote": q} for s, q in citations],
                "confidence": confidence,
                "insufficient_context": insufficient,
            }
        )

    def _usage(self, messages: list[Message], text: str, start: float) -> LLMResponse:
        prompt_tokens = sum(self._tok.count(m["content"]) for m in messages)
        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=self._tok.count(text),
            latency_ms=round((time.perf_counter() - start) * 1000, 3),
            usage_estimated=True,
        )

    async def complete(self, messages: list[Message], *, json_mode: bool = False) -> LLMResponse:
        start = time.perf_counter()
        self._last = self._usage(messages, self._render(messages, json_mode), start)
        return self._last

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        start = time.perf_counter()
        text = self._render(messages, json_mode=False)
        for piece in re.findall(r"\S+\s*", text):
            yield piece
        self._last = self._usage(messages, text, start)


class OpenAIChat:  # pragma: no cover - network provider, only in live tests
    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self._last: LLMResponse | None = None
        self._client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )

    @property
    def last_usage(self) -> LLMResponse | None:
        return self._last

    async def complete(self, messages: list[Message], *, json_mode: bool = False) -> LLMResponse:
        start = time.perf_counter()
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        resp = await self._client.post("/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        self._last = LLMResponse(
            text=data["choices"][0]["message"]["content"] or "",
            model=data.get("model", self.model),
            prompt_tokens=data["usage"]["prompt_tokens"],
            completion_tokens=data["usage"]["completion_tokens"],
            latency_ms=round((time.perf_counter() - start) * 1000, 3),
        )
        return self._last

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        start = time.perf_counter()
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        parts: list[str] = []
        usage: dict[str, int] = {}
        async with self._client.stream("POST", "/chat/completions", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {}).get("content")
                    if delta:
                        parts.append(delta)
                        yield delta
        self._last = LLMResponse(
            text="".join(parts),
            model=self.model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=round((time.perf_counter() - start) * 1000, 3),
        )


class GeminiChat:  # pragma: no cover - network provider, only in live tests
    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self._last: LLMResponse | None = None
        self._client = httpx.AsyncClient(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            headers={"x-goog-api-key": api_key},
            timeout=60,
        )

    @property
    def last_usage(self) -> LLMResponse | None:
        return self._last

    def _body(self, messages: list[Message], json_mode: bool) -> dict[str, Any]:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages
            if m["role"] != "system"
        ]
        config: dict[str, Any] = {"temperature": 0}
        if json_mode:
            config["responseMimeType"] = "application/json"
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": config,
        }

    def _response(self, data: dict[str, Any], text: str, start: float) -> LLMResponse:
        usage = data.get("usageMetadata", {})
        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=int(usage.get("promptTokenCount", 0)),
            completion_tokens=int(usage.get("candidatesTokenCount", 0)),
            latency_ms=round((time.perf_counter() - start) * 1000, 3),
        )

    async def complete(self, messages: list[Message], *, json_mode: bool = False) -> LLMResponse:
        start = time.perf_counter()
        resp = await self._client.post(
            f"/models/{self.model}:generateContent", json=self._body(messages, json_mode)
        )
        resp.raise_for_status()
        data = resp.json()
        parts = data["candidates"][0]["content"].get("parts", [])
        self._last = self._response(data, "".join(p.get("text", "") for p in parts), start)
        return self._last

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        start = time.perf_counter()
        chunks: list[str] = []
        last: dict[str, Any] = {}
        async with self._client.stream(
            "POST",
            f"/models/{self.model}:streamGenerateContent?alt=sse",
            json=self._body(messages, json_mode=False),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                last = json.loads(line[6:])
                for cand in last.get("candidates", []):
                    for part in cand.get("content", {}).get("parts", []):
                        if text := part.get("text"):
                            chunks.append(text)
                            yield text
        self._last = self._response(last, "".join(chunks), start)
