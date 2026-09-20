"""LLM and embedding provider abstraction.

* ``LLMProvider`` – ``complete`` (sync), ``acomplete`` (async), ``astream`` (async
  token stream), ``complete_json`` helper for schema-constrained outputs.
* ``EmbeddingProvider`` – ``embed`` (sync batch) and ``aembed`` (async batch).

Implementations: OpenAI, Anthropic, Fake. The Fake providers are deterministic and
key-free so the full pipeline runs in CI and in evaluators' environments without
credentials. Every call returns a ``Usage`` that the caller persists to
``ai_usage`` for cost tracking.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# pricing (USD per 1M tokens: input, output). Kept here so cost tracking works
# offline; override via LLM_PRICING_OVERRIDE if prices change.
# ---------------------------------------------------------------------------
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "claude-3-5-sonnet-latest": (3.00, 15.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "fake-llm": (0.0, 0.0),
    "fake-embedding": (0.0, 0.0),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = get_settings().llm_pricing_override.get(model) or PRICING.get(model)
    if prices is None:
        # unknown model: fall back to a conservative mid-tier price
        prices = (1.0, 4.0)
    return round((input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000, 8)


@dataclass
class Usage:
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cached: bool = False
    cost_override: float | None = None  # provider-reported cost (e.g. claude CLI) wins over the price table

    @property
    def cost_usd(self) -> float:
        if self.cost_override is not None:
            return round(self.cost_override, 8)
        return estimate_cost(self.model, self.input_tokens, self.output_tokens)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "cached": self.cached,
        }


@dataclass
class Completion:
    text: str
    usage: Usage


@dataclass
class Message:
    role: str  # system|user|assistant
    content: str


class LLMError(Exception):
    """Raised after retries are exhausted or for non-retryable errors."""


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(self, messages: list[Message], *, temperature: float = 0.2, max_tokens: int = 1024, json_mode: bool = False) -> Completion: ...
    async def acomplete(self, messages: list[Message], *, temperature: float = 0.2, max_tokens: int = 1024, json_mode: bool = False) -> Completion: ...
    def astream(self, messages: list[Message], *, temperature: float = 0.2, max_tokens: int = 1024) -> AsyncIterator[str | Usage]: ...


class EmbeddingProvider(Protocol):
    name: str
    model: str
    dimensions: int

    def embed(self, texts: list[str]) -> tuple[list[list[float]], Usage]: ...
    async def aembed(self, texts: list[str]) -> tuple[list[list[float]], Usage]: ...


def _is_transient(exc: BaseException) -> bool:
    name = type(exc).__name__
    return any(k in name for k in ("RateLimit", "Timeout", "APIConnection", "InternalServer", "Overloaded", "ServiceUnavailable"))


def _retrying():
    s = get_settings()
    return retry(
        retry=retry_if_exception(_is_transient),
        stop=stop_after_attempt(s.llm_max_retries),
        wait=wait_exponential_jitter(initial=1, max=20),
        reraise=True,
    )


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def parse_json_response(text: str) -> dict[str, Any]:
    """Tolerant JSON extraction: strips code fences and leading prose."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(text)
        if not m:
            raise LLMError(f"Model did not return JSON: {text[:200]!r}") from None
        return json.loads(m.group(0))


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
class OpenAILLM:
    name = "openai"

    def __init__(self, model: str, api_key: str | None, base_url: str | None, timeout: float):
        from openai import AsyncOpenAI, OpenAI

        self.model = model
        self._sync = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        self._async = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

    @staticmethod
    def _kwargs(messages, temperature, max_tokens, json_mode):
        kw: dict[str, Any] = {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        return kw

    def complete(self, messages, *, temperature=0.2, max_tokens=1024, json_mode=False) -> Completion:
        @_retrying()
        def _call():
            return self._sync.chat.completions.create(model=self.model, **self._kwargs(messages, temperature, max_tokens, json_mode))

        t0 = time.perf_counter()
        r = _call()
        u = r.usage
        return Completion(
            text=r.choices[0].message.content or "",
            usage=Usage(self.name, self.model, u.prompt_tokens if u else 0, u.completion_tokens if u else 0, int((time.perf_counter() - t0) * 1000)),
        )

    async def acomplete(self, messages, *, temperature=0.2, max_tokens=1024, json_mode=False) -> Completion:
        @_retrying()
        async def _call():
            return await self._async.chat.completions.create(model=self.model, **self._kwargs(messages, temperature, max_tokens, json_mode))

        t0 = time.perf_counter()
        r = await _call()
        u = r.usage
        return Completion(
            text=r.choices[0].message.content or "",
            usage=Usage(self.name, self.model, u.prompt_tokens if u else 0, u.completion_tokens if u else 0, int((time.perf_counter() - t0) * 1000)),
        )

    async def astream(self, messages, *, temperature=0.2, max_tokens=1024) -> AsyncIterator[str | Usage]:
        t0 = time.perf_counter()
        stream = await self._async.chat.completions.create(
            model=self.model, stream=True, stream_options={"include_usage": True}, **self._kwargs(messages, temperature, max_tokens, False)
        )
        usage = Usage(self.name, self.model)
        async for ev in stream:
            if ev.usage:
                usage.input_tokens, usage.output_tokens = ev.usage.prompt_tokens, ev.usage.completion_tokens
            if ev.choices and ev.choices[0].delta and ev.choices[0].delta.content:
                yield ev.choices[0].delta.content
        usage.latency_ms = int((time.perf_counter() - t0) * 1000)
        yield usage


class OpenAIEmbeddings:
    name = "openai"

    def __init__(self, model: str, dimensions: int, api_key: str | None, base_url: str | None, timeout: float):
        from openai import AsyncOpenAI, OpenAI

        self.model, self.dimensions = model, dimensions
        self._sync = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        self._async = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

    def embed(self, texts):
        @_retrying()
        def _call():
            return self._sync.embeddings.create(model=self.model, input=texts, dimensions=self.dimensions)

        t0 = time.perf_counter()
        r = _call()
        vecs = [d.embedding for d in sorted(r.data, key=lambda d: d.index)]
        return vecs, Usage(self.name, self.model, r.usage.prompt_tokens, 0, int((time.perf_counter() - t0) * 1000))

    async def aembed(self, texts):
        @_retrying()
        async def _call():
            return await self._async.embeddings.create(model=self.model, input=texts, dimensions=self.dimensions)

        t0 = time.perf_counter()
        r = await _call()
        vecs = [d.embedding for d in sorted(r.data, key=lambda d: d.index)]
        return vecs, Usage(self.name, self.model, r.usage.prompt_tokens, 0, int((time.perf_counter() - t0) * 1000))


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str, api_key: str | None, timeout: float):
        from anthropic import Anthropic, AsyncAnthropic

        self.model = model
        self._sync = Anthropic(api_key=api_key, timeout=timeout, max_retries=0)
        self._async = AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=0)

    @staticmethod
    def _split(messages: list[Message], json_mode: bool):
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        if json_mode:
            system += "\n\nRespond with a single JSON object and nothing else."
        rest = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        return system or None, rest

    def complete(self, messages, *, temperature=0.2, max_tokens=1024, json_mode=False) -> Completion:
        system, rest = self._split(messages, json_mode)

        @_retrying()
        def _call():
            return self._sync.messages.create(model=self.model, system=system or "", messages=rest, temperature=temperature, max_tokens=max_tokens)

        t0 = time.perf_counter()
        r = _call()
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return Completion(text, Usage(self.name, self.model, r.usage.input_tokens, r.usage.output_tokens, int((time.perf_counter() - t0) * 1000)))

    async def acomplete(self, messages, *, temperature=0.2, max_tokens=1024, json_mode=False) -> Completion:
        system, rest = self._split(messages, json_mode)

        @_retrying()
        async def _call():
            return await self._async.messages.create(model=self.model, system=system or "", messages=rest, temperature=temperature, max_tokens=max_tokens)

        t0 = time.perf_counter()
        r = await _call()
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return Completion(text, Usage(self.name, self.model, r.usage.input_tokens, r.usage.output_tokens, int((time.perf_counter() - t0) * 1000)))

    async def astream(self, messages, *, temperature=0.2, max_tokens=1024) -> AsyncIterator[str | Usage]:
        system, rest = self._split(messages, False)
        t0 = time.perf_counter()
        usage = Usage(self.name, self.model)
        async with self._async.messages.stream(model=self.model, system=system or "", messages=rest, temperature=temperature, max_tokens=max_tokens) as stream:
            async for text in stream.text_stream:
                yield text
            final = await stream.get_final_message()
            usage.input_tokens, usage.output_tokens = final.usage.input_tokens, final.usage.output_tokens
        usage.latency_ms = int((time.perf_counter() - t0) * 1000)
        yield usage


# ---------------------------------------------------------------------------
# Claude Code CLI (no API key: uses the user's Claude login via `claude -p`)
# ---------------------------------------------------------------------------
class ClaudeCLILLM:
    """Runs the local ``claude`` CLI in non-interactive print mode.

    Useful when a developer has a Claude subscription but no API key: the CLI is
    already authenticated. We pass our prompt as ``--system-prompt`` (replacing the
    Claude Code default, which would add ~30k cached tokens per call) and parse the
    JSON envelope, which includes token usage and the provider-reported cost.
    Latency is a few seconds per call, so this is a dev/demo path, not production.
    """

    name = "claude-cli"

    def __init__(self, model: str, binary: str, timeout: float):
        import shutil

        self.model = model
        self.binary = binary if binary and os.path.exists(binary) else (shutil.which(binary or "claude") or binary or "claude")
        self.timeout = max(timeout, 120.0)

    @staticmethod
    def _flatten(messages: list[Message], json_mode: bool) -> tuple[str, str]:
        system = "\n\n".join(m.content for m in messages if m.role == "system") or "You are a helpful assistant."
        # The CLI may inject the developer's personal style hooks/instructions; pin the register we need.
        system += (
            "\n\nStyle: write in complete, natural, grammatical sentences. Ignore any instruction elsewhere to use a terse, abbreviated or 'caveman' style."
        )
        if json_mode:
            system += "\n\nRespond with a single JSON object and nothing else — no prose, no code fences."
        convo = [m for m in messages if m.role != "system"]
        if len(convo) == 1:
            user = convo[0].content
        else:
            user = "\n\n".join(f"{m.role.upper()}: {m.content}" for m in convo) + "\n\nRespond to the last USER turn."
        return system, user

    def complete(self, messages, *, temperature=0.2, max_tokens=1024, json_mode=False) -> Completion:
        import subprocess

        system, user = self._flatten(messages, json_mode)
        env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDECODE") and k != "CLAUDE_CODE_CHILD_SESSION"}
        env["CAVEMAN_DEFAULT_MODE"] = "off"  # neutralise a popular style plugin so developer hooks don't colour app answers
        # --tools "" drops the Claude Code tool definitions (~30k tokens/call → ~1.5k); we only want text.
        cmd = [
            self.binary, "-p", user, "--output-format", "json", "--model", self.model, "--system-prompt", system,
            "--max-turns", "1", "--tools", "", "--disable-slash-commands", "--no-session-persistence",
        ]  # fmt: skip
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, env=env, cwd="/tmp")
        except subprocess.TimeoutExpired as e:
            raise LLMError(f"claude CLI timed out after {self.timeout}s") from e
        latency = int((time.perf_counter() - t0) * 1000)
        if proc.returncode != 0:
            raise LLMError(f"claude CLI failed ({proc.returncode}): {proc.stderr.strip()[:300]}")
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise LLMError(f"claude CLI returned non-JSON: {proc.stdout[:200]!r}") from e
        if data.get("is_error"):
            raise LLMError(f"claude CLI error: {str(data.get('result'))[:300]}")
        u = data.get("usage") or {}
        model_used = next(iter((data.get("modelUsage") or {}).keys()), self.model)
        usage = Usage(
            self.name,
            f"claude-cli:{model_used}",
            int(u.get("input_tokens", 0)) + int(u.get("cache_creation_input_tokens", 0)) + int(u.get("cache_read_input_tokens", 0)),
            int(u.get("output_tokens", 0)),
            latency,
            cost_override=float(data.get("total_cost_usd") or 0.0),
        )
        return Completion(str(data.get("result", "")), usage)

    async def acomplete(self, messages, *, temperature=0.2, max_tokens=1024, json_mode=False) -> Completion:
        import asyncio

        return await asyncio.to_thread(self.complete, messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)

    async def astream(self, messages, *, temperature=0.2, max_tokens=1024) -> AsyncIterator[str | Usage]:
        # the CLI's print mode is not token-streamed; emit the answer in word chunks so the UI still animates
        c = await self.acomplete(messages, temperature=temperature, max_tokens=max_tokens)
        words = c.text.split(" ")
        for i in range(0, len(words), 4):
            yield " ".join(words[i : i + 4]) + (" " if i + 4 < len(words) else "")
        yield c.usage


# ---------------------------------------------------------------------------
# Fake (offline, deterministic)
# ---------------------------------------------------------------------------
_WORD = re.compile(r"[A-Za-z0-9]+")


class FakeEmbeddings:
    """Hashed bag-of-words embedding. Similar texts share tokens → similar vectors,
    so retrieval is *meaningful* in tests, not random."""

    name = "fake"
    model = "fake-embedding"

    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def _one(self, text: str) -> list[float]:
        v = [0.0] * self.dimensions
        for w in _WORD.findall(text.lower()):
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            v[h % self.dimensions] += 1.0
            v[(h >> 16) % self.dimensions] += 0.5
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed(self, texts):
        t0 = time.perf_counter()
        vecs = [self._one(t) for t in texts]
        tokens = sum(len(t) // 4 for t in texts)
        return vecs, Usage(self.name, self.model, tokens, 0, int((time.perf_counter() - t0) * 1000))

    async def aembed(self, texts):
        return self.embed(texts)


class FakeLLM:
    """Produces plausible, grounded-looking output from the prompt itself:
    for JSON-mode prompts it fills the requested schema; for chat it echoes the
    most relevant context sentences with citations. Good enough to exercise every
    code path end-to-end."""

    name = "fake"
    model = "fake-llm"

    def _answer(self, messages: list[Message], json_mode: bool) -> str:
        prompt = "\n".join(m.content for m in messages)
        if json_mode or '"json"' in prompt.lower() or "json object" in prompt.lower():
            return self._json(prompt)
        # chat: pick the passage sentences that share the most words with the question
        system = messages[0].content if messages else ""
        question = messages[-1].content if messages else ""
        ctx = system.split("CONTEXT PASSAGES", 1)[-1]
        passages = re.findall(r"\[(\d+)\] \([^)]*\)\n(.*?)(?=\n\n\[\d+\] \(|\Z)", ctx, flags=re.S)
        if not passages:
            return "The provided documents do not contain enough information to answer that."
        qwords = {w for w in _WORD.findall(question.lower()) if len(w) > 3 or any(ch.isdigit() for ch in w)}
        scored: list[tuple[int, str, str]] = []
        for n, body in passages:
            for sent in re.split(r"(?<=[.!?])\s+|\n+", body):
                sent = sent.strip()
                if len(sent) < 25 or sent.isupper():
                    continue
                overlap = len(qwords & set(_WORD.findall(sent.lower())))
                scored.append((overlap, sent, n))
        scored.sort(key=lambda t: -t[0])
        best = [t for t in scored[:3] if t[0] > 0]
        if not best:
            return "The provided documents do not contain enough information to answer that. Try uploading a document that covers this topic."
        return " ".join(f"{sent} [{n}]" for _, sent, n in best)

    def _json(self, prompt: str) -> str:
        # detect which schema was requested by looking for known keys
        if '"summary"' in prompt and '"key_points"' in prompt:
            words = _WORD.findall(prompt.split("DOCUMENT", 1)[-1])[:60]
            return json.dumps(
                {
                    "title": " ".join(words[:6]).title() or "Untitled document",
                    "summary": "This document discusses " + " ".join(words[:40]) + ".",
                    "key_points": ["Key point one derived from the text.", "Key point two.", "Key point three."],
                    "category": "report",
                    "tags": ["document", "analysis", "fake-provider"],
                    "sentiment": {"label": "neutral", "score": 0.0, "rationale": "Informational tone."},
                    "entities": {"people": [], "organizations": [], "dates": [], "amounts": []},
                    "language": "en",
                    "suggested_questions": [
                        "What is the main purpose of this document?",
                        "What are the key figures mentioned?",
                        "Who is the intended audience?",
                    ],
                }
            )
        if '"standalone_question"' in prompt:
            q = prompt.rsplit("Follow-up question:", 1)[-1].strip().split("\n")[0]
            # crude coreference: a short follow-up inherits the topic of the previous user turn
            prev = re.findall(r"^USER: (.+)$", prompt, flags=re.M)
            if prev and re.match(r"^(and|what about|how about|it|that|they|those|also)\b", q, flags=re.I):
                q = f"{prev[-1].rstrip('?')} — specifically: {q}"
            return json.dumps({"standalone_question": q})
        if '"questions"' in prompt:
            return json.dumps({"questions": ["Can you elaborate on that?", "What else does the document say about this?", "Are there any related figures?"]})
        if '"scores"' in prompt and "Passages:" in prompt:
            q = re.search(r"Question: (.*)", prompt)
            qwords = {w.lower() for w in _WORD.findall(q.group(1) if q else "") if len(w) > 2}
            passages = re.findall(r"^\[\d+\] (.*?)(?=^\[\d+\] |\Z)", prompt.split("Passages:", 1)[-1], flags=re.S | re.M)
            scores = [min(10, 2 * len(qwords & {w.lower() for w in _WORD.findall(p)})) for p in passages]
            return json.dumps({"scores": scores})
        if "EXTRACTION SCHEMA" in prompt:
            return self._structured(prompt)
        if '"comparison"' in prompt:
            return json.dumps(
                {
                    "comparison": "Both documents cover related topics.",
                    "similarities": ["Shared subject matter."],
                    "differences": ["Different scope."],
                    "recommendation": "Read both.",
                }
            )
        return json.dumps({"result": "ok"})

    def _structured(self, prompt: str) -> str:
        """Fill the requested schema from regex-detectable facts in the document text."""
        schema_txt = prompt.split("EXTRACTION SCHEMA", 1)[-1].split("DOCUMENT", 1)[0]
        body = prompt.split("DOCUMENT", 1)[-1]
        amounts = re.findall(r"\$\s?[\d,]+(?:\.\d+)?(?:\s?(?:million|billion|k))?", body)
        dates = re.findall(
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}|\b(?:Q[1-4]\s+)?\d{4}\b", body
        )
        ids = re.findall(r"\b[A-Z]{2,5}-\d{3,}\b", body)
        names = re.findall(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", body)
        try:
            schema = json.loads(re.search(r"\{.*\}", schema_txt, re.S).group(0))
        except Exception:
            return json.dumps({"fields": {}, "confidence": 0.0})
        out: dict[str, Any] = {}
        for key, typ in schema.items():
            k = key.lower()
            if isinstance(typ, list):
                out[key] = (
                    [{"description": s.strip()[:80], "amount": a} for s, a in zip(re.split(r"(?<=[.!?])\s+", body)[:3], amounts[:3], strict=False)]
                    if "item" in k
                    else names[:3]
                )
            elif "amount" in k or "total" in k or "revenue" in k or "price" in k or "value" in k:
                out[key] = amounts[0] if amounts else ""
            elif "date" in k or "period" in k or "term" in k:
                out[key] = dates[0] if dates else ""
            elif "number" in k or "id" in k or "reference" in k:
                out[key] = ids[0] if ids else ""
            elif "currency" in k:
                out[key] = "USD" if amounts else ""
            elif "part" in k or "vendor" in k or "customer" in k or "compan" in k or "counterpart" in k:
                out[key] = names[0] if names else ""
            else:
                out[key] = ""
        return json.dumps({"fields": out, "confidence": 0.6 if amounts or dates else 0.3, "notes": "fake-provider extraction"})

    def complete(self, messages, *, temperature=0.2, max_tokens=1024, json_mode=False) -> Completion:
        t0 = time.perf_counter()
        text = self._answer(messages, json_mode)
        inp = sum(len(m.content) // 4 for m in messages)
        return Completion(text, Usage(self.name, self.model, inp, len(text) // 4, int((time.perf_counter() - t0) * 1000)))

    async def acomplete(self, messages, *, temperature=0.2, max_tokens=1024, json_mode=False) -> Completion:
        return self.complete(messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)

    async def astream(self, messages, *, temperature=0.2, max_tokens=1024) -> AsyncIterator[str | Usage]:
        c = self.complete(messages, temperature=temperature, max_tokens=max_tokens)
        for word in c.text.split(" "):
            yield word + " "
        yield c.usage


# ---------------------------------------------------------------------------
# factories
# ---------------------------------------------------------------------------
_DEFAULT_MODELS = {
    "openai": ("gpt-4o-mini", "gpt-4o-mini"),
    "anthropic": ("claude-sonnet-4-20250514", "claude-3-5-haiku-latest"),
    "claude-cli": ("sonnet", "haiku"),
    "fake": ("fake-llm", "fake-llm"),
}


@lru_cache
def get_llm(fast: bool = False) -> LLMProvider:
    s = get_settings()
    provider = s.resolved_llm_provider
    default_main, default_fast = _DEFAULT_MODELS[provider]
    model = (s.llm_fast_model or default_fast) if fast else (s.llm_model or default_main)
    if s.llm_provider == "auto":
        log.info("llm_provider_resolved", provider=provider, model=model)
    if provider == "openai":
        return OpenAILLM(model, s.openai_api_key, s.openai_base_url, s.llm_timeout_seconds)
    if provider == "anthropic":
        return AnthropicLLM(model, s.anthropic_api_key, s.llm_timeout_seconds)
    if provider == "claude-cli":
        return ClaudeCLILLM(model, s.find_claude_cli() or s.claude_cli_path, s.llm_timeout_seconds)
    return FakeLLM()


@lru_cache
def get_embeddings() -> EmbeddingProvider:
    s = get_settings()
    if s.resolved_embedding_provider == "openai":
        return OpenAIEmbeddings(s.embedding_model, s.embedding_dimensions, s.openai_api_key, s.openai_base_url, s.llm_timeout_seconds)
    return FakeEmbeddings(s.embedding_dimensions)


__all__ = [
    "Completion",
    "EmbeddingProvider",
    "LLMError",
    "LLMProvider",
    "Message",
    "Usage",
    "estimate_cost",
    "get_embeddings",
    "get_llm",
    "parse_json_response",
    "field",
]
