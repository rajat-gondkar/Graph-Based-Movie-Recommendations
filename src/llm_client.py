"""LLM chat wrapper for any OpenAI-compatible provider (NanoGPT by default, OpenRouter still works).

Retries with backoff, token and cost logging, and robust JSON parsing (spec 6.5).
"""
import csv
import json
import re
import threading
import time
from datetime import datetime, timezone

import openai
import requests
from openai import OpenAI

import config

RETRIES = 3                                # spec 6.5: 3 retries after the first try, backoff 2, 4, 8 s
FATAL_STATUS = {400, 401, 402, 403, 404}   # bad request, bad key, no credits, forbidden, unknown model
REFERER = "http://localhost:8501"          # optional OpenRouter attribution header (harmless)
USAGE_FIELDS = ["time_utc", "purpose", "model", "prompt_tokens", "completion_tokens", "reasoning_tokens",
                "cost_usd", "finish_reason", "ok", "provider", "cost_source"]

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_client: OpenAI | None = None
_variant = 0          # which "extra settings" variant works with this provider (see _extra_bodies)
_header_checked = False
_lock = threading.Lock()  # explanations call chat() from several threads; guards the client and the usage log
_price_cache: dict = {}


class LLMError(RuntimeError):
    """No usable reply after all retries (network trouble, rate limits, empty replies)."""


class LLMFatalError(RuntimeError):
    """A request that retrying can't fix (bad key, no credits, unknown model). Callers should stop."""


def model_name() -> str:
    """The model id from .env (LLM_MODEL, or the old OPENROUTER_MODEL)."""
    return config.get_llm_settings()["model"]


def provider_name() -> str:
    """'nanogpt', 'openrouter' or 'custom'."""
    return config.get_llm_settings()["provider"]


def _get_client() -> OpenAI:
    """One shared OpenAI-SDK client for the configured provider. The SDK's own retries are off; chat() retries."""
    with _lock:
        return _make_client()


def _make_client() -> OpenAI:
    global _client
    if _client is None:
        s = config.get_llm_settings()
        headers = {"HTTP-Referer": REFERER, "X-Title": config.APP_TITLE} if s["provider"] == "openrouter" else {}
        _client = OpenAI(base_url=s["base_url"], api_key=s["api_key"], max_retries=0, timeout=60,
                         default_headers=headers)
    return _client


def _extra_bodies(provider: str) -> list[dict | None]:
    """Extra request settings to try, most useful first. If the provider rejects one (HTTP 400), the next is used.

    - reasoning effort: keeps hidden "thinking" short on models that always reason;
    - include_usage (NanoGPT): asks for token counts in non-streaming replies.
    """
    effort = config.LLM_REASONING_EFFORT
    if provider == "openrouter":
        return [{"reasoning": {"effort": effort, "exclude": True}}, None] if effort else [None]
    if provider == "nanogpt":
        usage = {"include_usage": True}
        return [{**usage, "reasoning_effort": effort}, usage, None] if effort else [usage, None]
    return [{"reasoning_effort": effort}, None] if effort else [None]


def _create(client: OpenAI, settings: dict, messages: list[dict], max_tokens: int, temperature: float):
    """One API request. If the provider rejects the extra settings (HTTP 400), fall back to plainer ones."""
    global _variant
    variants = _extra_bodies(settings["provider"])
    while True:
        extra = variants[min(_variant, len(variants) - 1)]
        try:
            return client.chat.completions.create(
                model=settings["model"], messages=messages, max_tokens=max_tokens,
                temperature=temperature, extra_body=extra,
            )
        except openai.APIStatusError as exc:
            if exc.status_code == 400 and _variant < len(variants) - 1:
                print(f"  Note: {settings['provider']} rejected the extra settings {sorted(extra or {})}; "
                      f"continuing without them ({str(exc.message)[:120]})")
                _variant += 1
                continue
            raise


def model_prices() -> tuple[float, float] | None:
    """(USD per prompt token, USD per completion token) from the provider's public model list, or None."""
    s = config.get_llm_settings()
    key = (s["base_url"], s["model"])
    if key not in _price_cache:
        prices = None
        try:
            params = {"detailed": "true"} if s["provider"] == "nanogpt" else None
            models = requests.get(f"{s['base_url']}/models", params=params, timeout=20).json()["data"]
            pricing = next(m for m in models if m["id"] == s["model"])["pricing"]
            scale = 1e-6 if s["provider"] == "nanogpt" else 1.0  # NanoGPT lists USD per million tokens
            prices = (float(pricing["prompt"]) * scale, float(pricing["completion"]) * scale)
        except Exception:  # offline, renamed model, or a format change: costs just stay unknown
            pass
        _price_cache[key] = prices
    return _price_cache[key]


def _ensure_usage_header() -> None:
    """Upgrade an older llm_usage.csv in place: add the provider and cost_source columns (once per run)."""
    global _header_checked
    if _header_checked:
        return
    _header_checked = True
    path = config.LLM_USAGE_PATH
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
        existing = list(rows[0].keys()) if rows else []
    if existing and existing != USAGE_FIELDS:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=USAGE_FIELDS)
            writer.writeheader()
            for row in rows:  # rows written before this change all came from OpenRouter, which reports the cost
                row.setdefault("provider", "openrouter")
                row.setdefault("cost_source", "reported")
                writer.writerow(row)


def _log_usage(purpose: str, settings: dict, usage, finish_reason, ok: bool) -> None:
    """Thread-safe wrapper: one row per call, never two writers at once."""
    with _lock:
        _log_usage_unlocked(purpose, settings, usage, finish_reason, ok)


def _log_usage_unlocked(purpose: str, settings: dict, usage, finish_reason, ok: bool) -> None:
    """Append one row per API call to data/cache/llm_usage.csv (D16).

    The cost is what the provider reports (OpenRouter) or, if it reports none (NanoGPT),
    tokens x the provider's public list price.
    """
    _ensure_usage_header()
    details = getattr(usage, "completion_tokens_details", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    cost, source = getattr(usage, "cost", None), "reported"
    if cost is None:
        prices = model_prices()
        if prices and prompt_tokens is not None and completion_tokens is not None:
            cost, source = round(prompt_tokens * prices[0] + completion_tokens * prices[1], 8), "list_price"
        else:
            source = "unknown"
    row = {
        "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": purpose,
        "model": settings["model"],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": getattr(details, "reasoning_tokens", None),
        "cost_usd": cost,
        "finish_reason": finish_reason,
        "ok": ok,
        "provider": settings["provider"],
        "cost_source": source,
    }
    new_file = not config.LLM_USAGE_PATH.exists()
    with config.LLM_USAGE_PATH.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=USAGE_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def chat(messages: list[dict], max_tokens: int = 200, temperature: float = 0.2, purpose: str = "chat") -> str:
    """Send chat messages to the configured model and return the reply text.

    Retries network errors, rate limits, server errors, and empty replies 3 times (2, 4, 8 s apart).
    Raises LLMError if nothing usable comes back, LLMFatalError for errors retrying can't fix.
    """
    client, settings = _get_client(), config.get_llm_settings()
    last_error = "unknown error"
    for attempt in range(RETRIES + 1):
        try:
            response = _create(client, settings, messages, max_tokens, temperature)
        except openai.APIStatusError as exc:
            if exc.status_code in FATAL_STATUS:
                raise LLMFatalError(f"{settings['provider']} refused the request (HTTP {exc.status_code}): "
                                    f"{str(exc.message)[:200]}") from None
            last_error = f"HTTP {exc.status_code}"
        except openai.APIConnectionError as exc:  # includes timeouts
            last_error = type(exc).__name__
        else:
            choice = response.choices[0] if response.choices else None
            text = (choice.message.content or "").strip() if choice else ""
            finish = choice.finish_reason if choice else None
            _log_usage(purpose, settings, response.usage, finish, bool(text))
            if text:
                return text
            last_error = f"empty reply (finish_reason={finish})"
        if attempt < RETRIES:
            time.sleep(2 ** (attempt + 1))
    raise LLMError(f"no usable reply after {RETRIES + 1} attempts ({last_error})")


def parse_json_object(text: str) -> dict | None:
    """First JSON object in an LLM reply, ignoring code fences, <think> blocks, and extra words. None if absent."""
    if not text:
        return None
    cleaned = _FENCE_RE.sub("", _THINK_RE.sub("", text))
    match = _OBJECT_RE.search(cleaned)
    if not match:
        return None
    first_close = cleaned.find("}", match.start()) + 1
    for candidate in (match.group(0), cleaned[match.start():first_close]):  # greedy, then the shortest
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return obj if isinstance(obj, dict) else None
    return None
