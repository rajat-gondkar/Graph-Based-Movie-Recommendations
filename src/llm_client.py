"""OpenRouter chat wrapper (spec 6.5): retries with backoff, token and cost logging, robust JSON parsing."""
import csv
import json
import re
import time
from datetime import datetime, timezone

import openai
from openai import OpenAI

import config

RETRIES = 3                                # spec 6.5: 3 retries after the first try, backoff 2, 4, 8 s
FATAL_STATUS = {400, 401, 402, 403, 404}   # bad request, bad key, no credits, forbidden, unknown model
REFERER = "http://localhost:8501"          # optional OpenRouter attribution header (harmless)
USAGE_FIELDS = ["time_utc", "purpose", "model", "prompt_tokens", "completion_tokens",
                "reasoning_tokens", "cost_usd", "finish_reason", "ok"]

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_client: OpenAI | None = None


class LLMError(RuntimeError):
    """No usable reply after all retries (network trouble, rate limits, empty replies)."""


class LLMFatalError(RuntimeError):
    """A request that retrying can't fix (bad key, no credits, unknown model). Callers should stop."""


def model_name() -> str:
    """The OpenRouter model id from .env (OPENROUTER_MODEL)."""
    return config.get_openrouter_settings()["model"]


def _get_client() -> OpenAI:
    """One shared OpenAI-SDK client pointed at OpenRouter. The SDK's own retries are off; chat() retries."""
    global _client
    if _client is None:
        settings = config.get_openrouter_settings()
        _client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=settings["api_key"],
            max_retries=0,
            timeout=60,
            default_headers={"HTTP-Referer": REFERER, "X-Title": config.APP_TITLE},
        )
    return _client


def _log_usage(purpose: str, model: str, usage, finish_reason, ok: bool) -> None:
    """Append one row per API call to data/cache/llm_usage.csv (D16). OpenRouter reports the cost in USD."""
    details = getattr(usage, "completion_tokens_details", None)
    row = {
        "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": purpose,
        "model": model,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None),
        "cost_usd": getattr(usage, "cost", None),
        "finish_reason": finish_reason,
        "ok": ok,
    }
    new_file = not config.LLM_USAGE_PATH.exists()
    with config.LLM_USAGE_PATH.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=USAGE_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def chat(messages: list[dict], max_tokens: int = 200, temperature: float = 0.2, purpose: str = "chat") -> str:
    """Send chat messages to the OpenRouter model and return the reply text.

    Retries network errors, rate limits, server errors, and empty replies 3 times (2, 4, 8 s apart).
    Raises LLMError if nothing usable comes back, LLMFatalError for errors retrying can't fix.
    """
    client, model = _get_client(), model_name()
    extra = None
    if config.LLM_REASONING_EFFORT:  # keep hidden "thinking" short on models that always reason
        extra = {"reasoning": {"effort": config.LLM_REASONING_EFFORT, "exclude": True}}
    last_error = "unknown error"
    for attempt in range(RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, extra_body=extra
            )
        except openai.APIStatusError as exc:
            if exc.status_code in FATAL_STATUS:
                raise LLMFatalError(f"OpenRouter refused the request (HTTP {exc.status_code}): {exc.message}") from None
            last_error = f"HTTP {exc.status_code}"
        except openai.APIConnectionError as exc:  # includes timeouts
            last_error = type(exc).__name__
        else:
            choice = response.choices[0] if response.choices else None
            text = (choice.message.content or "").strip() if choice else ""
            finish = choice.finish_reason if choice else None
            _log_usage(purpose, model, response.usage, finish, bool(text))
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
