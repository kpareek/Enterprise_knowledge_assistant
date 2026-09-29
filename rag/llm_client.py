"""
Thin wrapper so the rest of the codebase never imports `openai` directly.
Switching PROVIDER in config.py from "openai" to "azure_openai" is the only
change needed to move the whole assistant onto Azure OpenAI.
"""
from __future__ import annotations

import os

import config

_local_embedder = None
def get_client():
    # max_retries=0: we run our own Retry-After-aware retry loop in
    # chat_complete, so the SDK's built-in retries would only double up the
    # backoff sleeps. A 30s per-request timeout keeps a single hung call bounded.
    if config.RAG_MODE == "free":
        from openai import OpenAI
        return OpenAI(api_key=config.OPENROUTER_API_KEY, base_url=config.OPENROUTER_BASE_URL,
                      max_retries=0, timeout=90.0)
    elif config.PROVIDER == "azure_openai":
        from openai import AzureOpenAI
        return AzureOpenAI(api_key=config.AZURE_OPENAI_API_KEY,
                            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                            api_version=config.AZURE_OPENAI_API_VERSION)
    else:
        from openai import OpenAI
        return OpenAI(api_key=config.OPENAI_API_KEY)



def embedding_model_name() -> str:
    return config.AZURE_EMBEDDING_DEPLOYMENT if config.PROVIDER == "azure_openai" else config.EMBEDDING_MODEL


def chat_model_name() -> str:
    if config.RAG_MODE == "free":
        return config.OPENROUTER_CHAT_MODEL
    elif config.PROVIDER == "azure_openai":
        return config.AZURE_CHAT_DEPLOYMENT
    else:
        return config.CHAT_MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    if config.RAG_MODE == "free":
        embedder = _get_local_embedder()
        return embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()
    client = get_client()
    model = embedding_model_name()
    out = []
    for i in range(0, len(texts), 100):
        resp = client.embeddings.create(model=model, input=texts[i:i+100])
        out.extend([d.embedding for d in resp.data])
    return out


# How long a single call will keep retrying through 429s before giving up.
# Overridable via env so an eval run can be extra-patient on a flaky free pool.
RETRY_DEADLINE_SEC = float(os.getenv("LLM_RETRY_DEADLINE_SEC", "300"))


def chat_complete(messages, temperature=0.2, max_tokens=800, stop=None) -> str:
    import time
    from openai import RateLimitError, APITimeoutError, APIConnectionError, InternalServerError

    client = get_client()
    # `stop` lets callers (e.g. the ReAct agent) halt generation at a marker
    # like "Observation:" so the model can't hallucinate its own tool output.
    kwargs = {"stop": stop} if stop else {}
    # Free-tier providers (e.g. OpenRouter's shared :free pool) fail transiently
    # under load: 429 rate limits, slow responses that time out, dropped
    # connections, and 5xx blips. Rather than a fixed retry count (one long
    # burst would kill a whole eval run), keep retrying every transient error
    # until a wall-clock deadline so a call rides out the overload window.
    transient = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
    deadline = time.monotonic() + RETRY_DEADLINE_SEC
    attempt = 0
    while True:
        try:
            resp = client.chat.completions.create(model=chat_model_name(), messages=messages,
                                                   temperature=temperature, max_tokens=max_tokens,
                                                   **kwargs)
            # OpenRouter's free pool sometimes returns a 200 whose body carries a
            # provider error instead of `choices` (choices=None), or a message
            # whose `content` is None when a reasoning model spends its whole
            # token budget on the hidden reasoning field. Both are transient --
            # treat them like a rate-limit blip and retry within the deadline
            # rather than crashing a whole eval run on one flaky call.
            content = resp.choices[0].message.content if resp.choices else None
            if content:
                return content
            if time.monotonic() >= deadline:
                return content or ""
            time.sleep(min(2 ** attempt, 12))
            attempt += 1
        except transient as e:
            # A daily-quota 429 ("free-models-per-day") is a hard wall, not a
            # transient blip -- retrying for the full deadline would just stall
            # for minutes before failing anyway. Surface it immediately.
            if _is_daily_quota_error(e):
                raise
            if time.monotonic() >= deadline:
                raise
            time.sleep(_retry_after(e, default=min(2 ** attempt, 12)))
            attempt += 1


def _is_daily_quota_error(err) -> bool:
    """True for hard-wall 429s that retrying can never clear: OpenRouter's
    account-wide daily free-request cap, and OpenAI's exhausted credit balance
    (both arrive as 429s but are billing/quota walls, not transient throttles)."""
    msg = str(getattr(err, "message", "") or err).lower()
    return any(marker in msg for marker in (
        "per-day", "free-models-per-day",
        "insufficient_quota", "credit_balance_exhausted", "no credits remaining",
    ))


def _retry_after(err, default: float) -> float:
    """Pull the provider's suggested Retry-After (seconds) from a 429, if any."""
    try:
        body = getattr(err, "body", None) or {}
        meta = (body.get("error", {}) or {}).get("metadata", {}) or {}
        secs = meta.get("retry_after_seconds")
        if secs:
            return float(secs) + 1.0
    except Exception:
        pass
    return default

def _get_local_embedder():
    global _local_embedder
    if _local_embedder is None:
        from sentence_transformers import SentenceTransformer
        _local_embedder = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
    return _local_embedder

