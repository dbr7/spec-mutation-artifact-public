"""OpenAI LLM wrapper with disk cache and retry."""

import hashlib
import json
import logging

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from specmut.config import Config

logger = logging.getLogger(__name__)


class EmptyLLMResponseError(RuntimeError):
    """Raised when an API call succeeds but returns no visible content."""


def _is_retryable_llm_error(e: Exception) -> bool:
    return (
        isinstance(
            e,
            (
                RateLimitError,
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                EmptyLLMResponseError,
            ),
        )
        or (isinstance(e, APIStatusError) and 500 <= e.status_code < 600)
    )


def _strip_openai_prefix(model: str) -> str:
    return model[len("openai/"):] if model.startswith("openai/") else model


class LLMClient:
    def __init__(self, config: Config):
        self.config = config
        self.model = config.model_generator
        self.cache_dir = config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client: OpenAI | None = None

    def _client_for_openai(self) -> OpenAI:
        if self._client is not None:
            return self._client

        api_key = self.config.api_keys.get("openai")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found in the environment or api_key.txt."
            )

        kwargs = {"api_key": api_key, "max_retries": 0}
        if self.config.openai_base_url:
            kwargs["base_url"] = self.config.openai_base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def call(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 4096,
        cache_salt: str = "",
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        model = model or self.model
        if reasoning_effort is None:
            reasoning_effort = self._default_reasoning_effort(model)

        cache_key = self._cache_key(
            model, messages, cache_salt,
            reasoning_effort=reasoning_effort,
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        client = self._client_for_openai()
        bare_model = _strip_openai_prefix(model)
        attempts = max(1, self.config.llm_max_attempts)
        retryer = Retrying(
            retry=retry_if_exception(_is_retryable_llm_error),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            stop=stop_after_attempt(attempts),
            reraise=True,
        )
        for attempt in retryer:
            with attempt:
                response = self._api_call(
                    client, messages, bare_model, max_tokens,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                )
                self._cache_set(
                    cache_key, model, messages, response,
                    reasoning_effort=reasoning_effort,
                )
                return response
        raise RuntimeError("LLM retry loop exited without a response")

    def _default_reasoning_effort(self, model: str) -> str | None:
        if model == self.config.model_generator:
            return self.config.generator_reasoning_effort
        if model == self.config.model_tooling:
            return self.config.tooling_reasoning_effort
        return None

    def _api_call(
        self,
        client: OpenAI,
        messages: list[dict],
        model: str,
        max_tokens: int,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        if model.startswith(("o1", "o3", "o4", "gpt-5")):
            kwargs = {"max_completion_tokens": max(max_tokens * 8, 32768)}
        else:
            kwargs = {"max_tokens": max_tokens}
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        timeout = self.config.llm_timeout_seconds
        if timeout is None:
            timeout = 300.0 if model.startswith(("o1", "o3", "o4", "gpt-5")) else 120.0

        if temperature is not None and not model.startswith(
            ("o1", "o3", "o4", "gpt-5")
        ):
            kwargs["temperature"] = temperature

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=timeout,
            **kwargs,
        )
        content = response.choices[0].message.content
        if not content:
            raise EmptyLLMResponseError(
                f"{model} returned an empty chat completion"
            )
        return content

    def _cache_key(
        self,
        model: str,
        messages: list[dict],
        salt: str = "",
        reasoning_effort: str | None = None,
    ) -> str:
        payload = {"model": model, "messages": messages}
        if salt:
            payload["salt"] = salt
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        content = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _cache_get(self, key: str) -> str | None:
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text())
            return data["response"]
        return None

    def _cache_set(
        self,
        key: str,
        model: str,
        messages: list[dict],
        response: str,
        reasoning_effort: str | None = None,
    ):
        path = self.cache_dir / key[:2] / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"model": model, "messages": messages, "response": response}
        if reasoning_effort:
            data["reasoning_effort"] = reasoning_effort
        path.write_text(json.dumps(data, indent=2))
