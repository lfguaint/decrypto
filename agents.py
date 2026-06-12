from __future__ import annotations
import json
import os
import random
import re

from openai import OpenAI

from config import AgentConfig, Backend, BACKEND_SETTINGS, OLLAMA_MODELS, OPENROUTER_MODELS
from models import KeywordPair, RoundRecord
import prompts


def _make_client(backend: str) -> OpenAI:
    settings = BACKEND_SETTINGS[backend]

    if backend == Backend.OLLAMA:
        return OpenAI(
            base_url=settings["base_url"],
            api_key=settings["api_key"],
            max_retries=5,
        )

    if backend == Backend.OPENROUTER:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY environment variable is not set.\n"
                "Get your key at https://openrouter.ai/keys"
            )
        return OpenAI(
            base_url=settings["base_url"],
            api_key=api_key,
            default_headers=settings["default_headers"],
            max_retries=5,   # exponential backoff on 429/5xx
        )

    raise ValueError(f"Unknown backend: {backend!r}. Use 'ollama' or 'openrouter'.")


def _resolve_model(model: str, backend: str) -> str:
    if backend == Backend.OLLAMA:
        return OLLAMA_MODELS.get(model, model)
    if backend == Backend.OPENROUTER:
        return OPENROUTER_MODELS.get(model, model)
    return model


MAX_RETRIES = 3


def _call(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float | None,
    max_tokens: int | None = None,
) -> str:
    kwargs: dict = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    response = client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


def _parse_json(text: str) -> dict:
    # Strip markdown code fences and stray backticks
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # Extract first {...} block if the model added surrounding prose
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)


def _random_guess(num_keywords: int, code_length: int) -> tuple:
    return tuple(random.sample(range(1, num_keywords + 1), code_length))


def _random_clues(code_length: int) -> tuple:
    fallbacks = ["thing", "object", "item", "concept", "word"]
    return tuple(fallbacks[i % len(fallbacks)] for i in range(code_length))


def _parse_guess(
    text: str, code_length: int, num_keywords: int, label: str = ""
) -> tuple:
    """Returns ((digits...), fallback_used)."""
    try:
        data = _parse_json(text)
        g = data["guess"]
        result = tuple(int(x) for x in g[:code_length])
        # Validate digits are in range and distinct
        if (len(result) == code_length
                and all(1 <= d <= num_keywords for d in result)
                and len(set(result)) == code_length):
            return result, False
        raise ValueError(f"Invalid guess digits: {result}")
    except Exception as e:
        print(f" [parse_guess{' '+label if label else ''} fallback: {e}]", end="")
        return _random_guess(num_keywords, code_length), True


def _parse_clues(text: str, code_length: int, label: str = "") -> tuple:
    """Returns ((clues...), fallback_used)."""
    try:
        data = _parse_json(text)
        return tuple(str(data[f"clue{i+1}"]) for i in range(code_length)), False
    except Exception as e:
        print(f" [parse_clues{' '+label if label else ''} fallback: {e}]", end="")
        return _random_clues(code_length), True


class Encryptor:
    def __init__(
        self,
        client: OpenAI,
        cfg: AgentConfig,
        backend: str,
        keywords: list[KeywordPair],
        code_length: int,
    ):
        self._client = client
        self._cfg = cfg
        self._model = _resolve_model(cfg.encryptor_model, backend)
        self._code_length = code_length
        if cfg.encryptor_system_template:
            self._system = prompts.render_encryptor_template(
                cfg.encryptor_system_template, keywords, code_length)
        else:
            self._system = prompts.encryptor_system(keywords, code_length, cfg.prompt_version)

    def give_clues(self, code: tuple, history: list[RoundRecord]) -> tuple:
        user = prompts.encryptor_user(code, history)
        raw = _call(self._client, self._model, self._system, user,
                    self._cfg.encryptor_temperature, self._cfg.max_tokens)
        return _parse_clues(raw, self._code_length, label="Enc")


class Interceptor:
    def __init__(
        self,
        client: OpenAI,
        cfg: AgentConfig,
        backend: str,
        num_keywords: int,
        code_length: int,
    ):
        self._client = client
        self._cfg = cfg
        self._model = _resolve_model(cfg.interceptor_model, backend)
        self._num_keywords = num_keywords
        self._code_length = code_length
        self._system = prompts.interceptor_system(num_keywords, code_length)

    def guess(self, clues: tuple, history: list[RoundRecord]) -> tuple:
        user = prompts.interceptor_user(clues, history, self._num_keywords)
        raw = _call(self._client, self._model, self._system, user,
                    self._cfg.interceptor_temperature, self._cfg.max_tokens)
        return _parse_guess(raw, self._code_length, self._num_keywords, label="Int")


class Decoder:
    def __init__(
        self,
        client: OpenAI,
        cfg: AgentConfig,
        backend: str,
        keywords: list[KeywordPair],
        code_length: int,
    ):
        self._client = client
        self._cfg = cfg
        self._model = _resolve_model(cfg.decoder_model, backend)
        self._num_keywords = len(keywords)
        self._code_length = code_length
        self._system = prompts.decoder_system(keywords, code_length)

    def guess(self, clues: tuple, history: list[RoundRecord]) -> tuple:
        user = prompts.decoder_user(clues, history, self._num_keywords)
        raw = _call(self._client, self._model, self._system, user,
                    self._cfg.decoder_temperature, self._cfg.max_tokens)
        return _parse_guess(raw, self._code_length, self._num_keywords, label="Dec")
