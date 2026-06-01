from __future__ import annotations
import json
import os
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
        )

    raise ValueError(f"Unknown backend: {backend!r}. Use 'ollama' or 'openrouter'.")


def _resolve_model(model: str, backend: str) -> str:
    if backend == Backend.OLLAMA:
        return OLLAMA_MODELS.get(model, model)
    if backend == Backend.OPENROUTER:
        return OPENROUTER_MODELS.get(model, model)
    return model


def _call(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()


def _parse_json(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    return json.loads(cleaned)


def _parse_guess(text: str, code_length: int) -> tuple:
    data = _parse_json(text)
    g = data["guess"]
    return tuple(int(x) for x in g[:code_length])


def _parse_clues(text: str, code_length: int) -> tuple:
    data = _parse_json(text)
    return tuple(str(data[f"clue{i+1}"]) for i in range(code_length))


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
        self._system = prompts.encryptor_system(keywords, code_length)

    def give_clues(self, code: tuple, history: list[RoundRecord]) -> tuple:
        user = prompts.encryptor_user(code, history)
        raw = _call(self._client, self._model, self._system, user,
                    self._cfg.max_tokens, self._cfg.temperature)
        return _parse_clues(raw, self._code_length)


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
                    self._cfg.max_tokens, self._cfg.temperature)
        return _parse_guess(raw, self._code_length)


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
                    self._cfg.max_tokens, self._cfg.temperature)
        return _parse_guess(raw, self._code_length)
