from __future__ import annotations
import json
import os
import re

from openai import OpenAI

from config import AgentConfig
from models import KeywordPair, RoundRecord
import prompts


def _make_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set.\n"
            "Get your key at https://openrouter.ai/keys"
        )
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/decrypto-simulation",
            "X-Title": "Decrypto LLM Simulation",
        },
    )


def _call(client: OpenAI, model: str, system: str, user: str, max_tokens: int, temperature: float) -> str:
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
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    return json.loads(cleaned)


def _parse_guess(text: str) -> tuple[int, int, int]:
    data = _parse_json(text)
    g = data["guess"]
    return (int(g[0]), int(g[1]), int(g[2]))


def _parse_clues(text: str) -> tuple[str, str, str]:
    data = _parse_json(text)
    return (str(data["clue1"]), str(data["clue2"]), str(data["clue3"]))


class Encryptor:
    def __init__(self, client: OpenAI, cfg: AgentConfig, keywords: list[KeywordPair]):
        self._client = client
        self._cfg = cfg
        self._system = prompts.encryptor_system(keywords)

    def give_clues(
        self, code: tuple[int, int, int], history: list[RoundRecord]
    ) -> tuple[str, str, str]:
        user = prompts.encryptor_user(code, history)
        raw = _call(
            self._client, self._cfg.encryptor_model, self._system, user,
            self._cfg.max_tokens, self._cfg.temperature,
        )
        return _parse_clues(raw)


class Interceptor:
    def __init__(self, client: OpenAI, cfg: AgentConfig):
        self._client = client
        self._cfg = cfg
        self._system = prompts.interceptor_system()

    def guess(
        self, clues: tuple[str, str, str], history: list[RoundRecord]
    ) -> tuple[int, int, int]:
        user = prompts.interceptor_user(clues, history)
        raw = _call(
            self._client, self._cfg.interceptor_model, self._system, user,
            self._cfg.max_tokens, self._cfg.temperature,
        )
        return _parse_guess(raw)


class Decoder:
    def __init__(self, client: OpenAI, cfg: AgentConfig, keywords: list[KeywordPair]):
        self._client = client
        self._cfg = cfg
        self._system = prompts.decoder_system(keywords)

    def guess(
        self, clues: tuple[str, str, str], history: list[RoundRecord]
    ) -> tuple[int, int, int]:
        user = prompts.decoder_user(clues, history)
        raw = _call(
            self._client, self._cfg.decoder_model, self._system, user,
            self._cfg.max_tokens, self._cfg.temperature,
        )
        return _parse_guess(raw)
