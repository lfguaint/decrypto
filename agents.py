from __future__ import annotations
import json
import os
import re
import time
from datetime import datetime

import openai
from openai import OpenAI

from config import (AgentConfig, Backend, BACKEND_SETTINGS, REQUEST_TIMEOUT_S, TRANSIENT_RETRIES,
                    OLLAMA_MODELS, OPENROUTER_MODELS, DGX_MODELS, openrouter_routing)
from models import CallRecord, KeywordPair, RoundRecord, count_multiword
import prompts


def _make_client(backend: str) -> OpenAI:
    settings = BACKEND_SETTINGS[backend]

    if backend == Backend.OLLAMA:
        return OpenAI(
            base_url=settings["base_url"],
            api_key=settings["api_key"],
            max_retries=0,          # retries are handled in _call
            timeout=REQUEST_TIMEOUT_S,
        )

    if backend == Backend.CLUSTER:
        api_key = os.environ.get("DGX_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DGX_API_KEY environment variable is not set.\n"
                "Export it in your shell or add it to .env (never commit the key)."
            )
        return OpenAI(
            base_url=settings["base_url"],
            api_key=api_key,
            max_retries=0,          # retries are handled in _call
            timeout=REQUEST_TIMEOUT_S,
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
            max_retries=0,          # retries are handled in _call
            timeout=REQUEST_TIMEOUT_S,
        )

    raise ValueError(f"Unknown backend: {backend!r}.")


def _resolve_model(model: str, backend: str) -> str:
    if backend == Backend.OLLAMA:
        return OLLAMA_MODELS.get(model, model)
    if backend == Backend.OPENROUTER:
        return OPENROUTER_MODELS.get(model, model)
    if backend == Backend.CLUSTER:
        return DGX_MODELS.get(model, model)
    return model



def _extra(obj) -> dict:
    """Provider-specific fields the OpenAI SDK keeps outside its typed schema."""
    return (getattr(obj, "model_extra", None) or {}) if obj is not None else {}


def _create_with_retries(client: OpenAI, kwargs: dict):
    """Retry transient failures with exponential backoff; a timeout is raised at once."""
    for attempt in range(TRANSIENT_RETRIES + 1):
        started_at = datetime.now().isoformat(timespec="milliseconds")
        t0 = time.time()
        try:
            return client.chat.completions.create(**kwargs), time.time() - t0, started_at
        except openai.APITimeoutError:
            raise  # longer than REQUEST_TIMEOUT_S: do not pay for it again
        except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError):
            if attempt == TRANSIENT_RETRIES:
                raise
            time.sleep(min(60, 2 ** attempt))


def _call(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float | None,
    max_tokens: int | None = None,
    backend: str | None = None,
) -> tuple[str, CallRecord]:
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
    if backend == Backend.OPENROUTER:
        kwargs["extra_body"] = {
            "usage": {"include": True},  # ask OpenRouter to report cost
            "provider": openrouter_routing(model),
        }
    response, elapsed, started_at = _create_with_retries(client, kwargs)
    message = response.choices[0].message
    content = (message.content or "").strip()
    usage = response.usage
    details = getattr(usage, "completion_tokens_details", None)
    call = CallRecord(
        model_requested=model,
        model_served=getattr(response, "model", None),
        provider=_extra(response).get("provider"),
        response_id=getattr(response, "id", None),
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        reasoning_tokens=getattr(details, "reasoning_tokens", None),
        cost=_extra(usage).get("cost"),
        started_at=started_at,
        elapsed_seconds=elapsed,
        content=content,
        reasoning=_extra(message).get("reasoning"),
    )
    return content, call


def _parse_json(text: str) -> dict:
    # Strip reasoning traces from thinking models:
    #   - full <think>...</think> blocks (e.g. QwQ)
    #   - bare ...text...</think> prefix (e.g. Nemotron — no opening tag)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"^.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()
    # Strip markdown code fences and stray backticks
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip().rstrip("`").strip()
    # Extract first {...} block if the model added surrounding prose
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)


class InvalidResponse(ValueError):
    """A response the game cannot use; `kind` is 'format_error' or 'rule_violation'."""

    def __init__(self, kind: str, detail: str):
        super().__init__(detail)
        self.kind = kind


def _parse_guess(text: str, code_length: int, num_keywords: int) -> tuple:
    try:
        g = _parse_json(text)["guess"]
        result = tuple(int(x) for x in g)
    except Exception as e:
        raise InvalidResponse("format_error", f"unparseable guess: {type(e).__name__}: {e}")
    if (len(result) != code_length
            or not all(1 <= d <= num_keywords for d in result)
            or len(set(result)) != code_length):
        raise InvalidResponse("format_error", f"invalid guess digits: {list(result)}")
    return result


def _parse_clues(text: str, code_length: int) -> tuple:
    try:
        data = _parse_json(text)
        clues = tuple(str(data[f"clue{i+1}"]).strip() for i in range(code_length))
    except Exception as e:
        raise InvalidResponse("format_error", f"unparseable clues: {type(e).__name__}: {e}")
    if not all(clues):
        raise InvalidResponse("format_error", f"empty clue: {list(clues)}")
    return clues


def _clue_rule_violations(clues: tuple, keywords: list[KeywordPair],
                          history: list[RoundRecord]) -> list[str]:
    """The clue rules of the prompt that need no interpretation. Literal reading of
    the manual: a keyword (or its plain plural) is forbidden, a word containing one
    ('iceberg' for 'ice') is allowed. Meaning-vs-spelling/sound/position is not checked."""
    out = []
    used = {c.lower() for r in history for c in r.clues}
    for i, clue in enumerate(clues, start=1):
        c = clue.lower()
        if count_multiword([clue]):
            out.append(f"clue{i} {clue!r}: more than one word")
        if any(ch.isdigit() for ch in c):
            out.append(f"clue{i} {clue!r}: contains a digit")
        for kp in keywords:
            k = kp.word.lower()
            if c in (k, k + "s", k + "es"):
                out.append(f"clue{i} {clue!r}: is the keyword {kp.word!r}")
        if c in used:
            out.append(f"clue{i} {clue!r}: already used in an earlier round")
        if c in {x.lower() for x in clues[:i - 1]}:
            out.append(f"clue{i} {clue!r}: repeated within the round")
    return out


def _validated(call: CallRecord, parse) -> tuple:
    """Run `parse`; on an unusable response, mark the call and return None."""
    try:
        return parse(), call
    except InvalidResponse as e:
        call.status, call.error = e.kind, str(e)
        return None, call


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
        self._backend = backend
        self._model = _resolve_model(cfg.encryptor_model, backend)
        self._keywords = keywords
        self._code_length = code_length
        self._system = prompts.encryptor_system(keywords, code_length)

    def give_clues(self, code: tuple, history: list[RoundRecord]) -> tuple:
        """Returns (clues or None if unusable, call)."""
        user = prompts.encryptor_user(code, history)
        raw, call = _call(self._client, self._model, self._system, user,
                          self._cfg.encryptor_temperature, self._cfg.max_tokens, self._backend)

        def parse():
            clues = _parse_clues(raw, self._code_length)
            violations = _clue_rule_violations(clues, self._keywords, history)
            if violations:
                raise InvalidResponse("rule_violation", "; ".join(violations))
            return clues
        return _validated(call, parse)


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
        self._backend = backend
        self._model = _resolve_model(cfg.interceptor_model, backend)
        self._num_keywords = num_keywords
        self._code_length = code_length
        self._system = prompts.interceptor_system(num_keywords, code_length)

    def guess(self, clues: tuple, history: list[RoundRecord]) -> tuple:
        """Returns (guess or None if unusable, call)."""
        user = prompts.interceptor_user(clues, history, self._num_keywords)
        raw, call = _call(self._client, self._model, self._system, user,
                          self._cfg.interceptor_temperature, self._cfg.max_tokens, self._backend)
        return _validated(call, lambda: _parse_guess(raw, self._code_length, self._num_keywords))


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
        self._backend = backend
        self._model = _resolve_model(cfg.decoder_model, backend)
        self._num_keywords = len(keywords)
        self._code_length = code_length
        self._system = prompts.decoder_system(keywords, code_length)

    def guess(self, clues: tuple, history: list[RoundRecord]) -> tuple:
        """Returns (guess or None if unusable, call)."""
        user = prompts.decoder_user(clues, history, self._num_keywords)
        raw, call = _call(self._client, self._model, self._system, user,
                          self._cfg.decoder_temperature, self._cfg.max_tokens, self._backend)
        return _validated(call, lambda: _parse_guess(raw, self._code_length, self._num_keywords))
