"""Клиент OpenAI-совместимого Chat Completions API (OpenRouter и аналоги).

Модуль ничего не знает о Telegram и БД: получает готовый список сообщений,
возвращает текст либо исключение из иерархии LLMError.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Protocol

import aiohttp
from aiohttp_socks import ProxyConnectionError, ProxyConnector, ProxyError, ProxyTimeoutError

from app.config import Settings

logger = logging.getLogger("app.llm")

# Временные ошибки, после которых допустим один повтор.
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 1.0


class LLMError(Exception):
    """Базовая ошибка обращения к модели. Текст не содержит секретов и диалога."""


class LLMTimeoutError(LLMError):
    pass


class LLMUnavailableError(LLMError):
    """Сеть, 5xx, лимит частоты, ошибка авторизации или отказ сервиса."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class LLMNotConfiguredError(LLMError):
    """Ключ API не задан: запрос к модели не отправляется."""


class LLMBadResponseError(LLMError):
    """Ответ пришёл, но в нём нет ожидаемого текста."""


@dataclass(frozen=True)
class ChatMessage:
    role: str  # system | user | assistant
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class LLMResult:
    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class LLMClient(Protocol):
    model: str

    async def complete(self, messages: list[ChatMessage], *, temperature: float) -> LLMResult: ...


def parse_completion(data: object) -> LLMResult:
    """Проверяет форму ответа Chat Completions и извлекает текст."""
    if not isinstance(data, dict):
        raise LLMBadResponseError("ответ не является JSON-объектом")
    if data.get("error"):
        # OpenRouter может вернуть ошибку провайдера с HTTP 200.
        raise LLMUnavailableError("сервис вернул ошибку в теле ответа")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LLMBadResponseError("в ответе нет choices")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise LLMBadResponseError("в ответе нет текста")
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}

    def tokens(key: str) -> int | None:
        value = usage.get(key)
        return value if isinstance(value, int) else None

    finish = choices[0].get("finish_reason")
    return LLMResult(
        text=content.strip(),
        finish_reason=finish if isinstance(finish, str) else None,
        prompt_tokens=tokens("prompt_tokens"),
        completion_tokens=tokens("completion_tokens"),
        total_tokens=tokens("total_tokens"),
    )


class OpenAICompatibleClient:
    """POST {base_url}/chat/completions с Bearer-ключом и конечным таймаутом."""

    def __init__(self, settings: Settings):
        self.model = settings.openrouter_model
        self._url = settings.openrouter_base_url + "/chat/completions"
        self._api_key = settings.openrouter_api_key
        self._proxy = settings.llm_proxy_url
        self._timeout = aiohttp.ClientTimeout(total=settings.llm_timeout_seconds)
        self._max_tokens = settings.llm_max_tokens
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = ProxyConnector.from_url(self._proxy) if self._proxy else None
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=self._timeout,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()

    async def complete(self, messages: list[ChatMessage], *, temperature: float) -> LLMResult:
        payload = {
            "model": self.model,
            "messages": [message.as_dict() for message in messages],
            "temperature": temperature,
            "max_tokens": self._max_tokens,
        }
        if not self._api_key:
            logger.warning("LLM: OPENROUTER_API_KEY не задан, запрос не отправлен.")
            raise LLMNotConfiguredError("ключ API не задан")
        request_id = uuid.uuid4().hex[:8]
        for attempt in range(1, MAX_ATTEMPTS + 1):
            started = time.monotonic()
            logger.info(
                "LLM %s: запрос, модель=%s, сообщений=%d, temperature=%s, попытка=%d",
                request_id,
                self.model,
                len(messages),
                temperature,
                attempt,
            )
            try:
                result = await self._send(payload)
            except LLMError as exc:
                elapsed = time.monotonic() - started
                retry = (
                    attempt < MAX_ATTEMPTS
                    and isinstance(exc, LLMUnavailableError)
                    and (exc.status is None or exc.status in RETRYABLE_STATUSES)
                )
                logger.warning(
                    "LLM %s: ошибка %s (%s) за %.2f с%s",
                    request_id,
                    type(exc).__name__,
                    exc,
                    elapsed,
                    ", повтор" if retry else "",
                )
                if not retry:
                    raise
                await asyncio.sleep(RETRY_DELAY_SECONDS)
                continue
            logger.info(
                "LLM %s: ответ за %.2f с, finish=%s, токены prompt/completion/total=%s/%s/%s",
                request_id,
                time.monotonic() - started,
                result.finish_reason,
                result.prompt_tokens,
                result.completion_tokens,
                result.total_tokens,
            )
            return result
        raise AssertionError("unreachable")  # pragma: no cover

    async def _send(self, payload: dict) -> LLMResult:
        try:
            async with self._get_session().post(self._url, json=payload) as response:
                if response.status >= 400:
                    # Тело ошибки не логируем: провайдер может повторить в нём запрос.
                    raise LLMUnavailableError(f"HTTP {response.status}", status=response.status)
                try:
                    data = await response.json(content_type=None)
                except ValueError:
                    raise LLMBadResponseError("ответ не является JSON") from None
        except (TimeoutError, ProxyTimeoutError):
            raise LLMTimeoutError("истёк таймаут ответа модели") from None
        except (aiohttp.ClientError, ProxyError, ProxyConnectionError, OSError) as exc:
            raise LLMUnavailableError(f"сетевая ошибка {type(exc).__name__}") from None
        return parse_completion(data)
