"""Сценарии ассистента: режимы, история, настройки и обращение к модели.

Сервис не зависит от aiogram: получает chat_id и текст, возвращает текст для
пользователя. Telegram-обработчики только вызывают его методы.
"""

import asyncio
import logging
from collections import defaultdict

from app.context import MessageTooLongError, build_messages
from app.llm import (
    LLMClient,
    LLMError,
    LLMNotConfiguredError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.prompts import MODES, get_mode
from app.storage import Repository

logger = logging.getLogger("app.assistant")

ALLOWED_TEMPERATURES = (0.0, 0.3, 0.7, 1.0)

TEXT_TIMEOUT = "Модель слишком долго отвечает. Попробуйте повторить запрос чуть позже."
TEXT_RATE_LIMIT = "Слишком много запросов к модели. Подождите минуту и повторите."
TEXT_REJECTED = "Сервис языковой модели отклонил запрос. Сообщите администратору бота."
TEXT_UNAVAILABLE = "Сервис языковой модели сейчас недоступен. Попробуйте позже."
TEXT_NOT_CONFIGURED = (
    "Языковая модель не настроена: в .env не задан OPENROUTER_API_KEY. "
    "Добавьте ключ и перезапустите бота."
)
TEXT_EMPTY = "Модель не вернула текст ответа. Попробуйте переформулировать запрос."
TEXT_TOO_LONG = "Сообщение слишком длинное для обработки. Сократите его и отправьте снова."
TEXT_TRUNCATED = (
    "\n\n[Ответ обрезан: достигнут лимит длины. Напишите «продолжи», чтобы получить остаток.]"
)
TEXT_RESET = "История диалога очищена. Режим и temperature сохранены."
TEXT_BAD_TEMPERATURE = "Недопустимое значение temperature. Выберите 0.0, 0.3, 0.7 или 1.0."


def format_temperature(value: float) -> str:
    return f"{value:.1f}"


def parse_temperature(raw: str) -> float | None:
    """Принимает только одно из разрешённых значений; запятая допускается."""
    try:
        value = float(raw.strip().replace(",", "."))
    except ValueError:
        return None
    return value if value in ALLOWED_TEMPERATURES else None


def error_text(error: LLMError) -> str:
    if isinstance(error, LLMNotConfiguredError):
        return TEXT_NOT_CONFIGURED
    if isinstance(error, LLMTimeoutError):
        return TEXT_TIMEOUT
    if isinstance(error, LLMUnavailableError):
        if error.status == 429:
            return TEXT_RATE_LIMIT
        if error.status in {400, 401, 402, 403, 404}:
            return TEXT_REJECTED
        return TEXT_UNAVAILABLE
    return TEXT_EMPTY


def start_text() -> str:
    modes = "\n".join(f"/{mode.command} — {mode.description}" for mode in MODES.values())
    return (
        "Привет! Я AI-ассистент студента: отвечаю с помощью языковой модели и помню "
        "контекст диалога.\n\n"
        f"Режимы (по умолчанию — /study):\n{modes}\n\n"
        "Настройки:\n"
        "/settings — текущий режим, модель и temperature, выбор temperature\n"
        "/reset — очистить историю диалога\n\n"
        "Просто напишите сообщение — оно уйдёт в активный режим."
    )


class AssistantService:
    def __init__(
        self, repository: Repository, llm: LLMClient, *, max_messages: int, max_chars: int
    ):
        self.repository = repository
        self.llm = llm
        self.max_messages = max_messages
        self.max_chars = max_chars
        # Сообщения и сброс одного чата выполняются по очереди, чтобы ответ на старый
        # запрос не попал в историю после /reset или смены режима.
        self._locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def reply(self, chat_id: int, text: str) -> str:
        async with self._locks[chat_id]:
            settings = await self.repository.get_settings(chat_id)
            mode = get_mode(settings.mode)
            history = await self.repository.load_history(chat_id, self.max_messages)
            try:
                messages = build_messages(
                    mode,
                    history,
                    text,
                    max_messages=self.max_messages,
                    max_chars=self.max_chars,
                )
            except MessageTooLongError:
                return TEXT_TOO_LONG
            try:
                result = await self.llm.complete(messages, temperature=settings.temperature)
            except LLMError as exc:
                # Подробности уже записаны клиентом LLM; ассистентское сообщение не сохраняем.
                return error_text(exc)
            await self.repository.save_exchange(chat_id, text, result.text, keep=self.max_messages)
            logger.info(
                "Ответ подготовлен: режим=%s, история=%d сообщ., передано=%d сообщ.",
                mode.command,
                len(history),
                len(messages),
            )
            return result.text + (TEXT_TRUNCATED if result.truncated else "")

    async def switch_mode(self, chat_id: int, mode_name: str) -> str:
        mode = MODES[mode_name]
        async with self._locks[chat_id]:
            await self.repository.set_mode(chat_id, mode.command)
        return f"Режим «{mode.title}» включён, история диалога очищена.\n{mode.switch_hint}"

    async def reset(self, chat_id: int) -> str:
        async with self._locks[chat_id]:
            await self.repository.clear_history(chat_id)
        return TEXT_RESET

    async def settings_text(self, chat_id: int) -> str:
        settings = await self.repository.get_settings(chat_id)
        mode = get_mode(settings.mode)
        return (
            "Текущие настройки:\n"
            f"Режим: /{mode.command} — {mode.title}\n"
            f"Модель: {self.llm.model}\n"
            f"temperature: {format_temperature(settings.temperature)}\n\n"
            "Выберите новое значение temperature кнопкой ниже или командой, "
            "например /settings 0.7."
        )

    async def set_temperature(self, chat_id: int, raw: str) -> str:
        value = parse_temperature(raw)
        if value is None:
            return TEXT_BAD_TEMPERATURE
        await self.repository.set_temperature(chat_id, value)
        return (
            f"temperature = {format_temperature(value)} сохранена и будет применена "
            "со следующего запроса."
        )
