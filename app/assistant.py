"""Сценарии ассистента: режимы, история, настройки и обращение к модели.

Сервис не зависит от aiogram: получает chat_id и текст, возвращает текст для
пользователя. Telegram-обработчики только вызывают его методы.
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from app.agent import Agent, AgentReply
from app.confirmations import ReminderConfirmations
from app.context import MessageTooLongError, build_messages
from app.llm import LLMClient, LLMError, LLMTimeoutError, LLMUnavailableError
from app.mcp_gateway import McpGateway
from app.prompts import MODES, get_mode
from app.storage import AgentEvent, Repository
from app.timeutil import normalize_timezone

logger = logging.getLogger("app.assistant")

ALLOWED_TEMPERATURES = (0.0, 0.3, 0.7, 1.0)

TEXT_TIMEOUT = "Модель слишком долго отвечает. Попробуйте повторить запрос чуть позже."
TEXT_RATE_LIMIT = "Слишком много запросов к модели. Подождите минуту и повторите."
TEXT_REJECTED = "Сервис языковой модели отклонил запрос. Сообщите администратору бота."
TEXT_UNAVAILABLE = "Сервис языковой модели сейчас недоступен. Попробуйте позже."
TEXT_EMPTY = "Модель не вернула текст ответа. Попробуйте переформулировать запрос."
TEXT_TOO_LONG = "Сообщение слишком длинное для обработки. Сократите его и отправьте снова."
TEXT_TRUNCATED = (
    "\n\n[Ответ обрезан: достигнут лимит длины. Напишите «продолжи», чтобы получить остаток.]"
)
TEXT_RESET = "История диалога очищена. Режим и temperature сохранены."
TEXT_BAD_TEMPERATURE = "Недопустимое значение temperature. Выберите 0.0, 0.3, 0.7 или 1.0."
TEXT_BAD_TIMEZONE = (
    "Неизвестный часовой пояс. Укажите имя из базы IANA в формате Регион/Город, "
    "например: /timezone Europe/Moscow или /timezone Asia/Yekaterinburg"
)
TEXT_NO_TIMEZONE = "Часовой пояс не задан. Задайте его командой, например: /timezone Europe/Moscow"
TEXT_TOOLS_UNAVAILABLE = "Инструменты сейчас недоступны: MCP-сервер не подключён."
TEXT_NO_EVENTS = "Вы ещё не отправляли запросов агенту."
ACTION_TITLES = {
    "respond": "ответ без инструмента",
    "clarify": "уточнение",
    "limit": "остановка по лимиту вызовов",
    "error": "ошибка обработки",
}


@dataclass(frozen=True)
class Reply:
    text: str
    confirm_action_id: str | None = None


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
        "Агент (в режимах /study и /review):\n"
        "/tools — какие действия доступны через MCP-сервер\n"
        "/timezone <зона> — часовой пояс для напоминаний, например /timezone Europe/Moscow\n"
        "/why — почему агент выбрал действие в последнем запросе\n\n"
        "Просто напишите сообщение — оно уйдёт в активный режим. Можно спросить погоду, "
        "занятия на дату или попросить напомнить о чём-то."
    )


class AssistantService:
    def __init__(
        self,
        repository: Repository,
        llm: LLMClient,
        *,
        max_messages: int,
        max_chars: int,
        agent: Agent | None = None,
        confirmations: ReminderConfirmations | None = None,
        gateway: McpGateway | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.repository = repository
        self.llm = llm
        self.max_messages = max_messages
        self.max_chars = max_chars
        self.agent = agent
        self.confirmations = confirmations
        self.gateway = gateway
        self.clock = clock
        # Сообщения и сброс одного чата выполняются по очереди, чтобы ответ на старый
        # запрос не попал в историю после /reset или смены режима.
        self._locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def reply(self, chat_id: int, text: str, update_id: int | None = None) -> Reply:
        async with self._locks[chat_id]:
            started = time.monotonic()
            settings = await self.repository.get_settings(chat_id)
            mode = get_mode(settings.mode)
            history = await self.repository.load_history(chat_id, self.max_messages)
            try:
                if self.agent is not None and mode.tools:
                    outcome = await self.agent.run(
                        chat_id=chat_id,
                        update_id=update_id,
                        mode=mode,
                        history=history,
                        text=text,
                        temperature=settings.temperature,
                        timezone=settings.timezone,
                    )
                else:
                    outcome = await self._plain(mode, history, text, settings.temperature, started)
            except MessageTooLongError:
                return Reply(TEXT_TOO_LONG)
            except LLMError as exc:
                # Подробности уже записаны клиентом LLM; ответ ассистента не сохраняем.
                await self.repository.add_event(
                    chat_id, self._event("error", f"ошибка LLM: {type(exc).__name__}", started)
                )
                return Reply(error_text(exc))
            await self.repository.save_exchange(
                chat_id, text, outcome.history_text or outcome.text, keep=self.max_messages
            )
            await self.repository.add_event(chat_id, outcome.event)
            return Reply(outcome.text, outcome.confirm_action_id)

    @staticmethod
    def _event(action: str, execution: str, started: float, reason: str = "") -> AgentEvent:
        return AgentEvent(
            request_id=uuid.uuid4().hex[:8],
            action=action,
            tool_name=None,
            arguments_summary="—",
            validation="—",
            execution=execution,
            reason=reason or "инструмент не требовался",
            mcp_calls=0,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    async def _plain(self, mode, history, text, temperature, started) -> AgentReply:
        """Режим без инструментов (ЛР1): один запрос к модели."""
        messages = build_messages(
            mode, history, text, max_messages=self.max_messages, max_chars=self.max_chars
        )
        result = await self.llm.complete(messages, temperature=temperature)
        logger.info(
            "Ответ подготовлен: режим=%s, история=%d сообщ., передано=%d сообщ.",
            mode.command,
            len(history),
            len(messages),
        )
        reason = "режим без инструментов" if self.agent is not None else ""
        return AgentReply(
            text=result.text + (TEXT_TRUNCATED if result.truncated else ""),
            event=self._event("respond", "—", started, reason),
            history_text=result.text,
        )

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

    # ---------- ЛР2: часовой пояс, инструменты, аудит, подтверждения ----------

    async def set_timezone(self, chat_id: int, raw: str) -> str:
        zone = normalize_timezone(raw)
        if zone is None:
            return TEXT_BAD_TIMEZONE
        await self.repository.set_timezone(chat_id, zone)
        local = self.clock().astimezone(ZoneInfo(zone))
        return (
            f"Часовой пояс {zone} сохранён. Сейчас там {local:%d.%m.%Y %H:%M}. "
            "Время в напоминаниях будет пониматься в этом поясе."
        )

    async def timezone_text(self, chat_id: int) -> str:
        zone = (await self.repository.get_settings(chat_id)).timezone
        if zone is None:
            return TEXT_NO_TIMEZONE
        local = self.clock().astimezone(ZoneInfo(zone))
        return f"Ваш часовой пояс: {zone}. Сейчас там {local:%d.%m.%Y %H:%M}."

    def tools_text(self) -> str:
        tools = self.gateway.tools if self.gateway is not None else []
        if not tools:
            return TEXT_TOOLS_UNAVAILABLE
        lines = ["Доступные инструменты (получены от MCP-сервера):"]
        for tool in tools:
            kind = "чтение" if tool.read_only else "изменение, нужно подтверждение"
            lines.append(f"• {tool.name} — {tool.title} ({kind})")
        return "\n".join(lines)

    async def why_text(self, chat_id: int) -> str:
        event = await self.repository.last_event(chat_id)
        if event is None:
            return TEXT_NO_EVENTS
        fallback = self.agent.schedule_timezone if self.agent is not None else "UTC"
        zone = (await self.repository.get_settings(chat_id)).timezone or fallback
        moment = (event.created_at or self.clock()).astimezone(ZoneInfo(zone))
        action = (
            f"вызов инструмента {event.tool_name}"
            if event.action == "tool" and event.tool_name
            else ACTION_TITLES.get(event.action, event.action)
        )
        return (
            f"Последний запрос: {moment:%d.%m.%Y %H:%M} ({zone})\n"
            f"Действие: {action}\n"
            f"Аргументы: {event.arguments_summary}\n"
            f"Проверка: {event.validation}\n"
            f"Выполнение: {event.execution}\n"
            f"Вызовов MCP: {event.mcp_calls}\n"
            f"Причина: {event.reason}"
        )

    async def confirm(self, chat_id: int, action_id: str) -> str:
        if self.confirmations is None:
            return TEXT_TOOLS_UNAVAILABLE
        async with self._locks[chat_id]:
            return await self.confirmations.confirm(chat_id, action_id)

    async def cancel(self, chat_id: int, action_id: str) -> str:
        if self.confirmations is None:
            return TEXT_TOOLS_UNAVAILABLE
        return await self.confirmations.cancel(chat_id, action_id)
