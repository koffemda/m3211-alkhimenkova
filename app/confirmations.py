"""Подтверждение и отмена подготовленных напоминаний.

Состояния действия: pending → executing → done; pending → cancelled;
pending → expired (через 5 минут). Переход pending → executing выполняется
одним UPDATE, поэтому двойное нажатие «Подтвердить» исполняет действие один раз.
Сервер дополнительно защищён ключом идемпотентности = id действия.
"""

import logging
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.mcp_gateway import McpGateway, ToolCallError
from app.storage import Repository
from app.timeutil import format_local
from mcp_server import trust

logger = logging.getLogger("app.confirmations")

TEXT_NOT_FOUND = "Это действие не найдено."
TEXT_EXPIRED = "Подтверждение просрочено (действует 5 минут). Повторите запрос."
TEXT_CANCELLED = "Создание напоминания отменено."
TEXT_ALREADY_CANCELLED = "Это действие уже отменено."
TEXT_IN_PROGRESS = "Напоминание уже создаётся, подождите."
REASONS = {
    "timeout": "MCP-сервер не ответил вовремя",
    "unavailable": "MCP-сервер недоступен",
    "invalid_arguments": "время или текст напоминания больше не подходят",
    "untrusted_caller": "сервер не принял подтверждение",
    "bad_result": "сервер вернул некорректный ответ",
}


def parse_action_id(raw: str) -> str | None:
    try:
        return str(uuid.UUID(raw))
    except ValueError:
        return None


class ReminderConfirmations:
    def __init__(
        self,
        repository: Repository,
        gateway: McpGateway,
        *,
        trust_secret: str,
        clock: Callable[[], datetime],
    ):
        self.repository = repository
        self.gateway = gateway
        self.trust_secret = trust_secret
        self.clock = clock

    async def _status_text(self, chat_id: int, action_id: str) -> str:
        # Чужое действие неотличимо от несуществующего: запрос ограничен chat_id.
        action = await self.repository.get_action(chat_id, action_id)
        if action is None:
            return TEXT_NOT_FOUND
        if action.status == "done":
            return f"Напоминание №{action.reminder_id} уже создано."
        if action.status == "cancelled":
            return TEXT_ALREADY_CANCELLED
        if action.status == "executing":
            return TEXT_IN_PROGRESS
        if action.status == "pending":  # срок истёк
            await self.repository.expire_action(chat_id, action_id)
        return TEXT_EXPIRED

    async def confirm(self, chat_id: int, raw_id: str) -> str:
        action_id = parse_action_id(raw_id)
        if action_id is None:
            return TEXT_NOT_FOUND
        now = self.clock()
        action = await self.repository.claim_action(chat_id, action_id, now)
        if action is None:
            return await self._status_text(chat_id, action_id)
        arguments = {
            "text": action.text,
            "remind_at": action.remind_at.astimezone(ZoneInfo(action.timezone)).isoformat(),
        }
        meta = trust.sign(
            self.trust_secret,
            owner_chat_id=chat_id,  # из Telegram update, не из ответа модели
            action_id=action.id,
            timezone=action.timezone,
            arguments=arguments,
            expires_at=int((now + timedelta(minutes=1)).timestamp()),
        )
        try:
            result = await self.gateway.call("add_reminder", arguments, meta=meta)
        except ToolCallError as exc:
            await self.repository.release_action(action.id)
            logger.warning("Напоминание не создано: %s", exc.code)
            reason = REASONS.get(exc.code, "ошибка инструмента")
            return (
                f"Напоминание не создано: {reason}. Можно нажать «Подтвердить» ещё раз, "
                "пока подтверждение действует."
            )
        await self.repository.finish_action(action.id, int(result["reminder_id"]))
        moment = datetime.fromisoformat(result["remind_at"])
        return (
            f"Напоминание №{result['reminder_id']} создано: {result['text']} — "
            f"{format_local(moment, result['timezone'])} ({result['timezone']})."
        )

    async def cancel(self, chat_id: int, raw_id: str) -> str:
        action_id = parse_action_id(raw_id)
        if action_id is None:
            return TEXT_NOT_FOUND
        if await self.repository.cancel_action(chat_id, action_id):
            return TEXT_CANCELLED
        return await self._status_text(chat_id, action_id)
