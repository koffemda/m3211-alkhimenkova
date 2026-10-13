"""Работа с PostgreSQL: миграции схемы, настройки пользователей и история диалога.

Весь SQL приложения находится здесь. Пользовательские значения передаются только
параметрами $1, $2, … Данные разных пользователей разделены по chat_id.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import asyncpg

from app.llm import ChatMessage
from app.prompts import DEFAULT_MODE

DEFAULT_TEMPERATURE = 0.3

# Миграции только добавляются в конец списка; применённые не редактируются.
MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE user_settings (
            chat_id     BIGINT PRIMARY KEY,
            mode        TEXT NOT NULL,
            temperature DOUBLE PRECISION NOT NULL
                        CHECK (temperature IN (0.0, 0.3, 0.7, 1.0)),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE TABLE chat_messages (
            id         BIGSERIAL PRIMARY KEY,
            chat_id    BIGINT NOT NULL,
            role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content    TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX chat_messages_chat_id_id ON chat_messages (chat_id, id);
        """,
    ),
    (
        2,
        """
        ALTER TABLE user_settings ADD COLUMN timezone TEXT;

        -- Запись делает MCP-сервер (add_reminder); владельца он берёт из доверенного контекста.
        CREATE TABLE reminders (
            id              BIGSERIAL PRIMARY KEY,
            owner_chat_id   BIGINT NOT NULL,
            idempotency_key TEXT NOT NULL UNIQUE,
            text            TEXT NOT NULL CHECK (char_length(text) BETWEEN 1 AND 500),
            remind_at       TIMESTAMPTZ NOT NULL,
            timezone        TEXT NOT NULL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX reminders_owner ON reminders (owner_chat_id, remind_at);

        -- Подготовленные действия, ожидающие подтверждения пользователя.
        CREATE TABLE pending_actions (
            id               UUID PRIMARY KEY,
            chat_id          BIGINT NOT NULL,
            source_update_id BIGINT NOT NULL,
            tool             TEXT NOT NULL,
            text             TEXT NOT NULL,
            remind_at        TIMESTAMPTZ NOT NULL,
            timezone         TEXT NOT NULL,
            status           TEXT NOT NULL CHECK (status IN
                             ('pending', 'executing', 'done', 'cancelled', 'expired')),
            reminder_id      BIGINT,
            created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
            expires_at       TIMESTAMPTZ NOT NULL,
            -- Повторная доставка того же Telegram update не создаёт второе действие.
            UNIQUE (chat_id, source_update_id)
        );

        -- Краткий аудит решений агента; источник для /why.
        CREATE TABLE agent_events (
            id                BIGSERIAL PRIMARY KEY,
            chat_id           BIGINT NOT NULL,
            request_id        TEXT NOT NULL,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
            action            TEXT NOT NULL,
            tool_name         TEXT,
            arguments_summary TEXT NOT NULL,
            validation        TEXT NOT NULL,
            execution         TEXT NOT NULL,
            reason            TEXT NOT NULL,
            mcp_calls         INTEGER NOT NULL,
            duration_ms       INTEGER NOT NULL
        );
        CREATE INDEX agent_events_chat_id_id ON agent_events (chat_id, id);
        """,
    ),
]

# Произвольная константа: блокировка не даёт двум процессам применять миграции одновременно.
MIGRATION_LOCK_ID = 2026_0901


async def apply_migrations(pool: asyncpg.Pool) -> list[int]:
    """Применяет недостающие миграции в одной транзакции; повторный запуск безопасен."""
    async with pool.acquire() as connection, connection.transaction():
        await connection.execute("SELECT pg_advisory_xact_lock($1)", MIGRATION_LOCK_ID)
        await connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version INTEGER PRIMARY KEY,"
            " applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        done = {
            row["version"]
            for row in await connection.fetch("SELECT version FROM schema_migrations")
        }
        applied = []
        for version, sql in MIGRATIONS:
            if version in done:
                continue
            await connection.execute(sql)
            await connection.execute("INSERT INTO schema_migrations (version) VALUES ($1)", version)
            applied.append(version)
        return applied


@dataclass(frozen=True)
class UserSettings:
    mode: str = DEFAULT_MODE
    temperature: float = DEFAULT_TEMPERATURE
    timezone: str | None = None


@dataclass(frozen=True)
class PendingAction:
    id: str
    chat_id: int
    text: str
    remind_at: datetime
    timezone: str
    status: str
    expires_at: datetime
    reminder_id: int | None = None


@dataclass(frozen=True)
class AgentEvent:
    request_id: str
    action: str  # respond | clarify | tool | limit | error
    tool_name: str | None
    arguments_summary: str
    validation: str
    execution: str
    reason: str
    mcp_calls: int
    duration_ms: int
    created_at: datetime | None = None


class Repository(Protocol):
    async def get_settings(self, chat_id: int) -> UserSettings: ...

    async def set_mode(self, chat_id: int, mode: str) -> None: ...

    async def set_temperature(self, chat_id: int, temperature: float) -> None: ...

    async def clear_history(self, chat_id: int) -> None: ...

    async def load_history(self, chat_id: int, limit: int) -> list[ChatMessage]: ...

    async def save_exchange(
        self, chat_id: int, user_text: str, answer: str, *, keep: int
    ) -> None: ...

    async def set_timezone(self, chat_id: int, timezone: str) -> None: ...

    async def add_event(self, chat_id: int, event: AgentEvent) -> None: ...

    async def last_event(self, chat_id: int) -> AgentEvent | None: ...

    async def prepare_action(
        self,
        chat_id: int,
        update_id: int,
        text: str,
        remind_at: datetime,
        timezone: str,
        expires_at: datetime,
    ) -> tuple[PendingAction, bool]: ...

    async def get_action(self, chat_id: int, action_id: str) -> PendingAction | None: ...

    async def claim_action(
        self, chat_id: int, action_id: str, now: datetime
    ) -> PendingAction | None: ...

    async def finish_action(self, action_id: str, reminder_id: int) -> None: ...

    async def release_action(self, action_id: str) -> None: ...

    async def cancel_action(self, chat_id: int, action_id: str) -> bool: ...

    async def expire_action(self, chat_id: int, action_id: str) -> None: ...


class PostgresRepository:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def get_settings(self, chat_id: int) -> UserSettings:
        row = await self.pool.fetchrow(
            "SELECT mode, temperature, timezone FROM user_settings WHERE chat_id = $1", chat_id
        )
        if row is None:
            return UserSettings()
        return UserSettings(
            mode=row["mode"], temperature=row["temperature"], timezone=row["timezone"]
        )

    async def set_mode(self, chat_id: int, mode: str) -> None:
        # Смена режима и очистка истории атомарны: нет состояния «новый режим, старая история».
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute(
                "INSERT INTO user_settings (chat_id, mode, temperature) VALUES ($1, $2, $3) "
                "ON CONFLICT (chat_id) DO UPDATE SET mode = EXCLUDED.mode, updated_at = now()",
                chat_id,
                mode,
                DEFAULT_TEMPERATURE,
            )
            await connection.execute("DELETE FROM chat_messages WHERE chat_id = $1", chat_id)

    async def set_temperature(self, chat_id: int, temperature: float) -> None:
        await self.pool.execute(
            "INSERT INTO user_settings (chat_id, mode, temperature) VALUES ($1, $2, $3) "
            "ON CONFLICT (chat_id) DO UPDATE "
            "SET temperature = EXCLUDED.temperature, updated_at = now()",
            chat_id,
            DEFAULT_MODE,
            temperature,
        )

    async def clear_history(self, chat_id: int) -> None:
        await self.pool.execute("DELETE FROM chat_messages WHERE chat_id = $1", chat_id)

    async def load_history(self, chat_id: int, limit: int) -> list[ChatMessage]:
        rows = await self.pool.fetch(
            "SELECT role, content FROM ("
            " SELECT id, role, content FROM chat_messages WHERE chat_id = $1"
            " ORDER BY id DESC LIMIT $2"
            ") AS latest ORDER BY id",
            chat_id,
            limit,
        )
        return [ChatMessage(row["role"], row["content"]) for row in rows]

    async def save_exchange(self, chat_id: int, user_text: str, answer: str, *, keep: int) -> None:
        """Сохраняет пару «вопрос — ответ» и удаляет сообщения старше последних keep."""
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.executemany(
                "INSERT INTO chat_messages (chat_id, role, content) VALUES ($1, $2, $3)",
                [(chat_id, "user", user_text), (chat_id, "assistant", answer)],
            )
            await connection.execute(
                "DELETE FROM chat_messages WHERE chat_id = $1 AND id < ("
                " SELECT min(id) FROM ("
                "  SELECT id FROM chat_messages WHERE chat_id = $1 ORDER BY id DESC LIMIT $2"
                " ) AS latest)",
                chat_id,
                keep,
            )

    async def set_timezone(self, chat_id: int, timezone: str) -> None:
        await self.pool.execute(
            "INSERT INTO user_settings (chat_id, mode, temperature, timezone)"
            " VALUES ($1, $2, $3, $4)"
            " ON CONFLICT (chat_id) DO UPDATE SET timezone = EXCLUDED.timezone, updated_at = now()",
            chat_id,
            DEFAULT_MODE,
            DEFAULT_TEMPERATURE,
            timezone,
        )

    # ---------- аудит решений агента ----------

    async def add_event(self, chat_id: int, event: AgentEvent) -> None:
        await self.pool.execute(
            "INSERT INTO agent_events (chat_id, request_id, action, tool_name, arguments_summary,"
            " validation, execution, reason, mcp_calls, duration_ms)"
            " VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
            chat_id,
            event.request_id,
            event.action,
            event.tool_name,
            event.arguments_summary,
            event.validation,
            event.execution,
            event.reason,
            event.mcp_calls,
            event.duration_ms,
        )

    async def last_event(self, chat_id: int) -> AgentEvent | None:
        row = await self.pool.fetchrow(
            "SELECT request_id, action, tool_name, arguments_summary, validation, execution,"
            " reason, mcp_calls, duration_ms, created_at FROM agent_events"
            " WHERE chat_id = $1 ORDER BY id DESC LIMIT 1",
            chat_id,
        )
        return AgentEvent(**dict(row)) if row else None

    # ---------- подтверждение действий ----------

    @staticmethod
    def _action(row) -> PendingAction:
        return PendingAction(
            id=str(row["id"]),
            chat_id=row["chat_id"],
            text=row["text"],
            remind_at=row["remind_at"],
            timezone=row["timezone"],
            status=row["status"],
            expires_at=row["expires_at"],
            reminder_id=row["reminder_id"],
        )

    _ACTION_COLUMNS = "id, chat_id, text, remind_at, timezone, status, expires_at, reminder_id"

    async def prepare_action(
        self,
        chat_id: int,
        update_id: int,
        text: str,
        remind_at: datetime,
        timezone: str,
        expires_at: datetime,
    ) -> tuple[PendingAction, bool]:
        """Создаёт действие; для повторного update возвращает уже созданное (created=False)."""
        row = await self.pool.fetchrow(
            "INSERT INTO pending_actions (id, chat_id, source_update_id, tool, text, remind_at,"
            " timezone, status, expires_at)"
            " VALUES ($1, $2, $3, 'add_reminder', $4, $5, $6, 'pending', $7)"
            f" ON CONFLICT (chat_id, source_update_id) DO NOTHING RETURNING {self._ACTION_COLUMNS}",
            uuid.uuid4(),
            chat_id,
            update_id,
            text,
            remind_at,
            timezone,
            expires_at,
        )
        if row is not None:
            return self._action(row), True
        row = await self.pool.fetchrow(
            f"SELECT {self._ACTION_COLUMNS} FROM pending_actions"
            " WHERE chat_id = $1 AND source_update_id = $2",
            chat_id,
            update_id,
        )
        return self._action(row), False

    async def get_action(self, chat_id: int, action_id: str) -> PendingAction | None:
        row = await self.pool.fetchrow(
            f"SELECT {self._ACTION_COLUMNS} FROM pending_actions WHERE id = $1 AND chat_id = $2",
            uuid.UUID(action_id),
            chat_id,
        )
        return self._action(row) if row else None

    async def claim_action(
        self, chat_id: int, action_id: str, now: datetime
    ) -> PendingAction | None:
        """Атомарно переводит pending → executing. Из двух одновременных нажатий
        «Подтвердить» строку получает только одно."""
        row = await self.pool.fetchrow(
            "UPDATE pending_actions SET status = 'executing'"
            " WHERE id = $1 AND chat_id = $2 AND status = 'pending' AND expires_at > $3"
            f" RETURNING {self._ACTION_COLUMNS}",
            uuid.UUID(action_id),
            chat_id,
            now,
        )
        return self._action(row) if row else None

    async def finish_action(self, action_id: str, reminder_id: int) -> None:
        await self.pool.execute(
            "UPDATE pending_actions SET status = 'done', reminder_id = $2 WHERE id = $1",
            uuid.UUID(action_id),
            reminder_id,
        )

    async def release_action(self, action_id: str) -> None:
        # После сбоя MCP действие снова можно подтвердить; дубликат исключает ключ на сервере.
        await self.pool.execute(
            "UPDATE pending_actions SET status = 'pending' WHERE id = $1 AND status = 'executing'",
            uuid.UUID(action_id),
        )

    async def cancel_action(self, chat_id: int, action_id: str) -> bool:
        result = await self.pool.execute(
            "UPDATE pending_actions SET status = 'cancelled'"
            " WHERE id = $1 AND chat_id = $2 AND status = 'pending'",
            uuid.UUID(action_id),
            chat_id,
        )
        return result.endswith(" 1")

    async def expire_action(self, chat_id: int, action_id: str) -> None:
        await self.pool.execute(
            "UPDATE pending_actions SET status = 'expired'"
            " WHERE id = $1 AND chat_id = $2 AND status = 'pending'",
            uuid.UUID(action_id),
            chat_id,
        )
