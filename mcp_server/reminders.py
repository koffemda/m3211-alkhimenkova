"""Запись напоминаний в PostgreSQL с идемпотентностью по ключу действия.

Таблицу reminders создают миграции приложения (app/storage.py); сервер только
пишет в неё. Повтор того же подтверждённого действия (тот же idempotency_key)
возвращает уже созданную запись с created=false.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import asyncpg


class ReminderConflictError(Exception):
    """Ключ уже использован другим владельцем."""


@dataclass(frozen=True)
class StoredReminder:
    reminder_id: int
    text: str
    remind_at: datetime
    created: bool


class ReminderStore(Protocol):
    async def add(
        self, *, owner_chat_id: int, key: str, text: str, remind_at: datetime, timezone: str
    ) -> StoredReminder: ...


class PostgresReminderStore:
    def __init__(self, dsn_kwargs: dict):
        self._dsn_kwargs = dsn_kwargs
        self._pool: asyncpg.Pool | None = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                **self._dsn_kwargs, min_size=1, max_size=2, timeout=5, command_timeout=5
            )
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()

    async def add(
        self, *, owner_chat_id: int, key: str, text: str, remind_at: datetime, timezone: str
    ) -> StoredReminder:
        pool = await self._get_pool()
        async with pool.acquire() as connection, connection.transaction():
            new_id = await connection.fetchval(
                "INSERT INTO reminders (owner_chat_id, idempotency_key, text, remind_at, timezone)"
                " VALUES ($1, $2, $3, $4, $5)"
                " ON CONFLICT (idempotency_key) DO NOTHING RETURNING id",
                owner_chat_id,
                key,
                text,
                remind_at,
                timezone,
            )
            row = await connection.fetchrow(
                "SELECT id, owner_chat_id, text, remind_at FROM reminders"
                " WHERE idempotency_key = $1",
                key,
            )
        if row is None or row["owner_chat_id"] != owner_chat_id:
            raise ReminderConflictError
        return StoredReminder(row["id"], row["text"], row["remind_at"], created=new_id is not None)


class MemoryReminderStore:
    """Хранилище для тестов и запуска из MCP Inspector без БД."""

    def __init__(self):
        self.rows: dict[str, tuple[int, int, str, datetime, str]] = {}

    async def add(
        self, *, owner_chat_id: int, key: str, text: str, remind_at: datetime, timezone: str
    ) -> StoredReminder:
        created = key not in self.rows
        if created:
            self.rows[key] = (len(self.rows) + 1, owner_chat_id, text, remind_at, timezone)
        reminder_id, owner, stored_text, stored_at, _ = self.rows[key]
        if owner != owner_chat_id:
            raise ReminderConflictError
        return StoredReminder(reminder_id, stored_text, stored_at, created)
