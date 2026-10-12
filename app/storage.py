"""Работа с PostgreSQL: миграции схемы, настройки пользователей и история диалога.

Весь SQL приложения находится здесь. Пользовательские значения передаются только
параметрами $1, $2, … Данные разных пользователей разделены по chat_id.
"""

from dataclasses import dataclass
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


class Repository(Protocol):
    async def get_settings(self, chat_id: int) -> UserSettings: ...

    async def set_mode(self, chat_id: int, mode: str) -> None: ...

    async def set_temperature(self, chat_id: int, temperature: float) -> None: ...

    async def clear_history(self, chat_id: int) -> None: ...

    async def load_history(self, chat_id: int, limit: int) -> list[ChatMessage]: ...

    async def save_exchange(
        self, chat_id: int, user_text: str, answer: str, *, keep: int
    ) -> None: ...


class PostgresRepository:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def get_settings(self, chat_id: int) -> UserSettings:
        row = await self.pool.fetchrow(
            "SELECT mode, temperature FROM user_settings WHERE chat_id = $1", chat_id
        )
        if row is None:
            return UserSettings()
        return UserSettings(mode=row["mode"], temperature=row["temperature"])

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
