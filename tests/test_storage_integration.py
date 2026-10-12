"""Репозиторий на настоящем PostgreSQL.

RUN_INTEGRATION=1 python -m pytest -m integration -v
Без TEST_POSTGRES_DSN тест поднимает собственный контейнер postgres:16 и удаляет его;
с TEST_POSTGRES_DSN использует указанную пустую тестовую БД (не учебную!).
"""

import os
import secrets
import socket
import time

import asyncpg
import pytest

from app.assistant import AssistantService
from app.llm import ChatMessage
from app.storage import PostgresRepository, UserSettings, apply_migrations
from scripts.common import CommandError, run_command
from tests.conftest import FakeLLM

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="Включите RUN_INTEGRATION=1 для тестов с PostgreSQL",
    ),
]

ALICE, BOB = 1001, 2002


@pytest.fixture(scope="module")
def dsn():
    if os.environ.get("TEST_POSTGRES_DSN"):
        yield os.environ["TEST_POSTGRES_DSN"]
        return
    name = "itmo-test-" + secrets.token_hex(6)
    password = secrets.token_urlsafe(24)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    def docker(*args):
        return run_command(["docker", *args], label="Интеграционный Docker-тест", timeout=180)

    docker(
        "run", "-d", "--rm", "--name", name, "-p", f"127.0.0.1:{port}:5432",
        "-e", "POSTGRES_DB=bot", "-e", "POSTGRES_USER=bot", "-e", f"POSTGRES_PASSWORD={password}",
        "postgres:16-bookworm",
    )  # fmt: skip
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                docker("exec", name, "pg_isready", "-h", "127.0.0.1", "-U", "bot", "-d", "bot")
                break
            except CommandError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)
        yield f"postgresql://bot:{password}@127.0.0.1:{port}/bot"
    finally:
        docker("rm", "-f", name)


@pytest.fixture
async def pool(dsn):
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3)
    await pool.execute("DROP TABLE IF EXISTS chat_messages, user_settings, schema_migrations")
    yield pool
    await pool.close()


async def test_migrations_are_idempotent(pool):
    # Act
    first = await apply_migrations(pool)
    second = await apply_migrations(pool)
    # Assert
    assert first == [1]
    assert second == []


async def test_settings_and_history_survive_restart_and_are_isolated(dsn, pool):
    # Arrange
    await apply_migrations(pool)
    repository = PostgresRepository(pool)
    await repository.set_mode(ALICE, "translate")
    await repository.set_temperature(ALICE, 0.7)
    await repository.save_exchange(ALICE, "вопрос А", "ответ А", keep=20)
    await repository.save_exchange(BOB, "вопрос Б", "ответ Б", keep=20)
    # Act: новое подключение = перезапуск приложения.
    restarted = await asyncpg.create_pool(dsn, min_size=1, max_size=1)
    try:
        fresh = PostgresRepository(restarted)
        settings = await fresh.get_settings(ALICE)
        history = await fresh.load_history(ALICE, 20)
        other = await fresh.get_settings(BOB)
    finally:
        await restarted.close()
    # Assert
    assert settings == UserSettings("translate", 0.7)
    assert history == [ChatMessage("user", "вопрос А"), ChatMessage("assistant", "ответ А")]
    assert other == UserSettings()


async def test_mode_switch_and_reset_clear_only_own_history(pool):
    # Arrange
    await apply_migrations(pool)
    repository = PostgresRepository(pool)
    for chat in (ALICE, BOB):
        await repository.save_exchange(chat, "в", "о", keep=20)
    await repository.set_temperature(ALICE, 1.0)
    # Act
    await repository.set_mode(ALICE, "review")
    after_switch = await repository.load_history(ALICE, 20)
    await repository.save_exchange(ALICE, "в2", "о2", keep=20)
    await repository.clear_history(ALICE)
    # Assert
    assert after_switch == []
    assert await repository.load_history(ALICE, 20) == []
    assert await repository.get_settings(ALICE) == UserSettings("review", 1.0)
    assert len(await repository.load_history(BOB, 20)) == 2


async def test_history_order_limit_and_pruning(pool):
    # Arrange
    await apply_migrations(pool)
    repository = PostgresRepository(pool)
    # Act
    for i in range(1, 6):
        await repository.save_exchange(ALICE, f"в{i}", f"о{i}", keep=6)
    # Assert
    stored = await pool.fetchval("SELECT count(*) FROM chat_messages WHERE chat_id = $1", ALICE)
    assert stored == 6
    latest = await repository.load_history(ALICE, 4)
    assert [m.content for m in latest] == ["в4", "о4", "в5", "о5"]


async def test_database_rejects_unsupported_temperature(pool):
    # Arrange
    await apply_migrations(pool)
    # Act / Assert
    with pytest.raises(asyncpg.CheckViolationError):
        await PostgresRepository(pool).set_temperature(ALICE, 0.5)


async def test_service_with_real_db_keeps_context_after_restart(dsn, pool):
    # Arrange
    await apply_migrations(pool)
    first = AssistantService(
        PostgresRepository(pool), FakeLLM("О1"), max_messages=20, max_chars=12000
    )
    await first.reply(ALICE, "В1")
    # Act: новый сервис и новый пул, как после перезапуска бота.
    restarted = await asyncpg.create_pool(dsn, min_size=1, max_size=1)
    try:
        llm = FakeLLM("О2")
        second = AssistantService(
            PostgresRepository(restarted), llm, max_messages=20, max_chars=12000
        )
        await second.reply(ALICE, "В2")
    finally:
        await restarted.close()
    # Assert
    messages, _ = llm.calls[0]
    assert [m.content for m in messages[1:]] == ["В1", "О1", "В2"]
