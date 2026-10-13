"""ЛР2 на настоящем PostgreSQL: подтверждения, идемпотентность, изоляция пользователей.

RUN_INTEGRATION=1 python -m pytest -m integration -v
Фиктивное хранилище не доказывает защиту от гонок, поэтому эти проверки идут на реальной БД.
"""

import asyncio
import os
from datetime import timedelta

import pytest

from app.confirmations import TEXT_IN_PROGRESS, ReminderConfirmations
from app.mcp_gateway import McpGateway
from app.storage import AgentEvent, PostgresRepository, apply_migrations
from mcp_server.reminders import PostgresReminderStore, ReminderConflictError
from mcp_server.schedule import ScheduleSource
from mcp_server.server import create_server
from tests.conftest import NOW, SECRET, FakeClock, FakeWeather

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="Включите RUN_INTEGRATION=1 для тестов с PostgreSQL",
    ),
]

ALICE, BOB = 1001, 2002
REMIND_AT = NOW + timedelta(days=1, hours=9, minutes=30)


@pytest.fixture
async def repository(pool):
    await apply_migrations(pool)
    return PostgresRepository(pool)


async def prepare(repository, chat_id=ALICE, update_id=1):
    return await repository.prepare_action(
        chat_id,
        update_id,
        "отправить отчёт",
        REMIND_AT,
        "Europe/Moscow",
        NOW + timedelta(minutes=5),
    )


async def test_repeated_update_creates_one_prepared_action(repository, pool):
    # Act: один и тот же update обрабатывается дважды одновременно.
    results = await asyncio.gather(prepare(repository), prepare(repository))
    # Assert
    (first, created_first), (second, created_second) = results
    assert first.id == second.id
    assert sorted([created_first, created_second]) == [False, True]
    assert await pool.fetchval("SELECT count(*) FROM pending_actions") == 1


async def test_concurrent_double_confirmation_creates_one_reminder(
    repository, pool, dsn, schedule_file
):
    # Arrange: настоящий MCP-сервер пишет в ту же БД.
    clock = FakeClock()
    store = PostgresReminderStore({"dsn": dsn})
    server = create_server(
        weather=FakeWeather(),
        schedule=ScheduleSource(schedule_file, "Europe/Moscow", clock),
        reminders=store,
        trust_secret=SECRET,
        clock=clock,
    )
    gateway = McpGateway(server, call_timeout=5)
    await gateway.start()
    confirmations = ReminderConfirmations(repository, gateway, trust_secret=SECRET, clock=clock)
    action, _ = await prepare(repository)
    try:
        # Act: два нажатия «Подтвердить» одновременно (без блокировки процесса бота).
        answers = await asyncio.gather(
            confirmations.confirm(ALICE, action.id), confirmations.confirm(ALICE, action.id)
        )
        repeat = await confirmations.confirm(ALICE, action.id)
    finally:
        await gateway.stop()
        await store.close()
    # Assert
    created = [a for a in answers if a.startswith("Напоминание №") and "создано:" in a]
    assert len(created) == 1
    assert set(answers) - set(created) <= {TEXT_IN_PROGRESS, created[0].split(":")[0]}
    assert repeat.endswith("уже создано.")
    rows = await pool.fetch("SELECT owner_chat_id, text, idempotency_key FROM reminders")
    assert [(r["owner_chat_id"], r["text"], r["idempotency_key"]) for r in rows] == [
        (ALICE, "отправить отчёт", action.id)
    ]
    status = await pool.fetchval(
        "SELECT status FROM pending_actions WHERE id = $1::uuid", action.id
    )
    assert status == "done"


async def test_reminder_idempotency_key_is_enforced_by_database(repository, pool, dsn):
    # Arrange
    store = PostgresReminderStore({"dsn": dsn})
    try:
        # Act
        first = await store.add(
            owner_chat_id=ALICE, key="k-1", text="т", remind_at=REMIND_AT, timezone="Europe/Moscow"
        )
        second = await store.add(
            owner_chat_id=ALICE, key="k-1", text="т", remind_at=REMIND_AT, timezone="Europe/Moscow"
        )
        with pytest.raises(ReminderConflictError):
            await store.add(
                owner_chat_id=BOB,
                key="k-1",
                text="т",
                remind_at=REMIND_AT,
                timezone="Europe/Moscow",
            )
    finally:
        await store.close()
    # Assert
    assert (first.created, second.created) == (True, False)
    assert first.reminder_id == second.reminder_id
    assert await pool.fetchval("SELECT count(*) FROM reminders") == 1


async def test_user_cannot_touch_other_users_action(repository):
    # Arrange
    action, _ = await prepare(repository, ALICE)
    # Act / Assert
    assert await repository.get_action(BOB, action.id) is None
    assert await repository.claim_action(BOB, action.id, NOW) is None
    assert await repository.cancel_action(BOB, action.id) is False
    assert (await repository.get_action(ALICE, action.id)).status == "pending"


async def test_expired_action_cannot_be_claimed(repository):
    # Arrange
    action, _ = await prepare(repository)
    # Act
    claimed = await repository.claim_action(ALICE, action.id, NOW + timedelta(minutes=6))
    await repository.expire_action(ALICE, action.id)
    # Assert
    assert claimed is None
    assert (await repository.get_action(ALICE, action.id)).status == "expired"


async def test_audit_events_are_isolated_and_ordered(repository):
    # Arrange
    def event(action: str) -> AgentEvent:
        return AgentEvent("r", action, None, "—", "—", "—", "причина", 0, 5)

    await repository.add_event(ALICE, event("respond"))
    await repository.add_event(ALICE, event("clarify"))
    # Act
    alice = await repository.last_event(ALICE)
    bob = await repository.last_event(BOB)
    # Assert
    assert alice.action == "clarify" and alice.created_at is not None
    assert bob is None


async def test_timezone_is_persisted(repository):
    # Act
    await repository.set_timezone(ALICE, "Asia/Yekaterinburg")
    await repository.set_mode(ALICE, "review")
    # Assert
    settings = await repository.get_settings(ALICE)
    assert (settings.mode, settings.timezone) == ("review", "Asia/Yekaterinburg")
