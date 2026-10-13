"""MCP-сервер через настоящий MCP-клиент SDK (in-process): схемы, валидация, доверие."""

import json
from datetime import UTC, datetime

import pytest
from mcp import Client

from mcp_server import trust
from mcp_server.schedule import ScheduleSource
from mcp_server.server import create_server
from tests.conftest import MSK, NOW, SECRET, FakeClock, weather_ok

REMINDER = {"text": "отправить отчёт", "remind_at": "2026-10-13T18:30:00+03:00"}


def signed(arguments: dict, *, owner: int = 42, action: str = "act-1", ttl: int = 60) -> dict:
    return trust.sign(
        SECRET,
        owner_chat_id=owner,
        action_id=action,
        timezone="Europe/Moscow",
        arguments=arguments,
        expires_at=int(NOW.timestamp()) + ttl,
    )


@pytest.fixture
def client(mcp_server):
    # Клиент открывается внутри теста: anyio требует входа и выхода в одной задаче.
    return mcp_server


async def call(server, name, arguments, meta=None) -> tuple[bool, dict]:
    async with Client(server) as client:
        result = await client.call_tool(name, arguments, meta=meta)
    return result.is_error, result.structured_content


# ---------- обнаружение ----------


async def test_discovery_lists_three_tools_with_strict_schemas_and_resource(mcp_server):
    # Act
    async with Client(mcp_server) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        resources = (await client.list_resources()).resources
        capabilities = client.server_capabilities
    # Assert
    assert set(tools) == {"get_weather", "get_schedule", "add_reminder"}
    for tool in tools.values():
        assert tool.input_schema["additionalProperties"] is False
        assert tool.output_schema["type"] == "object"
    assert tools["get_weather"].input_schema["properties"]["city"]["maxLength"] == 100
    assert tools["get_schedule"].input_schema["required"] == ["date"]
    assert set(tools["add_reminder"].input_schema["properties"]) == {"text", "remind_at"}
    assert tools["add_reminder"].annotations.read_only_hint is False
    assert tools["get_weather"].annotations.read_only_hint is True
    assert capabilities.tools is not None
    assert [str(r.uri) for r in resources] == ["schedule://current-week"]


# ---------- get_schedule (сценарии 4–6) ----------


async def test_schedule_for_date_with_lessons(client):
    # Act
    is_error, data = await call(client, "get_schedule", {"date": "2026-10-13"})
    # Assert
    assert is_error is False
    assert data == {
        "date": "2026-10-13",
        "timezone": "Europe/Moscow",
        "lessons": [
            {
                "start": "2026-10-13T08:20:00+03:00",
                "end": "2026-10-13T09:50:00+03:00",
                "title": "Дискретная математика",
                "location": "ауд. 2337",
            }
        ],
        "source": "Учебное расписание (файл сервера)",
    }


async def test_schedule_for_free_day_is_empty_list(client):
    # Act
    is_error, data = await call(client, "get_schedule", {"date": "2026-10-18"})
    # Assert
    assert is_error is False
    assert data["lessons"] == []


@pytest.mark.parametrize("value", ["13.10.2026", "2026-10-13T10:00", "2026-02-30", "завтра", ""])
async def test_schedule_rejects_bad_date(client, value):
    # Act
    is_error, data = await call(client, "get_schedule", {"date": value})
    # Assert
    assert is_error is True
    assert data["error"]["code"] == "invalid_arguments"
    assert data["error"]["message"].startswith("date:")


# ---------- неизвестный инструмент и лишние аргументы (сценарий 14) ----------


async def test_unknown_tool_is_rejected(client):
    # Act
    is_error, data = await call(client, "read_system_prompt", {})
    # Assert
    assert is_error is True
    assert data["error"]["code"] == "unknown_tool"


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("get_schedule", {"date": "2026-10-13", "path": "/etc/passwd"}),
        ("get_weather", {"city": "Казань", "url": "http://evil"}),
    ],
)
async def test_extra_arguments_are_rejected(client, name, arguments):
    # Act
    is_error, data = await call(client, name, arguments)
    # Assert
    assert is_error is True
    assert data["error"]["code"] == "invalid_arguments"
    assert "лишнее поле" in data["error"]["message"]


async def test_weather_city_length_is_checked_after_strip(client, weather):
    # Act
    is_error, data = await call(client, "get_weather", {"city": "   "})
    # Assert
    assert is_error is True and data["error"]["code"] == "invalid_arguments"
    assert weather.queries == []


# ---------- add_reminder: доверенный владелец (сценарии 7–9) ----------


async def test_reminder_without_trusted_context_is_rejected(client, reminder_store):
    # Act: так выглядит вызов из постороннего клиента (MCP Inspector).
    is_error, data = await call(client, "add_reminder", REMINDER)
    # Assert
    assert is_error is True
    assert data["error"]["code"] == "untrusted_caller"
    assert reminder_store.rows == {}


async def test_owner_in_model_arguments_is_rejected(client, reminder_store):
    # Arrange
    arguments = {**REMINDER, "owner_chat_id": 999}
    # Act
    is_error, data = await call(client, "add_reminder", arguments, signed(arguments))
    # Assert
    assert is_error is True
    assert data["error"]["code"] == "invalid_arguments"
    assert "owner_chat_id: лишнее поле" in data["error"]["message"]
    assert reminder_store.rows == {}


async def test_reminder_is_created_for_trusted_owner_in_future(client, reminder_store):
    # Act
    is_error, data = await call(client, "add_reminder", REMINDER, signed(REMINDER, owner=42))
    # Assert
    assert is_error is False
    assert data == {
        "reminder_id": 1,
        "text": "отправить отчёт",
        "remind_at": "2026-10-13T18:30:00+03:00",
        "timezone": "Europe/Moscow",
        "created": True,
    }
    assert reminder_store.rows["act-1"][1] == 42


async def test_same_idempotency_key_does_not_create_second_record(client, reminder_store):
    # Arrange
    meta = signed(REMINDER, action="same-action")
    await call(client, "add_reminder", REMINDER, meta)
    # Act
    is_error, data = await call(client, "add_reminder", REMINDER, meta)
    # Assert
    assert is_error is False
    assert data["created"] is False and data["reminder_id"] == 1
    assert len(reminder_store.rows) == 1


@pytest.mark.parametrize(
    "arguments",
    [
        {"text": "отчёт", "remind_at": "2026-10-12T08:59:00+03:00"},  # прошлое
        {"text": "   ", "remind_at": "2026-10-13T18:30:00+03:00"},  # пустой текст
        {"text": "x" * 501, "remind_at": "2026-10-13T18:30:00+03:00"},
        {"text": "отчёт", "remind_at": "2026-10-13T18:30:00"},  # без часового пояса
    ],
    ids=["past", "blank", "too-long", "naive"],
)
async def test_invalid_reminder_is_rejected(client, reminder_store, arguments):
    # Act
    is_error, data = await call(client, "add_reminder", arguments, signed(arguments))
    # Assert
    assert is_error is True
    assert data["error"]["code"] == "invalid_arguments"
    assert reminder_store.rows == {}


async def test_signature_does_not_transfer_to_changed_arguments(client, reminder_store):
    # Arrange
    meta = signed(REMINDER)
    changed = {**REMINDER, "remind_at": "2026-10-14T18:30:00+03:00"}
    # Act
    is_error, data = await call(client, "add_reminder", changed, meta)
    # Assert
    assert is_error is True
    assert "отличаются" in data["error"]["message"]
    assert reminder_store.rows == {}


async def test_expired_or_forged_signature_is_rejected(client):
    # Arrange
    expired = signed(REMINDER, ttl=-1)
    forged = signed(REMINDER)
    forged[trust.META_KEY] = {**forged[trust.META_KEY], "owner_chat_id": 7}
    # Act / Assert
    for meta in (expired, forged):
        is_error, data = await call(client, "add_reminder", REMINDER, meta)
        assert is_error is True and data["error"]["code"] == "untrusted_caller"


async def test_server_without_secret_always_refuses_reminders(weather, schedule_file, clock):
    # Arrange: так сервер запускается вручную для MCP Inspector.
    from mcp_server.reminders import MemoryReminderStore

    store = MemoryReminderStore()
    server = create_server(
        weather=weather,
        schedule=ScheduleSource(schedule_file, "Europe/Moscow", clock),
        reminders=store,
        trust_secret=None,
        clock=clock,
    )
    # Act
    is_error, data = await call(server, "add_reminder", REMINDER, signed(REMINDER))
    # Assert
    assert is_error is True and data["error"]["code"] == "untrusted_caller"
    assert store.rows == {}


# ---------- ресурс schedule://current-week (сценарий 12) ----------


async def read_week(server_factory, now: datetime) -> dict:
    async with Client(server_factory(FakeClock(now))) as client:
        result = await client.read_resource("schedule://current-week")
    return json.loads(result.contents[0].text)


@pytest.fixture
def server_with_clock(weather, schedule_file, reminder_store):
    def factory(clock):
        return create_server(
            weather=weather,
            schedule=ScheduleSource(schedule_file, "Europe/Moscow", clock),
            reminders=reminder_store,
            trust_secret=SECRET,
            clock=clock,
        )

    return factory


async def test_week_resource_on_month_boundary(server_with_clock):
    # Act: суббота 31 октября — неделя 26.10–01.11, следующая лекция 02.11 не входит.
    week = await read_week(server_with_clock, datetime(2026, 10, 31, 23, 30, tzinfo=MSK))
    # Assert
    assert (week["week_start"], week["week_end"]) == ("2026-10-26", "2026-11-01")
    assert [day["date"] for day in week["days"]][-2:] == ["2026-10-31", "2026-11-01"]
    titles = [lesson["title"] for day in week["days"] for lesson in day["lessons"]]
    assert titles == ["Хэллоуин-семинар"]


async def test_week_resource_on_year_boundary_uses_schedule_timezone(server_with_clock):
    # Act: 31.12 21:30 UTC — в Москве уже 1 января 2027 (пятница той же недели).
    week = await read_week(server_with_clock, datetime(2026, 12, 31, 21, 30, tzinfo=UTC))
    # Assert
    assert (week["week_start"], week["week_end"]) == ("2026-12-28", "2027-01-03")
    lessons = {day["date"]: [x["title"] for x in day["lessons"]] for day in week["days"]}
    assert lessons["2026-12-31"] == ["Последняя пара года"]
    assert lessons["2027-01-03"] == ["Первая пара года"]
    assert len(week["days"]) == 7


async def test_unknown_resource_is_error(mcp_server):
    # Act / Assert
    async with Client(mcp_server) as client:
        with pytest.raises(Exception, match="не найден"):
            await client.read_resource("schedule://../../etc/passwd")


async def test_weather_result_passes_output_schema(client, weather):
    # Arrange
    weather.answers["Санкт-Петербург"] = weather_ok("Санкт-Петербург")
    # Act
    is_error, data = await call(client, "get_weather", {"city": "Санкт-Петербург"})
    # Assert
    assert is_error is False
    assert data["current"]["temperature_c"] == 7.5
    assert data["current"]["wind_speed_unit"] == "м/с"
