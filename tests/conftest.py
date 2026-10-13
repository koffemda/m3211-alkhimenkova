"""Контролируемые замены внешних границ: LLM API, PostgreSQL, Telegram, погодный API.

MCP-сервер в тестах настоящий (create_server), но подключается in-process,
а погода, часы и хранилище напоминаний подменены детерминированными фикстурами.
"""

import json
import os
import secrets
import socket
import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from itertools import count
from pathlib import Path
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import asyncpg
import pytest
from aiogram import Bot, Dispatcher
from aiogram.methods import AnswerCallbackQuery, SendMessage
from aiogram.types import CallbackQuery, Chat, Message, Update, User

from app.agent import Agent
from app.assistant import AssistantService
from app.confirmations import ReminderConfirmations
from app.handlers.assistant import create_router
from app.llm import ChatMessage, LLMResult, ToolCall
from app.mcp_gateway import McpGateway
from app.storage import AgentEvent, PendingAction, UserSettings
from mcp_server.reminders import MemoryReminderStore
from mcp_server.schedule import ScheduleSource
from mcp_server.schemas import CurrentWeather, Location, WeatherOutput
from mcp_server.server import create_server
from mcp_server.weather import WeatherError
from scripts.common import CommandError, run_command

TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"
MSK = ZoneInfo("Europe/Moscow")
# Время эксперимента из docs/evaluation/lab2-routing.json.
NOW = datetime(2026, 10, 12, 9, 0, tzinfo=MSK)
SECRET = "test-trust-secret"


def tool_call(name: str, call_id: str = "call-1", **arguments) -> LLMResult:
    """Ответ модели с одним нативным вызовом инструмента."""
    return LLMResult(text="", tool_calls=(ToolCall(call_id, name, json.dumps(arguments)),))


class FakeLLM:
    """Возвращает заданные ответы по очереди и запоминает каждый запрос."""

    model = "test/model-1"

    def __init__(self, *answers: str | LLMResult | Exception):
        self.answers = list(answers)
        self.calls: list[tuple[list[ChatMessage], float]] = []
        self.tools: list[list[dict] | None] = []

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float, tools: list[dict] | None = None
    ) -> LLMResult:
        self.calls.append((list(messages), temperature))
        self.tools.append(tools)
        answer = self.answers.pop(0) if self.answers else "ok"
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, LLMResult) else LLMResult(text=answer)


class FakeClock:
    def __init__(self, now: datetime = NOW):
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta) -> None:
        self.now += timedelta(**delta)


class MemoryRepository:
    """Повторяет контракт PostgresRepository в памяти (SQL проверяют интеграционные тесты)."""

    def __init__(self):
        self.settings: dict[int, UserSettings] = {}
        self.messages: dict[int, list[ChatMessage]] = {}
        self.events: dict[int, list[AgentEvent]] = {}
        self.actions: dict[str, tuple[int, int, PendingAction]] = {}  # id -> (chat, update, act)

    async def get_settings(self, chat_id: int) -> UserSettings:
        return self.settings.get(chat_id, UserSettings())

    async def set_mode(self, chat_id: int, mode: str) -> None:
        current = await self.get_settings(chat_id)
        self.settings[chat_id] = replace(current, mode=mode)
        self.messages.pop(chat_id, None)

    async def set_temperature(self, chat_id: int, temperature: float) -> None:
        current = await self.get_settings(chat_id)
        self.settings[chat_id] = replace(current, temperature=temperature)

    async def set_timezone(self, chat_id: int, timezone: str) -> None:
        current = await self.get_settings(chat_id)
        self.settings[chat_id] = replace(current, timezone=timezone)

    async def clear_history(self, chat_id: int) -> None:
        self.messages.pop(chat_id, None)

    async def load_history(self, chat_id: int, limit: int) -> list[ChatMessage]:
        return list(self.messages.get(chat_id, [])[-limit:])

    async def save_exchange(self, chat_id: int, user_text: str, answer: str, *, keep: int) -> None:
        history = self.messages.setdefault(chat_id, [])
        history += [ChatMessage("user", user_text), ChatMessage("assistant", answer)]
        del history[:-keep]

    async def add_event(self, chat_id: int, event: AgentEvent) -> None:
        self.events.setdefault(chat_id, []).append(replace(event, created_at=NOW))

    async def last_event(self, chat_id: int) -> AgentEvent | None:
        return (self.events.get(chat_id) or [None])[-1]

    async def prepare_action(self, chat_id, update_id, text, remind_at, timezone, expires_at):
        for owner, update, action in self.actions.values():
            if (owner, update) == (chat_id, update_id):
                return action, False
        action = PendingAction(
            str(uuid.uuid4()), chat_id, text, remind_at, timezone, "pending", expires_at
        )
        self.actions[action.id] = (chat_id, update_id, action)
        return action, True

    async def get_action(self, chat_id, action_id):
        entry = self.actions.get(action_id)
        return entry[2] if entry and entry[0] == chat_id else None

    def _set(self, action_id, **changes):
        chat, update, action = self.actions[action_id]
        self.actions[action_id] = (chat, update, replace(action, **changes))
        return self.actions[action_id][2]

    async def claim_action(self, chat_id, action_id, now):
        action = await self.get_action(chat_id, action_id)
        if action and action.status == "pending" and action.expires_at > now:
            return self._set(action_id, status="executing")
        return None

    async def finish_action(self, action_id, reminder_id):
        self._set(action_id, status="done", reminder_id=reminder_id)

    async def release_action(self, action_id):
        if self.actions[action_id][2].status == "executing":
            self._set(action_id, status="pending")

    async def cancel_action(self, chat_id, action_id):
        action = await self.get_action(chat_id, action_id)
        if action and action.status == "pending":
            self._set(action_id, status="cancelled")
            return True
        return False

    async def expire_action(self, chat_id, action_id):
        action = await self.get_action(chat_id, action_id)
        if action and action.status == "pending":
            self._set(action_id, status="expired")


class FakeWeather:
    """Подмена OpenMeteoClient для тестов агента: город → результат или исключение."""

    def __init__(self):
        self.answers: dict[str, WeatherOutput | Exception] = {}
        self.queries: list[str] = []

    async def current(self, city: str) -> WeatherOutput:
        self.queries.append(city)
        answer = self.answers.get(city)
        if isinstance(answer, Exception):
            raise answer
        return answer or WeatherOutput(status="not_found", query=city)


def weather_ok(city: str, temperature: float = 7.5) -> WeatherOutput:
    return WeatherOutput(
        status="ok",
        query=city,
        location=Location(
            name=city, country="Россия", admin1="Санкт-Петербург", latitude=59.94, longitude=30.31
        ),
        current=CurrentWeather(
            temperature_c=temperature,
            condition_code=61,
            condition="небольшой дождь",
            wind_speed=4.2,
            wind_speed_unit="м/с",
            observed_at="2026-10-12T09:00:00+03:00",
            timezone="Europe/Moscow",
        ),
    )


SCHEDULE = {
    "lessons": [
        {"date": "2026-10-12", "start": "10:00", "end": "11:30", "title": "Матанализ",
         "location": "ауд. 2337"},
        {"date": "2026-10-13", "start": "08:20", "end": "09:50", "title": "Дискретная математика",
         "location": "ауд. 2337"},
        {"date": "2026-10-31", "start": "10:00", "end": "11:30", "title": "Хэллоуин-семинар",
         "location": "ауд. 1"},
        {"date": "2026-11-02", "start": "10:00", "end": "11:30", "title": "Ноябрьская лекция",
         "location": "ауд. 2"},
        {"date": "2026-12-31", "start": "12:00", "end": "13:00", "title": "Последняя пара года",
         "location": "ауд. 3"},
        {"date": "2027-01-03", "start": "12:00", "end": "13:00", "title": "Первая пара года",
         "location": "ауд. 4"},
    ]
}  # fmt: skip


@pytest.fixture
def schedule_file(tmp_path) -> Path:
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps(SCHEDULE, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def weather() -> FakeWeather:
    return FakeWeather()


@pytest.fixture
def reminder_store() -> MemoryReminderStore:
    return MemoryReminderStore()


@pytest.fixture
def mcp_server(weather, schedule_file, reminder_store, clock):
    return create_server(
        weather=weather,
        schedule=ScheduleSource(schedule_file, "Europe/Moscow", clock),
        reminders=reminder_store,
        trust_secret=SECRET,
        clock=clock,
    )


@pytest.fixture
async def gateway(mcp_server):
    gateway = McpGateway(mcp_server, call_timeout=5, connect_timeout=5)
    await gateway.start()
    assert gateway.available
    yield gateway
    await gateway.stop()


class TelegramHarness:
    """Прогоняет update через настоящий Dispatcher; HTTP-сессия бота подменена."""

    def __init__(self, assistant: AssistantService):
        self.bot = Bot(TOKEN)
        self.bot.session = AsyncMock()
        self.dispatcher = Dispatcher()
        self.dispatcher.include_router(create_router())
        self.assistant = assistant
        self.ids = count(1)

    async def send(
        self, chat_id: int, text: str, chat_type: str = "private", update_id: int | None = None
    ) -> list[str]:
        message = Message(
            message_id=next(self.ids),
            date=datetime.now(UTC),
            chat=Chat(id=chat_id, type=chat_type),
            from_user=User(id=chat_id, is_bot=False, first_name="Студент"),
            text=text,
        )
        update_id = next(self.ids) if update_id is None else update_id
        return await self._feed(Update(update_id=update_id, message=message))

    async def press(self, chat_id: int, data: str) -> list[str]:
        message = Message(
            message_id=next(self.ids),
            date=datetime.now(UTC),
            chat=Chat(id=chat_id, type="private"),
            text="Текущие настройки",
        )
        callback = CallbackQuery(
            id=str(next(self.ids)),
            from_user=User(id=chat_id, is_bot=False, first_name="Студент"),
            chat_instance="test",
            message=message,
            data=data,
        )
        return await self._feed(Update(update_id=next(self.ids), callback_query=callback))

    async def _feed(self, update: Update) -> list[str]:
        self.bot.session.reset_mock()
        await self.dispatcher.feed_update(self.bot, update, assistant=self.assistant)
        return [method.text for method in self.sent_methods(SendMessage)]

    def sent_methods(self, kind=SendMessage) -> list:
        return [
            call.args[1]
            for call in self.bot.session.call_args_list
            if isinstance(call.args[1], kind)
        ]

    def callback_answers(self) -> list[AnswerCallbackQuery]:
        return self.sent_methods(AnswerCallbackQuery)

    def last_buttons(self) -> list[str]:
        """callback_data кнопок последнего отправленного сообщения."""
        markup = self.sent_methods(SendMessage)[-1].reply_markup
        return [button.callback_data for row in markup.inline_keyboard for button in row]


@pytest.fixture
def repository() -> MemoryRepository:
    return MemoryRepository()


@pytest.fixture
def make_bot(repository):
    """ЛР1: make_bot(*ответы_модели, max_messages=…, max_chars=…) -> (harness, llm)."""

    def factory(*answers, max_messages: int = 20, max_chars: int = 12000):
        llm = FakeLLM(*answers)
        assistant = AssistantService(
            repository, llm, max_messages=max_messages, max_chars=max_chars
        )
        return TelegramHarness(assistant), llm

    return factory


@pytest.fixture
def make_agent_bot(repository, gateway, clock):
    """ЛР2: бот с агентом и настоящим MCP-сервером in-process -> (harness, llm)."""

    def factory(*answers):
        llm = FakeLLM(*answers)
        agent = Agent(
            llm,
            gateway,
            repository,
            max_messages=20,
            max_chars=12000,
            schedule_timezone="Europe/Moscow",
            clock=clock,
        )
        assistant = AssistantService(
            repository,
            llm,
            max_messages=20,
            max_chars=12000,
            agent=agent,
            confirmations=ReminderConfirmations(
                repository, gateway, trust_secret=SECRET, clock=clock
            ),
            gateway=gateway,
            clock=clock,
        )
        return TelegramHarness(assistant), llm

    return factory


# ---------- PostgreSQL для интеграционных тестов (RUN_INTEGRATION=1) ----------


@pytest.fixture(scope="session")
def dsn():
    """Пустая тестовая БД: TEST_POSTGRES_DSN или собственный контейнер postgres:16."""
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
    await pool.execute(
        "DROP TABLE IF EXISTS chat_messages, user_settings, schema_migrations, reminders,"
        " pending_actions, agent_events"
    )
    yield pool
    await pool.close()


__all__ = ["NOW", "SECRET", "FakeLLM", "WeatherError", "tool_call", "weather_ok"]
