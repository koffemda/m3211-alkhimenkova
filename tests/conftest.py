"""Контролируемые замены внешних границ: LLM API, PostgreSQL и Telegram."""

from datetime import UTC, datetime
from itertools import count
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.methods import AnswerCallbackQuery, SendMessage
from aiogram.types import CallbackQuery, Chat, Message, Update, User

from app.assistant import AssistantService
from app.handlers.assistant import create_router
from app.llm import ChatMessage, LLMResult
from app.storage import UserSettings

TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"


class FakeLLM:
    """Возвращает заданные ответы по очереди и запоминает каждый запрос."""

    model = "test/model-1"

    def __init__(self, *answers: str | LLMResult | Exception):
        self.answers = list(answers)
        self.calls: list[tuple[list[ChatMessage], float]] = []

    async def complete(self, messages: list[ChatMessage], *, temperature: float) -> LLMResult:
        self.calls.append((list(messages), temperature))
        answer = self.answers.pop(0) if self.answers else "ok"
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, LLMResult) else LLMResult(text=answer)


class MemoryRepository:
    """Повторяет контракт PostgresRepository в памяти."""

    def __init__(self):
        self.settings: dict[int, UserSettings] = {}
        self.messages: dict[int, list[ChatMessage]] = {}

    async def get_settings(self, chat_id: int) -> UserSettings:
        return self.settings.get(chat_id, UserSettings())

    async def set_mode(self, chat_id: int, mode: str) -> None:
        current = await self.get_settings(chat_id)
        self.settings[chat_id] = UserSettings(mode=mode, temperature=current.temperature)
        self.messages.pop(chat_id, None)

    async def set_temperature(self, chat_id: int, temperature: float) -> None:
        current = await self.get_settings(chat_id)
        self.settings[chat_id] = UserSettings(mode=current.mode, temperature=temperature)

    async def clear_history(self, chat_id: int) -> None:
        self.messages.pop(chat_id, None)

    async def load_history(self, chat_id: int, limit: int) -> list[ChatMessage]:
        return list(self.messages.get(chat_id, [])[-limit:])

    async def save_exchange(self, chat_id: int, user_text: str, answer: str, *, keep: int) -> None:
        history = self.messages.setdefault(chat_id, [])
        history += [ChatMessage("user", user_text), ChatMessage("assistant", answer)]
        del history[:-keep]


class TelegramHarness:
    """Прогоняет update через настоящий Dispatcher; HTTP-сессия бота подменена."""

    def __init__(self, assistant: AssistantService):
        self.bot = Bot(TOKEN)
        self.bot.session = AsyncMock()
        self.dispatcher = Dispatcher()
        self.dispatcher.include_router(create_router())
        self.assistant = assistant
        self.ids = count(1)

    async def send(self, chat_id: int, text: str, chat_type: str = "private") -> list[str]:
        message = Message(
            message_id=next(self.ids),
            date=datetime.now(UTC),
            chat=Chat(id=chat_id, type=chat_type),
            from_user=User(id=chat_id, is_bot=False, first_name="Студент"),
            text=text,
        )
        return await self._feed(Update(update_id=next(self.ids), message=message))

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


@pytest.fixture
def repository() -> MemoryRepository:
    return MemoryRepository()


@pytest.fixture
def make_bot(repository):
    """make_bot(*ответы_модели, max_messages=…, max_chars=…) -> (harness, llm)."""

    def factory(*answers, max_messages: int = 20, max_chars: int = 12000):
        llm = FakeLLM(*answers)
        assistant = AssistantService(
            repository, llm, max_messages=max_messages, max_chars=max_chars
        )
        return TelegramHarness(assistant), llm

    return factory
