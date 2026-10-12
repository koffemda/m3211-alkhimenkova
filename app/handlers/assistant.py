"""Telegram-обработчики: разбирают команды и передают работу AssistantService."""

import logging

from aiogram import Bot, F, Router
from aiogram.enums import ChatType
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import (
    CallbackQuery,
    ErrorEvent,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from aiogram.utils.chat_action import ChatActionSender

from app.assistant import ALLOWED_TEMPERATURES, AssistantService, format_temperature, start_text
from app.prompts import MODES
from app.telegram import split_message

logger = logging.getLogger("app.handlers")

TEMPERATURE_PREFIX = "temperature:"
TEXT_ONLY = "Я понимаю только текстовые сообщения. Напишите вопрос текстом."
TEXT_UNKNOWN_COMMAND = "Неизвестная команда. Список команд — /start."
TEXT_EMPTY_FALLBACK = "Ответ пуст. Попробуйте переформулировать запрос."
TEXT_INTERNAL_ERROR = "Не удалось обработать сообщение из-за внутренней ошибки. Попробуйте позже."


async def send_long(message: Message, text: str) -> None:
    # parse_mode=None: ответ модели отправляется как обычный текст без разбора разметки.
    for part in split_message(text) or [TEXT_EMPTY_FALLBACK]:
        await message.answer(part, parse_mode=None)


def temperature_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=format_temperature(value),
                    callback_data=TEMPERATURE_PREFIX + format_temperature(value),
                )
                for value in ALLOWED_TEMPERATURES
            ]
        ]
    )


async def start(message: Message) -> None:
    await message.answer(start_text(), parse_mode=None)


async def switch_mode(
    message: Message, command: CommandObject, assistant: AssistantService
) -> None:
    # Сама команда в модель не отправляется.
    text = await assistant.switch_mode(message.chat.id, command.command)
    await message.answer(text, parse_mode=None)


async def reset(message: Message, assistant: AssistantService) -> None:
    await message.answer(await assistant.reset(message.chat.id), parse_mode=None)


async def settings(message: Message, command: CommandObject, assistant: AssistantService) -> None:
    if command.args:
        text = await assistant.set_temperature(message.chat.id, command.args)
        await message.answer(text, parse_mode=None)
        return
    await message.answer(
        await assistant.settings_text(message.chat.id),
        parse_mode=None,
        reply_markup=temperature_keyboard(),
    )


async def choose_temperature(callback: CallbackQuery, assistant: AssistantService) -> None:
    # callback_data приходит от клиента и проверяется так же, как текстовый ввод.
    raw = callback.data.removeprefix(TEMPERATURE_PREFIX)
    text = await assistant.set_temperature(callback.message.chat.id, raw)
    await callback.answer()
    await callback.message.answer(text, parse_mode=None)


async def unknown_command(message: Message) -> None:
    await message.answer(TEXT_UNKNOWN_COMMAND, parse_mode=None)


async def user_text(message: Message, bot: Bot, assistant: AssistantService) -> None:
    # Пока модель думает, Telegram показывает «печатает…» (обновляется каждые 5 с).
    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        answer = await assistant.reply(message.chat.id, message.text)
    await send_long(message, answer)


async def not_text(message: Message) -> None:
    await message.answer(TEXT_ONLY, parse_mode=None)


async def on_error(event: ErrorEvent) -> None:
    # Текст сообщения пользователя не логируем; секреты скрывает фильтр логирования.
    logger.error(
        "Ошибка обработки update %s: %s",
        event.update.update_id,
        type(event.exception).__name__,
        exc_info=event.exception,
    )
    if event.update.message:
        await event.update.message.answer(TEXT_INTERNAL_ERROR, parse_mode=None)
    elif event.update.callback_query:
        await event.update.callback_query.answer(TEXT_INTERNAL_ERROR, show_alert=True)


def create_router() -> Router:
    """Новый роутер на каждый вызов: aiogram не позволяет подключить роутер дважды."""
    router = Router(name="assistant")
    # Бот работает только в личных чатах; группы и каналы игнорируются.
    router.message.filter(F.chat.type == ChatType.PRIVATE)
    router.callback_query.filter(F.message.chat.type == ChatType.PRIVATE)
    # Порядок важен: команды раньше общего обработчика текста.
    router.message.register(start, CommandStart())
    router.message.register(start, Command("help"))
    router.message.register(switch_mode, Command(*MODES))
    router.message.register(reset, Command("reset"))
    router.message.register(settings, Command("settings"))
    router.message.register(unknown_command, F.text.startswith("/"))
    router.message.register(user_text, F.text)
    router.message.register(not_text)
    router.callback_query.register(choose_temperature, F.data.startswith(TEMPERATURE_PREFIX))
    router.errors.register(on_error)
    return router
