"""Обязательные сценарии ЛР1 через настоящий Dispatcher aiogram, без Telegram и LLM API."""

import pytest
from aiogram.methods import SendMessage

from app.assistant import (
    TEXT_BAD_TEMPERATURE,
    TEXT_EMPTY,
    TEXT_RATE_LIMIT,
    TEXT_REJECTED,
    TEXT_RESET,
    TEXT_TIMEOUT,
    TEXT_TOO_LONG,
    TEXT_TRUNCATED,
    TEXT_UNAVAILABLE,
    start_text,
)
from app.handlers.assistant import TEXT_ONLY, TEXT_UNKNOWN_COMMAND
from app.llm import (
    ChatMessage,
    LLMBadResponseError,
    LLMResult,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.prompts import MODES, REVIEW_PROMPT, STUDY_PROMPT, TRANSLATE_EXAMPLES, TRANSLATE_PROMPT
from app.storage import UserSettings

ALICE, BOB = 1001, 2002


def roles(messages: list[ChatMessage]) -> list[str]:
    return [message.role for message in messages]


# 1. Обычное сообщение передаётся модели с инструкцией активного режима.


async def test_first_message_uses_study_prompt_and_default_temperature(make_bot):
    # Arrange
    harness, llm = make_bot("Указатель хранит адрес.")
    # Act
    sent = await harness.send(ALICE, "Что такое указатель?")
    # Assert
    assert sent == ["Указатель хранит адрес."]
    messages, temperature = llm.calls[0]
    assert messages == [
        ChatMessage("system", STUDY_PROMPT),
        ChatMessage("user", "Что такое указатель?"),
    ]
    assert temperature == 0.3
    assert harness.sent_methods(SendMessage)[0].parse_mode is None


async def test_translate_mode_sends_its_prompt_few_shot_and_history_in_order(make_bot, repository):
    # Arrange
    harness, llm = make_bot("Good morning!", "Good evening!")
    await harness.send(ALICE, "/translate")
    await harness.send(ALICE, "на английский: Доброе утро!")
    # Act
    await harness.send(ALICE, "на английский: Добрый вечер!")
    # Assert
    messages, _ = llm.calls[1]
    few_shot = [text for pair in TRANSLATE_EXAMPLES for text in pair]
    assert messages[0] == ChatMessage("system", TRANSLATE_PROMPT)
    assert [m.content for m in messages[1 : 1 + len(few_shot)]] == few_shot
    assert messages[1 + len(few_shot) :] == [
        ChatMessage("user", "на английский: Доброе утро!"),
        ChatMessage("assistant", "Good morning!"),
        ChatMessage("user", "на английский: Добрый вечер!"),
    ]
    assert roles(messages[1:]) == ["user", "assistant"] * (len(few_shot) // 2 + 1) + ["user"]
    # Few-shot-примеры и системная инструкция не сохраняются в истории пользователя.
    assert await repository.load_history(ALICE, 20) == messages[1 + len(few_shot) :] + [
        ChatMessage("assistant", "Good evening!")
    ]


async def test_start_lists_commands_and_does_not_call_llm(make_bot):
    # Arrange
    harness, llm = make_bot()
    # Act
    sent = await harness.send(ALICE, "/start")
    # Assert
    assert sent == [start_text()]
    assert all(f"/{name}" in sent[0] for name in [*MODES, "settings", "reset"])
    assert llm.calls == []


@pytest.mark.parametrize("text", ["/unknown", "/translate_all"])
async def test_unknown_command_is_not_sent_to_llm(make_bot, text):
    # Arrange
    harness, llm = make_bot()
    # Act
    sent = await harness.send(ALICE, text)
    # Assert
    assert sent == [TEXT_UNKNOWN_COMMAND]
    assert llm.calls == []


async def test_group_messages_are_ignored(make_bot):
    # Arrange
    harness, llm = make_bot()
    # Act
    sent = await harness.send(-100500, "Привет", chat_type="group")
    # Assert
    assert sent == []
    assert llm.calls == []


async def test_non_text_message_gets_hint(make_bot):
    # Arrange
    from datetime import UTC, datetime

    from aiogram.types import Chat, Message, PhotoSize, Update

    harness, llm = make_bot()
    photo = Message(
        message_id=1,
        date=datetime.now(UTC),
        chat=Chat(id=ALICE, type="private"),
        photo=[PhotoSize(file_id="a", file_unique_id="b", width=1, height=1)],
    )
    # Act
    await harness.dispatcher.feed_update(
        harness.bot, Update(update_id=1, message=photo), assistant=harness.assistant
    )
    # Assert
    assert [m.text for m in harness.sent_methods(SendMessage)] == [TEXT_ONLY]
    assert llm.calls == []


# 2. Истории двух пользователей не смешиваются.


async def test_histories_of_two_users_are_separate(make_bot, repository):
    # Arrange
    harness, llm = make_bot("Ответ Алисе 1", "Ответ Бобу 1", "Ответ Алисе 2")
    await harness.send(ALICE, "Вопрос Алисы 1")
    await harness.send(BOB, "Вопрос Боба 1")
    # Act
    await harness.send(ALICE, "Вопрос Алисы 2")
    # Assert
    alice_request, _ = llm.calls[2]
    assert alice_request[1:] == [
        ChatMessage("user", "Вопрос Алисы 1"),
        ChatMessage("assistant", "Ответ Алисе 1"),
        ChatMessage("user", "Вопрос Алисы 2"),
    ]
    assert await repository.load_history(BOB, 20) == [
        ChatMessage("user", "Вопрос Боба 1"),
        ChatMessage("assistant", "Ответ Бобу 1"),
    ]


# 3. Сокращение истории при превышении ограничения (через бота; детали — test_context.py).


async def test_history_is_trimmed_by_message_count_keeping_order_and_current_query(make_bot):
    # Arrange
    harness, llm = make_bot(*[f"О{i}" for i in range(1, 6)], max_messages=4)
    for i in range(1, 5):
        await harness.send(ALICE, f"В{i}")
    # Act
    await harness.send(ALICE, "В5")
    # Assert
    messages, _ = llm.calls[-1]
    assert [m.content for m in messages[1:]] == ["В3", "О3", "В4", "О4", "В5"]
    assert messages[0].role == "system"


async def test_too_long_message_is_rejected_before_llm(make_bot):
    # Arrange
    harness, llm = make_bot(max_chars=8000)
    # Act
    sent = await harness.send(ALICE, "x" * 7000)
    # Assert
    assert sent == [TEXT_TOO_LONG]
    assert llm.calls == []


# 4. Смена режима очищает историю и сохраняет новую настройку.


async def test_mode_switch_clears_history_and_persists_mode(make_bot, repository):
    # Arrange
    harness, llm = make_bot("Ответ", "Ревью")
    await harness.send(ALICE, "Вопрос по учёбе")
    # Act
    sent = await harness.send(ALICE, "/review")
    await harness.send(ALICE, "int main() {}")
    # Assert
    assert sent == [
        "Режим «Код-ревью» включён, история диалога очищена.\n"
        "Пришлите фрагмент кода и, по желанию, что он должен делать."
    ]
    assert (await repository.get_settings(ALICE)).mode == "review"
    messages, _ = llm.calls[1]
    assert messages[0] == ChatMessage("system", REVIEW_PROMPT)
    assert "Вопрос по учёбе" not in [m.content for m in messages]
    assert len(llm.calls) == 2  # команда /review в модель не отправлялась


async def test_mode_switch_keeps_temperature(make_bot, repository):
    # Arrange
    harness, _ = make_bot()
    await harness.send(ALICE, "/settings 1.0")
    # Act
    await harness.send(ALICE, "/translate")
    # Assert
    assert await repository.get_settings(ALICE) == UserSettings("translate", 1.0)


# 5. /reset очищает только историю вызвавшего пользователя.


async def test_reset_clears_only_caller_history_and_keeps_settings(make_bot, repository):
    # Arrange
    harness, _ = make_bot("А", "Б")
    await harness.send(ALICE, "/translate")
    await harness.send(ALICE, "/settings 0.7")
    await harness.send(ALICE, "Текст Алисы")
    await harness.send(BOB, "Текст Боба")
    # Act
    sent = await harness.send(ALICE, "/reset")
    # Assert
    assert sent == [TEXT_RESET]
    assert await repository.load_history(ALICE, 20) == []
    assert len(await repository.load_history(BOB, 20)) == 2
    assert await repository.get_settings(ALICE) == UserSettings("translate", 0.7)


# 6. /settings принимает разрешённое и отклоняет некорректное значение.


async def test_settings_shows_mode_model_and_temperature(make_bot):
    # Arrange
    harness, _ = make_bot()
    # Act
    sent = await harness.send(ALICE, "/settings")
    # Assert
    assert sent == [
        "Текущие настройки:\n"
        "Режим: /study — Учёба\n"
        "Модель: test/model-1\n"
        "temperature: 0.3\n\n"
        "Выберите новое значение temperature кнопкой ниже или командой, например /settings 0.7."
    ]
    keyboard = harness.sent_methods(SendMessage)[0].reply_markup.inline_keyboard[0]
    assert [button.callback_data for button in keyboard] == [
        "temperature:0.0",
        "temperature:0.3",
        "temperature:0.7",
        "temperature:1.0",
    ]


@pytest.mark.parametrize("value", ["0.7", "0,7"])
async def test_allowed_temperature_is_saved_and_applied_to_next_request(
    make_bot, repository, value
):
    # Arrange
    harness, llm = make_bot("ответ")
    # Act
    sent = await harness.send(ALICE, f"/settings {value}")
    await harness.send(ALICE, "Вопрос")
    # Assert
    assert sent == ["temperature = 0.7 сохранена и будет применена со следующего запроса."]
    assert llm.calls[0][1] == 0.7
    assert (await repository.get_settings(BOB)).temperature == 0.3  # другие не затронуты


@pytest.mark.parametrize("value", ["0.5", "2", "-1", "abc", "nan", "inf"])
async def test_invalid_temperature_is_rejected_before_api(make_bot, repository, value):
    # Arrange
    harness, llm = make_bot()
    # Act
    sent = await harness.send(ALICE, f"/settings {value}")
    # Assert
    assert sent == [TEXT_BAD_TEMPERATURE]
    assert (await repository.get_settings(ALICE)).temperature == 0.3
    assert llm.calls == []


async def test_temperature_button_is_validated(make_bot, repository):
    # Arrange
    harness, _ = make_bot()
    # Act
    accepted = await harness.press(ALICE, "temperature:1.0")
    rejected = await harness.press(ALICE, "temperature:9.9")
    # Assert
    assert accepted == ["temperature = 1.0 сохранена и будет применена со следующего запроса."]
    assert rejected == [TEXT_BAD_TEMPERATURE]
    assert (await repository.get_settings(ALICE)).temperature == 1.0
    assert len(harness.callback_answers()) == 1


# 7. Ошибка LLM преобразуется в безопасное пользовательское сообщение.


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (LLMTimeoutError("истёк таймаут"), TEXT_TIMEOUT),
        (LLMUnavailableError("HTTP 503", status=503), TEXT_UNAVAILABLE),
        (LLMUnavailableError("сетевая ошибка", status=None), TEXT_UNAVAILABLE),
        (LLMUnavailableError("HTTP 429", status=429), TEXT_RATE_LIMIT),
        (LLMUnavailableError("HTTP 401", status=401), TEXT_REJECTED),
        (LLMBadResponseError("в ответе нет текста"), TEXT_EMPTY),
    ],
)
async def test_llm_error_becomes_safe_message_and_is_not_saved(
    make_bot, repository, error, expected
):
    # Arrange
    harness, _ = make_bot(error)
    # Act
    sent = await harness.send(ALICE, "Вопрос")
    # Assert
    assert sent == [expected]
    assert await repository.load_history(ALICE, 20) == []
    assert "HTTP" not in sent[0] and "Traceback" not in sent[0]


async def test_unexpected_error_gets_generic_answer(make_bot, repository, monkeypatch):
    # Arrange
    from app.handlers.assistant import TEXT_INTERNAL_ERROR

    harness, _ = make_bot()

    async def broken(*args, **kwargs):
        raise ConnectionError("postgresql://bot:secret@db/bot")

    monkeypatch.setattr(repository, "get_settings", broken)
    # Act
    sent = await harness.send(ALICE, "Вопрос")
    # Assert
    assert sent == [TEXT_INTERNAL_ERROR]


async def test_truncated_answer_is_marked_but_saved_without_service_note(make_bot, repository):
    # Arrange
    harness, _ = make_bot(LLMResult(text="Начало ответа", finish_reason="length"))
    # Act
    sent = await harness.send(ALICE, "Вопрос")
    # Assert
    assert sent == ["Начало ответа" + TEXT_TRUNCATED]
    assert (await repository.load_history(ALICE, 20))[-1].content == "Начало ответа"


# 8. Ответ длиннее 4096 символов разбивается и отправляется без потери текста.


async def test_long_answer_is_split_into_ordered_parts(make_bot):
    # Arrange
    paragraphs = [f"Абзац {i}: " + "слово " * 120 for i in range(20)]
    answer = "\n".join(paragraphs).strip()
    harness, _ = make_bot(answer)
    # Act
    sent = await harness.send(ALICE, "Расскажи подробно")
    # Assert
    assert len(answer) > 4096
    assert len(sent) > 1
    assert all(0 < len(part) <= 4096 for part in sent)
    assert "".join(sent) == answer


async def test_missing_llm_key_gives_clear_message_without_http(repository):
    # Arrange
    from app.assistant import TEXT_NOT_CONFIGURED, AssistantService
    from app.config import Settings
    from app.llm import OpenAICompatibleClient
    from tests.conftest import TelegramHarness

    llm = OpenAICompatibleClient(Settings(bot_token="unused", postgres_password="unused"))
    harness = TelegramHarness(AssistantService(repository, llm, max_messages=20, max_chars=12000))
    # Act
    sent = await harness.send(ALICE, "Вопрос")
    # Assert
    assert sent == [TEXT_NOT_CONFIGURED]
    assert await repository.load_history(ALICE, 20) == []
    await llm.close()
