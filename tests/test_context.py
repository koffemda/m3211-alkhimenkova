"""Сокращение истории и разбиение длинных сообщений — чистые функции."""

import pytest

from app.context import MessageTooLongError, build_messages, trim_history
from app.llm import ChatMessage
from app.prompts import MODES
from app.telegram import split_message, utf16_length


def dialog(*pairs: tuple[str, str]) -> list[ChatMessage]:
    history = []
    for question, answer in pairs:
        history += [ChatMessage("user", question), ChatMessage("assistant", answer)]
    return history


def test_trim_keeps_newest_pairs_within_message_limit():
    # Arrange
    history = dialog(("в1", "о1"), ("в2", "о2"), ("в3", "о3"))
    # Act
    kept = trim_history(history, max_messages=4, max_chars=10_000)
    # Assert
    assert [m.content for m in kept] == ["в2", "о2", "в3", "о3"]


def test_trim_drops_whole_old_pairs_when_chars_exceeded():
    # Arrange
    history = dialog(("а" * 50, "б" * 50), ("в" * 10, "г" * 10), ("д" * 10, "е" * 10))
    # Act
    kept = trim_history(history, max_messages=100, max_chars=45)
    # Assert
    assert [m.content for m in kept] == ["в" * 10, "г" * 10, "д" * 10, "е" * 10]
    assert kept[0].role == "user"


def test_trim_skips_orphan_message_at_start():
    # Arrange: после обрезки в БД первым может оказаться ответ ассистента без вопроса.
    history = [ChatMessage("assistant", "сирота"), *dialog(("в", "о"))]
    # Act
    kept = trim_history(history, max_messages=3, max_chars=1000)
    # Assert
    assert kept == dialog(("в", "о"))


def test_trim_with_odd_limit_never_breaks_a_pair():
    # Arrange
    history = dialog(("в1", "о1"), ("в2", "о2"))
    # Act
    kept = trim_history(history, max_messages=3, max_chars=1000)
    # Assert
    assert kept == dialog(("в2", "о2"))


def test_build_keeps_system_prompt_and_current_query_when_history_overflows():
    # Arrange
    mode = MODES["study"]
    history = dialog(*[(f"вопрос {i} " + "x" * 3000, f"ответ {i}") for i in range(10)])
    limit = len(mode.system_prompt) + len("Текущий") + 3100
    # Act
    messages = build_messages(mode, history, "Текущий", max_messages=20, max_chars=limit)
    # Assert
    assert messages[0] == ChatMessage("system", mode.system_prompt)
    assert messages[-1] == ChatMessage("user", "Текущий")
    assert [m.content[:9] for m in messages[1:-1]] == ["вопрос 9 ", "ответ 9"]
    assert sum(len(m.content) for m in messages) <= limit


def test_build_counts_few_shot_examples_in_budget():
    # Arrange
    mode = MODES["review"]
    fixed = len(mode.system_prompt) + sum(len(a) + len(b) for a, b in mode.few_shot)
    history = dialog(("q" * 100, "a" * 100))
    # Act
    messages = build_messages(mode, history, "код", max_messages=20, max_chars=fixed + 3 + 199)
    # Assert: пара в 200 символов не помещается в оставшиеся 199.
    assert messages[-1] == ChatMessage("user", "код")
    assert len(messages) == 1 + 2 * len(mode.few_shot) + 1


def test_build_rejects_query_that_does_not_fit_alone():
    # Arrange
    mode = MODES["study"]
    # Act / Assert
    with pytest.raises(MessageTooLongError):
        build_messages(mode, [], "x" * 8000, max_messages=20, max_chars=8000)


@pytest.mark.parametrize("mode", list(MODES.values()), ids=list(MODES))
def test_every_mode_fits_telegram_message_with_minimal_char_limit(mode):
    # Arrange / Act: минимально допустимый HISTORY_MAX_CHARS из конфигурации.
    messages = build_messages(mode, [], "я" * 4096, max_messages=2, max_chars=8000)
    # Assert
    assert messages[-1].content == "я" * 4096


def test_split_short_text_is_single_part():
    assert split_message("Привет") == ["Привет"]


def test_split_prefers_line_breaks_and_loses_nothing():
    # Arrange
    text = "\n".join(f"строка {i:04d} " + "ж" * 80 for i in range(200))
    # Act
    parts = split_message(text)
    # Assert
    assert "".join(parts) == text
    assert all(len(part) <= 4096 for part in parts)
    assert all(part.endswith("\n") for part in parts[:-1])


def test_split_text_without_spaces_uses_hard_limit():
    # Arrange
    text = "a" * 10_000
    # Act
    parts = split_message(text)
    # Assert
    assert [len(part) for part in parts] == [4096, 4096, 1808]


def test_split_counts_emoji_as_two_utf16_units():
    # Arrange
    text = "😀" * 3000  # 6000 единиц UTF-16
    # Act
    parts = split_message(text)
    # Assert
    assert "".join(parts) == text
    assert all(utf16_length(part) <= 4096 for part in parts)
    assert len(parts) == 2


def test_split_never_returns_whitespace_only_parts():
    # Arrange
    text = "a" * 4096 + "\n" * 5000
    # Act
    parts = split_message(text)
    # Assert
    assert parts == ["a" * 4096]
