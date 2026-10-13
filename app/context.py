"""Сборка запроса к модели и сокращение истории.

Стратегия «скользящее окно по парам»: системная инструкция, few-shot-примеры и
текущее сообщение передаются всегда; из истории берутся самые новые пары
«вопрос — ответ», пока не превышены оба ограничения: число сообщений истории и
суммарный объём запроса в символах (оценка объёма без токенизатора конкретной
модели). Старые пары отбрасываются целиком, поэтому история остаётся связной и
всегда начинается с сообщения пользователя.
"""

from collections.abc import Sequence

from app.llm import ChatMessage
from app.prompts import Mode


class MessageTooLongError(ValueError):
    """Текущий запрос не помещается в лимит даже без истории."""


def fixed_part(mode: Mode, system_prompt: str | None = None) -> list[ChatMessage]:
    messages = [ChatMessage("system", system_prompt or mode.system_prompt)]
    for question, answer in mode.few_shot:
        messages += [ChatMessage("user", question), ChatMessage("assistant", answer)]
    return messages


def size(messages: Sequence[ChatMessage]) -> int:
    return sum(len(message.content) for message in messages)


def trim_history(
    history: Sequence[ChatMessage], *, max_messages: int, max_chars: int
) -> list[ChatMessage]:
    """Возвращает самый новый связный хвост истории в пределах ограничений."""
    kept: list[ChatMessage] = []
    used = 0
    index = len(history)
    while index > 0:
        # Сообщения берутся с конца парами; одиночное сообщение без пары не берётся.
        start = index - 2
        if start < 0:
            break
        pair = list(history[start:index])
        if pair[0].role != "user" or pair[1].role != "assistant":
            break
        pair_size = size(pair)
        if len(kept) + 2 > max_messages or used + pair_size > max_chars:
            break
        kept = pair + kept
        used += pair_size
        index = start
    return kept


def build_messages(
    mode: Mode,
    history: Sequence[ChatMessage],
    user_text: str,
    *,
    max_messages: int,
    max_chars: int,
    system_prompt: str | None = None,
) -> list[ChatMessage]:
    """system_prompt заменяет инструкцию режима (агент добавляет к ней правила инструментов)."""
    head = fixed_part(mode, system_prompt)
    current = ChatMessage("user", user_text)
    budget = max_chars - size(head) - len(user_text)
    if budget < 0:
        raise MessageTooLongError
    tail = trim_history(history, max_messages=max_messages, max_chars=budget)
    return [*head, *tail, current]
