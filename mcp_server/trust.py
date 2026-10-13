"""Доверенный владелец для add_reminder.

Бот генерирует случайный секрет при запуске и передаёт его только своему
дочернему процессу MCP-сервера через переменную окружения MCP_TRUST_SECRET.
Для каждого подтверждённого действия бот подписывает HMAC-SHA256 поля
«владелец, id действия, часовой пояс, хеш аргументов, срок» и кладёт их в
_meta запроса tools/call. Модель этих полей не видит и не формирует.

Сервер принимает вызов, только если подпись верна, срок не истёк и хеш совпадает
с фактическими аргументами. Посторонний MCP-клиент секрета не знает, а сервер,
запущенный без секрета (например, из MCP Inspector), отклоняет любой вызов.
"""

import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

META_KEY = "ru.itmo.assistant/trusted-owner"


class TrustError(Exception):
    """Вызов не содержит действительного доверенного контекста."""


@dataclass(frozen=True)
class TrustedOwner:
    owner_chat_id: int
    action_id: str
    timezone: str


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def arguments_digest(arguments: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical(dict(arguments))).hexdigest()


def _signature(secret: str, payload: Mapping[str, Any]) -> str:
    return hmac.new(secret.encode(), canonical(dict(payload)), hashlib.sha256).hexdigest()


def sign(
    secret: str,
    *,
    owner_chat_id: int,
    action_id: str,
    timezone: str,
    arguments: Mapping[str, Any],
    expires_at: int,
) -> dict[str, Any]:
    """Возвращает значение _meta для tools/call."""
    payload = {
        "owner_chat_id": owner_chat_id,
        "action_id": action_id,
        "timezone": timezone,
        "arguments_sha256": arguments_digest(arguments),
        "expires_at": expires_at,
    }
    return {META_KEY: {**payload, "signature": _signature(secret, payload)}}


def verify(
    secret: str | None,
    meta: Mapping[str, Any] | None,
    arguments: Mapping[str, Any],
    now_ts: float,
) -> TrustedOwner:
    if not secret:
        raise TrustError("сервер запущен без доверенного контекста приложения")
    token = (meta or {}).get(META_KEY)
    if not isinstance(token, Mapping):
        raise TrustError("нет доверенного владельца")
    payload = {key: value for key, value in token.items() if key != "signature"}
    signature = token.get("signature")
    if not isinstance(signature, str) or not hmac.compare_digest(
        signature, _signature(secret, payload)
    ):
        raise TrustError("подпись владельца недействительна")
    if set(payload) != {"owner_chat_id", "action_id", "timezone", "arguments_sha256", "expires_at"}:
        raise TrustError("неполный доверенный контекст")
    if not isinstance(payload["expires_at"], int) or payload["expires_at"] < now_ts:
        raise TrustError("срок подтверждения истёк")
    if payload["arguments_sha256"] != arguments_digest(arguments):
        raise TrustError("аргументы отличаются от подтверждённых")
    owner = payload["owner_chat_id"]
    if not isinstance(owner, int) or isinstance(owner, bool):
        raise TrustError("некорректный владелец")
    return TrustedOwner(owner, str(payload["action_id"]), str(payload["timezone"]))
