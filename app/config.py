"""Единственное место чтения и проверки настроек приложения."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aiogram.utils.token import TokenValidationError, validate_token
from dotenv import dotenv_values


class ConfigError(ValueError):
    """Ошибка настройки без секретных значений в сообщении."""


@dataclass(frozen=True)
class Settings:
    bot_token: str = field(repr=False)
    postgres_password: str = field(repr=False)
    telegram_proxy_url: str = field(default="", repr=False)
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_db: str = "bot"
    postgres_user: str = "bot"
    log_level: str = "INFO"
    health_port: int = 8080
    # Имена полей совпадают с переменными окружения: деплой передаёт их через asdict().
    openrouter_api_key: str = field(default="", repr=False)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-4o-mini"
    llm_proxy_url: str = field(default="", repr=False)
    llm_timeout_seconds: int = 60
    llm_max_tokens: int = 2000
    history_max_messages: int = 20
    history_max_chars: int = 12000
    # ЛР2: источник расписания MCP-сервера и таймаут MCP-вызова.
    schedule_file: str = "data/schedule.json"
    schedule_timezone: str = "Europe/Moscow"
    mcp_call_timeout_seconds: int = 20

    @classmethod
    def load(
        cls, env_file: Path | str = ".env", *, environ: Mapping[str, str] | None = None
    ) -> "Settings":
        values = {
            **dotenv_values(env_file, interpolate=False),
            **(os.environ if environ is None else environ),
        }

        def value(key: str, default: str = "") -> str:
            return values.get(key) or default

        def port(key: str, default: str) -> int:
            try:
                result = int(value(key, default))
                if not 1 <= result <= 65535:
                    raise ValueError
                return result
            except ValueError:
                raise ConfigError(f"{key}: нужен номер порта от 1 до 65535.") from None

        token = value("BOT_TOKEN")
        try:
            validate_token(token)
        except TokenValidationError:
            raise ConfigError("BOT_TOKEN: укажите токен, полученный у BotFather.") from None
        password = value("POSTGRES_PASSWORD")
        if not password:
            raise ConfigError("POSTGRES_PASSWORD: пароль базы данных не задан.")

        def number(key: str, default: str, low: int, high: int) -> int:
            try:
                result = int(value(key, default))
                if not low <= result <= high:
                    raise ValueError
                return result
            except ValueError:
                raise ConfigError(f"{key}: нужно целое число от {low} до {high}.") from None

        def proxy_url(key: str) -> str:
            proxy = value(key)
            if proxy:
                try:
                    parsed = urlsplit(proxy)
                    if (
                        parsed.scheme not in {"http", "socks5"}
                        or not parsed.hostname
                        or not parsed.port
                        or parsed.path not in {"", "/"}
                        or parsed.query
                        or parsed.fragment
                    ):
                        raise ValueError
                except ValueError:
                    raise ConfigError(
                        f"{key}: нужен http://host:port или socks5://host:port; "
                        "при необходимости добавьте user:password@."
                    ) from None
            return proxy

        proxy = proxy_url("TELEGRAM_PROXY_URL")
        level = value("LOG_LEVEL", "INFO").upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError("LOG_LEVEL: используйте DEBUG, INFO, WARNING, ERROR или CRITICAL.")
        # Ключ необязателен для запуска: без него бот работает, а на запросы к модели
        # отвечает, что LLM не настроена.
        api_key = value("OPENROUTER_API_KEY").strip()
        base_url = value("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        parsed_base = urlsplit(base_url)
        if (
            parsed_base.scheme not in {"http", "https"}
            or not parsed_base.hostname
            or parsed_base.username
            or parsed_base.password
            or parsed_base.query
        ):
            # Ключ хранится отдельно, поэтому учётные данные в адресе API не допускаются.
            raise ConfigError(
                "OPENROUTER_BASE_URL: нужен адрес вида https://host/api/v1 без логина и ключа."
            )
        model = value("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
        if not model:
            raise ConfigError("OPENROUTER_MODEL: укажите имя модели.")
        schedule_file = value("SCHEDULE_FILE", "data/schedule.json").strip()
        schedule_path = Path(schedule_file)
        if not schedule_path.is_absolute():
            schedule_path = Path(__file__).resolve().parents[1] / schedule_path
        if not schedule_path.is_file():
            raise ConfigError("SCHEDULE_FILE: файл расписания не найден.")
        schedule_timezone = value("SCHEDULE_TIMEZONE", "Europe/Moscow").strip()
        try:
            ZoneInfo(schedule_timezone)
        except (ZoneInfoNotFoundError, ValueError):
            raise ConfigError(
                "SCHEDULE_TIMEZONE: нужна зона IANA, например Europe/Moscow."
            ) from None
        return cls(
            bot_token=token,
            postgres_password=password,
            telegram_proxy_url=proxy,
            postgres_host=value("POSTGRES_HOST", "127.0.0.1"),
            postgres_port=port("POSTGRES_PORT", "5432"),
            postgres_db=value("POSTGRES_DB", "bot"),
            postgres_user=value("POSTGRES_USER", "bot"),
            log_level=level,
            health_port=port("HEALTH_PORT", "8080"),
            openrouter_api_key=api_key,
            openrouter_base_url=base_url,
            openrouter_model=model,
            llm_proxy_url=proxy_url("LLM_PROXY_URL"),
            llm_timeout_seconds=number("LLM_TIMEOUT_SECONDS", "60", 5, 600),
            llm_max_tokens=number("LLM_MAX_TOKENS", "2000", 64, 32000),
            history_max_messages=number("HISTORY_MAX_MESSAGES", "20", 2, 200),
            # Нижняя граница гарантирует место для промпта режима и сообщения Telegram.
            history_max_chars=number("HISTORY_MAX_CHARS", "12000", 8000, 200000),
            schedule_file=schedule_file,
            schedule_timezone=schedule_timezone,
            mcp_call_timeout_seconds=number("MCP_CALL_TIMEOUT_SECONDS", "20", 2, 120),
        )
