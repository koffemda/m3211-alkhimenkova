"""Настройки MCP-сервера из окружения. Модель и пользователь их не задают."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ROOT = Path(__file__).resolve().parents[1]


class ServerConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ServerConfig:
    schedule_file: Path
    schedule_timezone: str
    geocoding_url: str
    forecast_url: str
    weather_timeout_seconds: float
    postgres: dict = field(repr=False)
    trust_secret: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ServerConfig":
        env = os.environ if environ is None else environ
        schedule_file = Path(env.get("SCHEDULE_FILE") or ROOT / "data" / "schedule.json")
        if not schedule_file.is_absolute():
            schedule_file = ROOT / schedule_file
        if not schedule_file.is_file():
            raise ServerConfigError("SCHEDULE_FILE: файл расписания не найден.")
        zone = env.get("SCHEDULE_TIMEZONE") or "Europe/Moscow"
        try:
            ZoneInfo(zone)
        except (ZoneInfoNotFoundError, ValueError):
            raise ServerConfigError(
                "SCHEDULE_TIMEZONE: нужна зона IANA, например Europe/Moscow."
            ) from None
        return cls(
            schedule_file=schedule_file,
            schedule_timezone=zone,
            geocoding_url=env.get("WEATHER_GEOCODING_URL")
            or "https://geocoding-api.open-meteo.com/v1/search",
            forecast_url=env.get("WEATHER_FORECAST_URL")
            or "https://api.open-meteo.com/v1/forecast",
            weather_timeout_seconds=float(env.get("WEATHER_TIMEOUT_SECONDS") or 10),
            postgres={
                "host": env.get("POSTGRES_HOST") or "127.0.0.1",
                "port": int(env.get("POSTGRES_PORT") or 5432),
                "database": env.get("POSTGRES_DB") or "bot",
                "user": env.get("POSTGRES_USER") or "bot",
                "password": env.get("POSTGRES_PASSWORD") or "",
            },
            trust_secret=env.get("MCP_TRUST_SECRET") or None,
        )
