import argparse
import asyncio
import logging
import os
import secrets
import sys
from datetime import UTC, datetime
from pathlib import Path

from aiogram import Dispatcher
from mcp.client.stdio import StdioServerParameters

from app.agent import Agent
from app.assistant import AssistantService
from app.config import ConfigError, Settings
from app.confirmations import ReminderConfirmations
from app.db import create_pool
from app.handlers.assistant import create_router
from app.health import HealthState, start_health_server
from app.llm import OpenAICompatibleClient
from app.logging_setup import configure_logging
from app.mcp_gateway import McpGateway
from app.storage import PostgresRepository, apply_migrations
from app.telegram import create_bot

logger = logging.getLogger("app")
ROOT = Path(__file__).resolve().parents[1]
# Внешние адреса погодного сервиса передаются серверу, только если заданы явно.
PASSTHROUGH = ("WEATHER_GEOCODING_URL", "WEATHER_FORECAST_URL", "WEATHER_TIMEOUT_SECONDS")


def now() -> datetime:
    return datetime.now(UTC)


def mcp_server_parameters(settings: Settings, trust_secret: str) -> StdioServerParameters:
    """MCP-сервер запускается дочерним процессом; секрет доверия знает только этот процесс."""
    env = {
        "POSTGRES_HOST": settings.postgres_host,
        "POSTGRES_PORT": str(settings.postgres_port),
        "POSTGRES_DB": settings.postgres_db,
        "POSTGRES_USER": settings.postgres_user,
        "POSTGRES_PASSWORD": settings.postgres_password,
        "SCHEDULE_FILE": settings.schedule_file,
        "SCHEDULE_TIMEZONE": settings.schedule_timezone,
        "LOG_LEVEL": settings.log_level,
        "MCP_TRUST_SECRET": trust_secret,
        "PYTHONUTF8": "1",
        **{key: os.environ[key] for key in PASSTHROUGH if os.environ.get(key)},
    }
    return StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_server"], env=env, cwd=str(ROOT)
    )


async def run(settings: Settings) -> None:
    state = HealthState()
    bot = create_bot(settings)
    llm = OpenAICompatibleClient(settings)
    trust_secret = secrets.token_hex(32)
    gateway = McpGateway(
        mcp_server_parameters(settings, trust_secret),
        call_timeout=settings.mcp_call_timeout_seconds,
    )
    runner = None
    try:
        state.pool = await create_pool(settings)
        logger.info("PostgreSQL подключён: SELECT 1 выполнен.")
        applied = await apply_migrations(state.pool)
        logger.info("Схема БД актуальна; применены миграции: %s.", applied or "нет новых")
        repository = PostgresRepository(state.pool)
        # Таблицы reminders и pending_actions уже созданы миграциями — можно запускать сервер.
        await gateway.start()
        assistant = AssistantService(
            repository,
            llm,
            max_messages=settings.history_max_messages,
            max_chars=settings.history_max_chars,
            agent=Agent(
                llm,
                gateway,
                repository,
                max_messages=settings.history_max_messages,
                max_chars=settings.history_max_chars,
                schedule_timezone=settings.schedule_timezone,
                clock=now,
            ),
            confirmations=ReminderConfirmations(
                repository, gateway, trust_secret=trust_secret, clock=now
            ),
            gateway=gateway,
            clock=now,
        )
        logger.info("LLM: модель %s.", settings.openrouter_model)
        # Начальная проверка токена и маршрута через прокси ограничена по времени.
        async with asyncio.timeout(30):
            me = await bot.get_me()
            webhook = await bot.get_webhook_info()
        if webhook.url:
            raise ConfigError(
                "У бота установлен webhook. Удалите его перед запуском polling "
                "или используйте отдельного учебного бота."
            )
        logger.info("Telegram доступен. Бот @%s запускает polling.", me.username)
        dispatcher = Dispatcher()
        dispatcher.include_router(create_router())
        runner = await start_health_server(state, settings.health_port)
        state.polling_task = asyncio.create_task(
            dispatcher.start_polling(
                bot,
                db=state.pool,
                assistant=assistant,
                allowed_updates=dispatcher.resolve_used_update_types(),
                close_bot_session=False,
            )
        )
        state.initialized = True
        await state.polling_task
    finally:
        state.initialized = False
        if state.polling_task and not state.polling_task.done():
            state.polling_task.cancel()
            await asyncio.gather(state.polling_task, return_exceptions=True)
        if runner:
            await runner.cleanup()
        await bot.session.close()
        await gateway.stop()
        await llm.close()
        if state.pool:
            try:
                async with asyncio.timeout(10):
                    await state.pool.close()
            except TimeoutError:
                state.pool.terminate()


def main() -> int:
    parser = argparse.ArgumentParser(description="AI-ассистент студента в Telegram")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()
    try:
        settings = Settings.load(args.env_file)
    except ConfigError as exc:
        print(str(exc))
        return 1
    configure_logging(settings)
    try:
        asyncio.run(run(settings))
    except KeyboardInterrupt:
        logger.info("Бот остановлен.")
    except Exception:
        logger.exception("Не удалось запустить бот. Проверьте БД, токен и прокси.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
