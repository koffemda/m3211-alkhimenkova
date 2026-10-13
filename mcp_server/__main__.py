"""Запуск MCP-сервера по stdio: python -m mcp_server.

Обычно сервер запускает сам бот как дочерний процесс. Для проверки независимым
клиентом: npx @modelcontextprotocol/inspector python -m mcp_server
(без MCP_TRUST_SECRET инструмент add_reminder всегда отклоняет вызов).
"""

import asyncio
import logging
import os
import sys
from datetime import UTC, datetime

from mcp.server.stdio import stdio_server

from mcp_server.config import ServerConfig, ServerConfigError
from mcp_server.reminders import PostgresReminderStore
from mcp_server.schedule import ScheduleSource
from mcp_server.server import create_server
from mcp_server.weather import OpenMeteoClient


def now() -> datetime:
    return datetime.now(UTC)


async def serve(config: ServerConfig) -> None:
    store = PostgresReminderStore(config.postgres)
    server = create_server(
        weather=OpenMeteoClient(
            config.geocoding_url, config.forecast_url, config.weather_timeout_seconds
        ),
        schedule=ScheduleSource(config.schedule_file, config.schedule_timezone, now),
        reminders=store,
        trust_secret=config.trust_secret,
        clock=now,
    )
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        await store.close()


def main() -> int:
    # stdout занят протоколом MCP, поэтому журнал пишется в stderr.
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = ServerConfig.from_env()
    except ServerConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    asyncio.run(serve(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
