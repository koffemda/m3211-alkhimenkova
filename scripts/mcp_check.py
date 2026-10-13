"""Независимая проверка MCP-сервера отдельным клиентом (не ботом).

    .venv/bin/python -m scripts.mcp_check            # macOS / Linux
    .\\.venv\\Scripts\\python.exe -m scripts.mcp_check  # Windows

Скрипт запускает `python -m mcp_server` по stdio БЕЗ секрета доверия, как это
сделал бы MCP Inspector, обнаруживает инструменты и ресурс, вызывает чтение и
убеждается, что add_reminder от постороннего клиента отклоняется.
Погода запрашивается у настоящего Open-Meteo; при отсутствии сети будет ошибка
timeout/unavailable — это тоже ожидаемое безопасное поведение.
"""

import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters

ROOT = Path(__file__).resolve().parents[1]


def show(title: str, value) -> None:
    print(f"\n== {title}")
    print(json.dumps(value, ensure_ascii=False, indent=2)[:1500])


async def main() -> int:
    server = StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_server"], cwd=str(ROOT), env={"PYTHONUTF8": "1"}
    )
    async with Client(server) as client:
        print(f"Протокол: {client.protocol_version}; сервер: {client.server_info}")
        tools = (await client.list_tools()).tools
        show("Инструменты", [{"name": t.name, "input": t.input_schema} for t in tools])
        resources = (await client.list_resources()).resources
        show("Ресурсы", [str(r.uri) for r in resources])
        week = await client.read_resource("schedule://current-week")
        show("schedule://current-week", json.loads(week.contents[0].text))
        today = datetime.now().date().isoformat()
        result = await client.call_tool("get_schedule", {"date": today})
        show(f"get_schedule {today}", result.structured_content)
        result = await client.call_tool("get_schedule", {"date": "13.10.2026"})
        show(
            "get_schedule с неверной датой (ожидается invalid_arguments)", result.structured_content
        )
        result = await client.call_tool("get_weather", {"city": "Санкт-Петербург"})
        show("get_weather Санкт-Петербург", result.structured_content)
        remind_at = (datetime.now(UTC) + timedelta(days=1)).isoformat(timespec="seconds")
        result = await client.call_tool(
            "add_reminder", {"text": "проверка", "remind_at": remind_at}
        )
        show("add_reminder без доверенного контекста (ожидается отказ)", result.structured_content)
        rejected = (
            result.is_error and result.structured_content["error"]["code"] == "untrusted_caller"
        )
    print("\nИтог:", "посторонний клиент не может создать напоминание" if rejected else "ОШИБКА")
    return 0 if rejected else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
