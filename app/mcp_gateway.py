"""MCP-клиент приложения: подключение к серверу, обнаружение инструментов, вызовы.

Соединение принадлежит отдельной задаче-супервизору: она подключается, получает
список инструментов и держит сессию; после разрыва переподключается с паузой.
Бот при недоступном сервере продолжает работать без инструментов.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator
from mcp import Client

logger = logging.getLogger("app.mcp")

RECONNECT_DELAYS = (2, 5, 15, 30, 60)


class ToolCallError(Exception):
    """Единое представление ошибок MCP и внешних сервисов для агента."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ToolInfo:
    name: str
    title: str
    description: str
    input_schema: dict
    read_only: bool

    def openai_tool(self) -> dict:
        """Схема для нативного tool calling LLM API — берётся из обнаруженной inputSchema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


def schema_errors(schema: dict, arguments: Any) -> list[str]:
    """Проверка аргументов по inputSchema сервера; в сообщениях нет самих значений."""
    errors = []
    for error in Draft202012Validator(schema).iter_errors(arguments):
        field = ".".join(str(part) for part in error.absolute_path) or "arguments"
        if error.validator == "additionalProperties":
            extra = sorted(set(arguments) - set(schema.get("properties", {})))
            errors.append(f"лишние поля: {', '.join(extra)}")
        else:
            errors.append(f"{field}: нарушено правило {error.validator}")
    return errors


class McpGateway:
    def __init__(self, target: Any, *, call_timeout: float = 20, connect_timeout: float = 15):
        # target: StdioServerParameters (рабочий режим) или Server (тесты, in-process).
        self._target = target
        self._call_timeout = call_timeout
        self._connect_timeout = connect_timeout
        self._client: Client | None = None
        self._tools: dict[str, ToolInfo] = {}
        self._broken = asyncio.Event()
        self._ready = asyncio.Event()
        self._stopping = False
        self._task: asyncio.Task | None = None

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def tools(self) -> list[ToolInfo]:
        return list(self._tools.values()) if self.available else []

    def tool(self, name: str) -> ToolInfo | None:
        return self._tools.get(name) if self.available else None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._supervise(), name="mcp-supervisor")
        try:
            await asyncio.wait_for(self._ready.wait(), self._connect_timeout)
        except TimeoutError:
            logger.warning("MCP-сервер не подключился за %s с; бот работает без инструментов.",
                           self._connect_timeout)  # fmt: skip

    async def stop(self) -> None:
        self._stopping = True
        self._broken.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, 10)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()

    async def _supervise(self) -> None:
        attempt = 0
        while not self._stopping:
            try:
                async with Client(self._target) as client:
                    listing = await client.list_tools()
                    capabilities = client.server_capabilities
                    if capabilities.tools is None:
                        raise ToolCallError("unavailable", "сервер не объявил возможность tools")
                    self._tools = {
                        tool.name: ToolInfo(
                            name=tool.name,
                            title=tool.title or tool.name,
                            description=tool.description or "",
                            input_schema=dict(tool.input_schema),
                            read_only=bool(tool.annotations and tool.annotations.read_only_hint),
                        )
                        for tool in listing.tools
                    }
                    self._client = client
                    attempt = 0
                    self._broken.clear()
                    self._ready.set()
                    logger.info(
                        "MCP подключён: протокол %s, инструменты: %s",
                        client.protocol_version,
                        ", ".join(self._tools) or "нет",
                    )
                    await self._broken.wait()
            except Exception as exc:
                # Причину сбоя пишем без аргументов и секретов: только тип и краткий текст.
                logger.warning("MCP-сервер недоступен: %s: %s", type(exc).__name__, exc)
            finally:
                self._client = None
            if self._stopping:
                break
            delay = RECONNECT_DELAYS[min(attempt, len(RECONNECT_DELAYS) - 1)]
            attempt += 1
            logger.info("Повторное подключение к MCP через %s с.", delay)
            await asyncio.sleep(delay)

    def validate(self, name: str, arguments: Any) -> list[str]:
        tool = self.tool(name)
        if tool is None:
            return [f"инструмент {name} не обнаружен у MCP-сервера"]
        if not isinstance(arguments, dict):
            return ["аргументы должны быть JSON-объектом"]
        return schema_errors(tool.input_schema, arguments)

    async def call(self, name: str, arguments: dict, *, meta: dict | None = None) -> dict[str, Any]:
        client = self._client
        if client is None:
            raise ToolCallError("unavailable", "MCP-сервер не подключён")
        if self.tool(name) is None:
            raise ToolCallError("unknown_tool", name)
        problems = self.validate(name, arguments)
        if problems:
            raise ToolCallError("invalid_arguments", "; ".join(problems))
        started = time.monotonic()
        try:
            async with asyncio.timeout(self._call_timeout):
                result = await client.call_tool(name, arguments, meta=meta)
        except TimeoutError:
            raise ToolCallError("timeout", "MCP-сервер не ответил вовремя") from None
        except RuntimeError as exc:
            # Клиент SDK сверяет structuredContent с outputSchema инструмента.
            logger.warning("MCP %s: результат не прошёл outputSchema: %s", name, exc)
            raise ToolCallError("bad_result", "результат не соответствует схеме") from None
        except Exception as exc:
            logger.warning("MCP %s: разрыв соединения: %s", name, type(exc).__name__)
            self._broken.set()
            raise ToolCallError("unavailable", "соединение с MCP-сервером потеряно") from None
        finally:
            logger.info("MCP %s: %.2f с", name, time.monotonic() - started)
        data = result.structured_content
        if result.is_error:
            error = data.get("error") if isinstance(data, dict) else None
            if isinstance(error, dict) and isinstance(error.get("code"), str):
                raise ToolCallError(error["code"], str(error.get("message", "")))
            raise ToolCallError("bad_result", "ошибка без структурированного описания")
        if not isinstance(data, dict):
            raise ToolCallError("bad_result", "нет structuredContent")
        return data

    async def read_resource(self, uri: str) -> Any:
        client = self._client
        if client is None:
            raise ToolCallError("unavailable", "MCP-сервер не подключён")
        try:
            async with asyncio.timeout(self._call_timeout):
                result = await client.read_resource(uri)
            return json.loads(result.contents[0].text)
        except TimeoutError:
            raise ToolCallError("timeout", "MCP-сервер не ответил вовремя") from None
        except (ValueError, AttributeError, IndexError):
            raise ToolCallError("bad_result", "ресурс вернул некорректные данные") from None
