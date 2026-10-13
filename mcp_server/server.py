"""MCP-сервер: три инструмента и ресурс schedule://current-week.

Используется низкоуровневый Server из MCP Python SDK: схемы задаются явно из
моделей schemas.py, а каждый вызов повторно валидируется на сервере.
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, ValidationError

from mcp_server import trust
from mcp_server.reminders import ReminderConflictError, ReminderStore
from mcp_server.schedule import ScheduleSource
from mcp_server.schemas import (
    ReminderInput,
    ReminderOutput,
    ScheduleInput,
    ScheduleOutput,
    ToolError,
    WeatherInput,
    WeatherOutput,
    WeekOutput,
)
from mcp_server.weather import OpenMeteoClient, WeatherError

logger = logging.getLogger("mcp_server")

WEEK_URI = "schedule://current-week"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    title: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    read_only: bool


TOOLS = [
    ToolSpec(
        "get_weather",
        "текущая погода в городе",
        "Текущая погода в населённом пункте: температура, условия, ветер, время наблюдения. "
        "Вызывайте для любых вопросов о погоде сейчас или сегодня. Если вернулся статус "
        "ambiguous — пользователь должен выбрать город из candidates.",
        WeatherInput,
        WeatherOutput,
        read_only=True,
    ),
    ToolSpec(
        "get_schedule",
        "занятия на выбранную дату",
        "Учебные занятия пользователя на одну дату: начало, конец, название, аудитория. "
        "Пустой список означает, что занятий нет. Относительные даты («сегодня», «завтра», "
        "«в воскресенье») переводите в YYYY-MM-DD по текущей дате пользователя.",
        ScheduleInput,
        ScheduleOutput,
        read_only=True,
    ),
    ToolSpec(
        "add_reminder",
        "создание напоминания",
        "Создаёт напоминание пользователю на конкретный момент в будущем. Нужны текст и точные "
        "дата и время. Если время неточное («вечером», «потом»), не вызывайте инструмент, "
        "а уточните у пользователя. Выполняется только после подтверждения пользователем.",
        ReminderInput,
        ReminderOutput,
        read_only=False,
    ),
]
SPECS = {spec.name: spec for spec in TOOLS}


def _error(code: str, message: str) -> types.CallToolResult:
    payload = {"error": ToolError(code=code, message=message).model_dump()}
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
        structured_content=payload,
        is_error=True,
    )


def _success(result: BaseModel) -> types.CallToolResult:
    data = result.model_dump(mode="json")
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(data, ensure_ascii=False))],
        structured_content=data,
    )


def _validation_message(error: ValidationError) -> str:
    # Значения аргументов в сообщение не попадают — только поле и причина.
    parts = []
    for item in error.errors():
        field = ".".join(str(x) for x in item["loc"]) or "arguments"
        reason = "лишнее поле не допускается" if item["type"] == "extra_forbidden" else item["msg"]
        parts.append(f"{field}: {reason}")
    return "; ".join(parts)


def create_server(
    *,
    weather: OpenMeteoClient,
    schedule: ScheduleSource,
    reminders: ReminderStore,
    trust_secret: str | None,
    clock: Callable[[], datetime],
) -> Server:
    async def list_tools(_ctx, _params) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=spec.name,
                    title=spec.title,
                    description=spec.description,
                    input_schema=spec.input_model.model_json_schema(),
                    output_schema=spec.output_model.model_json_schema(),
                    annotations=types.ToolAnnotations(
                        title=spec.title,
                        read_only_hint=spec.read_only,
                        destructive_hint=False,
                        idempotent_hint=True,
                        open_world_hint=spec.name == "get_weather",
                    ),
                )
                for spec in TOOLS
            ]
        )

    async def run_tool(name: str, arguments: dict[str, Any], meta) -> BaseModel:
        if name == "get_weather":
            return await weather.current(WeatherInput.model_validate(arguments).city)
        if name == "get_schedule":
            day = date.fromisoformat(ScheduleInput.model_validate(arguments).date)
            return schedule.on_date(day)
        # add_reminder: сначала доверенный владелец, потом аргументы.
        owner = trust.verify(trust_secret, meta, arguments, clock().timestamp())
        data = ReminderInput.model_validate(arguments)
        moment = data.moment()
        if moment <= clock():
            raise ValueError("remind_at: время напоминания должно быть в будущем")
        zone = ZoneInfo(owner.timezone)
        stored = await reminders.add(
            owner_chat_id=owner.owner_chat_id,
            key=owner.action_id,
            text=data.text,
            remind_at=moment,
            timezone=owner.timezone,
        )
        return ReminderOutput(
            reminder_id=stored.reminder_id,
            text=stored.text,
            remind_at=stored.remind_at.astimezone(zone).isoformat(),
            timezone=owner.timezone,
            created=stored.created,
        )

    async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        started = time.monotonic()
        name = params.name
        if name not in SPECS:
            return _error("unknown_tool", f"инструмент {name} не существует")
        try:
            result = await run_tool(name, dict(params.arguments or {}), ctx.meta)
            status = "ok"
            return _success(result)
        except trust.TrustError as exc:
            status = "untrusted"
            return _error("untrusted_caller", f"вызов отклонён: {exc}")
        except ValidationError as exc:
            status = "invalid"
            return _error("invalid_arguments", _validation_message(exc))
        except ValueError as exc:
            status = "invalid"
            return _error("invalid_arguments", str(exc))
        except WeatherError as exc:
            status = exc.code
            return _error(exc.code, str(exc))
        except ReminderConflictError:
            status = "conflict"
            return _error("conflict", "ключ действия уже использован")
        except Exception:
            status = "error"
            logger.exception("Инструмент %s завершился ошибкой", name)
            return _error("unavailable", "внутренняя ошибка инструмента")
        finally:
            # Аргументы и владельца не логируем: только имя, статус и длительность.
            logger.info(
                "tool=%s status=%s duration=%.2fs", name, status, time.monotonic() - started
            )

    async def list_resources(_ctx, _params) -> types.ListResourcesResult:
        return types.ListResourcesResult(
            resources=[
                types.Resource(
                    name="current-week",
                    title="Расписание на текущую неделю",
                    uri=WEEK_URI,
                    description="Занятия с понедельника по воскресенье текущей недели, "
                    "сгруппированные по датам. Только чтение.",
                    mime_type="application/json",
                )
            ]
        )

    async def read_resource(_ctx, params: types.ReadResourceRequestParams):
        if str(params.uri) != WEEK_URI:
            raise MCPError(code=types.INVALID_PARAMS, message="Ресурс не найден")
        week: WeekOutput = schedule.current_week()
        return types.ReadResourceResult(
            contents=[
                types.TextResourceContents(
                    uri=WEEK_URI,
                    mime_type="application/json",
                    text=week.model_dump_json(),
                )
            ]
        )

    return Server(
        "student-assistant-tools",
        version="2.0.0",
        instructions="Погода, учебное расписание и напоминания для AI-ассистента студента.",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        on_list_resources=list_resources,
        on_read_resource=read_resource,
    )


__all__ = ["SPECS", "TOOLS", "WEEK_URI", "create_server"]
