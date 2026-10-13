"""Контракты инструментов и ресурса: единственный источник JSON-схем.

inputSchema/outputSchema, которые получает MCP-клиент, генерируются из этих
моделей. extra="forbid" превращается в additionalProperties: false, поэтому
лишние поля (например, owner от модели) отклоняются сервером.
"""

from datetime import date as date_type
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- get_weather ----------


class WeatherInput(Strict):
    city: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=100),
        Field(
            description=(
                "Название населённого пункта в именительном падеже, например «Казань». "
                "Если пользователь назвал страну или регион, добавьте через запятую: "
                "«Тбилиси, Грузия»."
            )
        ),
    ]


class Location(Strict):
    name: str
    country: str
    admin1: str = Field(description="Регион или область; пустая строка, если неизвестен")
    latitude: float
    longitude: float


class CurrentWeather(Strict):
    temperature_c: float = Field(description="Температура воздуха, °C")
    condition_code: int = Field(description="Код погодных условий WMO")
    condition: str = Field(description="Описание погодных условий")
    wind_speed: float
    wind_speed_unit: str
    observed_at: str = Field(description="Время наблюдения, ISO 8601 со смещением")
    timezone: str = Field(description="Часовой пояс данных (IANA)")


class WeatherOutput(Strict):
    status: Literal["ok", "ambiguous", "not_found"] = Field(
        description="ok — данные получены; ambiguous — нужно выбрать город из candidates; "
        "not_found — город не найден"
    )
    query: str
    location: Location | None = None
    current: CurrentWeather | None = None
    candidates: list[Location] = Field(default_factory=list)
    source: str = "Open-Meteo"


# ---------- get_schedule ----------


class ScheduleInput(Strict):
    date: Annotated[
        str,
        Field(
            pattern=r"^\d{4}-\d{2}-\d{2}$",
            description="Дата в формате ISO 8601 YYYY-MM-DD, например 2026-10-13",
        ),
    ]

    @field_validator("date")
    @classmethod
    def real_date(cls, value: str) -> str:
        date_type.fromisoformat(value)  # 2026-02-30 → ValueError
        return value


class Lesson(Strict):
    start: str = Field(description="Начало, ISO 8601 со смещением")
    end: str = Field(description="Окончание, ISO 8601 со смещением")
    title: str
    location: str


class ScheduleOutput(Strict):
    date: str
    timezone: str
    lessons: list[Lesson] = Field(description="Пустой список означает, что занятий нет")
    source: str = "Учебное расписание (файл сервера)"


class ScheduleDay(Strict):
    date: str
    lessons: list[Lesson]


class WeekOutput(Strict):
    week_start: str
    week_end: str
    timezone: str
    days: list[ScheduleDay]


# ---------- add_reminder ----------


class ReminderInput(Strict):
    text: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
        Field(description="Текст напоминания без слова «напомни», например «отправить отчёт»"),
    ]
    remind_at: Annotated[
        str,
        Field(
            json_schema_extra={"format": "date-time"},
            description=(
                "Момент напоминания: ISO 8601 с явным смещением часового пояса пользователя, "
                "например 2026-10-13T18:30:00+03:00. Только будущее время."
            ),
        ),
    ]

    @field_validator("remind_at")
    @classmethod
    def aware_iso(cls, value: str) -> str:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("нужен явный часовой пояс (смещение), например +03:00")
        return value

    def moment(self) -> datetime:
        return datetime.fromisoformat(self.remind_at)


class ReminderOutput(Strict):
    reminder_id: int
    text: str
    remind_at: str = Field(description="Нормализованное время в часовом поясе пользователя")
    timezone: str
    created: bool = Field(description="false — напоминание уже было создано ранее этим действием")


class ToolError(Strict):
    code: Literal[
        "invalid_arguments",
        "unknown_tool",
        "untrusted_caller",
        "timeout",
        "unavailable",
        "bad_response",
        "conflict",
    ]
    message: str
