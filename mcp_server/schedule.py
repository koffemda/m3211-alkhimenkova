"""Расписание из файла, путь к которому задаёт только конфигурация сервера."""

import json
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from mcp_server.schemas import Lesson, ScheduleDay, ScheduleOutput, WeekOutput


class _FileLesson(BaseModel):
    model_config = ConfigDict(extra="ignore")
    date: date
    start: time
    end: time
    title: str
    location: str


class _ScheduleFile(BaseModel):
    model_config = ConfigDict(extra="ignore")
    lessons: list[_FileLesson]


class ScheduleSource:
    def __init__(self, path: Path, timezone: str, clock: Callable[[], datetime]):
        self.path = path
        self.zone = ZoneInfo(timezone)
        self.timezone = timezone
        self.clock = clock

    def _load(self) -> list[_FileLesson]:
        # Файл читается при каждом запросе: правки расписания видны без перезапуска.
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return _ScheduleFile.model_validate(data).lessons

    def _lesson(self, item: _FileLesson) -> Lesson:
        def moment(at: time) -> str:
            return datetime.combine(item.date, at, tzinfo=self.zone).isoformat()

        return Lesson(
            start=moment(item.start), end=moment(item.end), title=item.title, location=item.location
        )

    def on_date(self, day: date) -> ScheduleOutput:
        lessons = sorted((x for x in self._load() if x.date == day), key=lambda x: x.start)
        return ScheduleOutput(
            date=day.isoformat(),
            timezone=self.timezone,
            lessons=[self._lesson(x) for x in lessons],
        )

    def current_week(self) -> WeekOutput:
        today = self.clock().astimezone(self.zone).date()
        monday = today - timedelta(days=today.weekday())
        lessons = self._load()
        days = []
        for offset in range(7):
            day = monday + timedelta(days=offset)
            items = sorted((x for x in lessons if x.date == day), key=lambda x: x.start)
            days.append(ScheduleDay(date=day.isoformat(), lessons=[self._lesson(x) for x in items]))
        return WeekOutput(
            week_start=monday.isoformat(),
            week_end=(monday + timedelta(days=6)).isoformat(),
            timezone=self.timezone,
            days=days,
        )
