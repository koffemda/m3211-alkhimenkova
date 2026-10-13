"""Часовые пояса IANA и интерпретация локального времени.

Работает одинаково в Linux, macOS и Windows: база зон берётся из пакета tzdata,
если в системе её нет (Windows).
"""

from datetime import datetime, timedelta
from functools import cache
from zoneinfo import ZoneInfo, available_timezones

WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]


class LocalTimeError(ValueError):
    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind  # ambiguous | nonexistent


@cache
def _known_zones() -> frozenset[str]:
    return frozenset(available_timezones())


def normalize_timezone(name: str) -> str | None:
    """Возвращает имя зоны IANA или None. Смещения вида +03:00 не принимаются."""
    name = name.strip()
    return name if name in _known_zones() else None


def resolve_local(wall: datetime, zone: ZoneInfo) -> datetime:
    """Привязывает локальное время без смещения к зоне.

    В день перехода на летнее/зимнее время одно и то же локальное время может
    встретиться дважды (ambiguous) или не существовать (nonexistent); тогда
    время нужно уточнить у пользователя, а не выбирать вариант молча.
    """
    first = wall.replace(tzinfo=zone, fold=0)
    second = wall.replace(tzinfo=zone, fold=1)
    if first.utcoffset() != second.utcoffset():
        # Для несуществующего времени перевод в UTC и обратно меняет «часы на стене».
        roundtrip = first.astimezone(ZoneInfo("UTC")).astimezone(zone).replace(tzinfo=None)
        raise LocalTimeError("nonexistent" if roundtrip != wall else "ambiguous")
    return first


def format_local(moment: datetime, zone_name: str) -> str:
    return moment.astimezone(ZoneInfo(zone_name)).strftime("%d.%m.%Y %H:%M")


def describe_now(now: datetime, zone_name: str) -> str:
    local = now.astimezone(ZoneInfo(zone_name))
    return f"{local.isoformat(timespec='seconds')} ({zone_name}), {WEEKDAYS[local.weekday()]}"


CONFIRMATION_TTL = timedelta(minutes=5)
