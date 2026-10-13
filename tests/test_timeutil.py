"""IANA-зоны и локальное время на переходах DST — без изменения системных часов."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.timeutil import LocalTimeError, normalize_timezone, resolve_local

BERLIN = ZoneInfo("Europe/Berlin")


@pytest.mark.parametrize("name", ["Europe/Moscow", "Asia/Yekaterinburg", "America/New_York", "UTC"])
def test_known_iana_zones_are_accepted(name):
    assert normalize_timezone(f"  {name} ") == name


@pytest.mark.parametrize("name", ["Moscow", "+03:00", "MSK+0", "Europe/Atlantis", "", "../etc"])
def test_unknown_zones_are_rejected(name):
    assert normalize_timezone(name) is None


def test_regular_local_time_gets_zone_offset():
    # Act
    moment = resolve_local(datetime(2026, 10, 13, 18, 30), ZoneInfo("Europe/Moscow"))
    # Assert
    assert moment.isoformat() == "2026-10-13T18:30:00+03:00"


def test_repeated_hour_in_autumn_is_ambiguous():
    with pytest.raises(LocalTimeError) as error:
        resolve_local(datetime(2026, 10, 25, 2, 30), BERLIN)
    assert error.value.kind == "ambiguous"


def test_skipped_hour_in_spring_does_not_exist():
    with pytest.raises(LocalTimeError) as error:
        resolve_local(datetime(2027, 3, 28, 2, 30), BERLIN)
    assert error.value.kind == "nonexistent"
