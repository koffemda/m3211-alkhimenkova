"""Текущая погода: Open-Meteo Geocoding API + Forecast API.

Open-Meteo выбран потому, что он бесплатен, не требует ключа, отдаёт названия
на русском (language=ru), а геокодер возвращает страну, регион и население.
Это нужно, чтобы отличать одноимённые города.
Адреса API задаёт конфигурация сервера; модель передаёт только название города.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import aiohttp
from pydantic import BaseModel, ConfigDict, ValidationError

from mcp_server.schemas import CurrentWeather, Location, WeatherOutput

logger = logging.getLogger("mcp_server.weather")

MAX_CANDIDATES = 5
# Город считается однозначным, если он не меньше 100 тыс. жителей и в 10 раз
# крупнее следующего одноимённого (Москва против деревни Москва в Тверской области).
DOMINANT_POPULATION = 100_000
DOMINANCE_RATIO = 10

WMO_CODES = {
    0: "ясно",
    1: "преимущественно ясно",
    2: "переменная облачность",
    3: "пасмурно",
    45: "туман",
    48: "туман с изморозью",
    51: "слабая морось",
    53: "морось",
    55: "сильная морось",
    56: "слабая ледяная морось",
    57: "ледяная морось",
    61: "небольшой дождь",
    63: "дождь",
    65: "сильный дождь",
    66: "ледяной дождь",
    67: "сильный ледяной дождь",
    71: "небольшой снег",
    73: "снег",
    75: "сильный снег",
    77: "снежные зёрна",
    80: "небольшой ливень",
    81: "ливень",
    82: "сильный ливень",
    85: "снегопад",
    86: "сильный снегопад",
    95: "гроза",
    96: "гроза с градом",
    99: "сильная гроза с градом",
}


class WeatherError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code  # timeout | unavailable | bad_response


class _Place(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    latitude: float
    longitude: float
    country: str = ""
    admin1: str = ""
    population: int | None = None


class _Geocoding(BaseModel):
    model_config = ConfigDict(extra="ignore")
    results: list[_Place] = []


class _Current(BaseModel):
    model_config = ConfigDict(extra="ignore")
    time: str
    temperature_2m: float
    weather_code: int
    wind_speed_10m: float


class _Forecast(BaseModel):
    model_config = ConfigDict(extra="ignore")
    timezone: str
    utc_offset_seconds: int
    current: _Current
    current_units: dict[str, str]


def normalize(text: str) -> str:
    return text.casefold().replace("ё", "е").strip()


def choose(query: str, places: list[_Place]) -> tuple[str, list[_Place]]:
    """Возвращает ("ok", [место]) | ("ambiguous", варианты) | ("not_found", [])."""
    name, _, qualifier = query.partition(",")
    if qualifier.strip():
        wanted = normalize(qualifier)
        places = [
            p for p in places if wanted in normalize(p.country) or wanted in normalize(p.admin1)
        ]
    exact = [p for p in places if normalize(p.name) == normalize(name)]
    unique: dict[tuple[str, str, str], _Place] = {}
    for place in exact or places:
        unique.setdefault((place.name, place.country, place.admin1), place)
    candidates = sorted(unique.values(), key=lambda p: p.population or 0, reverse=True)
    if not candidates:
        return "not_found", []
    if len(candidates) == 1:
        return "ok", candidates
    top, second = candidates[0].population or 0, candidates[1].population or 0
    if top >= DOMINANT_POPULATION and top >= DOMINANCE_RATIO * second:
        return "ok", candidates[:1]
    return "ambiguous", candidates[:MAX_CANDIDATES]


def location(place: _Place) -> Location:
    return Location(
        name=place.name,
        country=place.country,
        admin1=place.admin1,
        latitude=place.latitude,
        longitude=place.longitude,
    )


class OpenMeteoClient:
    def __init__(self, geocoding_url: str, forecast_url: str, timeout_seconds: float = 10):
        self.geocoding_url = geocoding_url
        self.forecast_url = forecast_url
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async def _get(self, url: str, params: dict) -> dict:
        # Операция чтения: допустим один повтор после временной ошибки.
        for attempt in (1, 2):
            try:
                async with (
                    aiohttp.ClientSession(timeout=self.timeout) as session,
                    session.get(url, params=params) as response,
                ):
                    if response.status >= 500 and attempt == 1:
                        await asyncio.sleep(0.5)
                        continue
                    if response.status >= 400:
                        raise WeatherError(
                            "unavailable", f"погодный сервис ответил HTTP {response.status}"
                        )
                    try:
                        data = await response.json(content_type=None)
                    except ValueError:
                        raise WeatherError(
                            "bad_response", "погодный сервис вернул не JSON"
                        ) from None
                    if not isinstance(data, dict):
                        raise WeatherError("bad_response", "неожиданная структура ответа")
                    return data
            except TimeoutError:
                if attempt == 2:
                    raise WeatherError("timeout", "погодный сервис не ответил вовремя") from None
            except aiohttp.ClientError:
                if attempt == 2:
                    raise WeatherError("unavailable", "погодный сервис недоступен") from None
        raise WeatherError("unavailable", "погодный сервис недоступен")

    async def current(self, city: str) -> WeatherOutput:
        name = city.partition(",")[0].strip()
        raw = await self._get(
            self.geocoding_url, {"name": name, "count": 10, "language": "ru", "format": "json"}
        )
        try:
            places = _Geocoding.model_validate(raw).results
        except ValidationError:
            raise WeatherError("bad_response", "неожиданный ответ геокодера") from None
        status, found = choose(city, places)
        if status != "ok":
            return WeatherOutput(status=status, query=city, candidates=[location(p) for p in found])
        place = found[0]
        raw = await self._get(
            self.forecast_url,
            {
                "latitude": place.latitude,
                "longitude": place.longitude,
                "current": "temperature_2m,weather_code,wind_speed_10m",
                "wind_speed_unit": "ms",
                "timezone": "auto",
            },
        )
        try:
            forecast = _Forecast.model_validate(raw)
            offset = timezone(timedelta(seconds=forecast.utc_offset_seconds))
            observed = datetime.fromisoformat(forecast.current.time).replace(tzinfo=offset)
        except (ValidationError, ValueError):
            raise WeatherError("bad_response", "неожиданный ответ погодного API") from None
        unit = forecast.current_units.get("wind_speed_10m", "m/s")
        return WeatherOutput(
            status="ok",
            query=city,
            location=location(place),
            current=CurrentWeather(
                temperature_c=forecast.current.temperature_2m,
                condition_code=forecast.current.weather_code,
                condition=WMO_CODES.get(forecast.current.weather_code, "нет описания"),
                wind_speed=forecast.current.wind_speed_10m,
                wind_speed_unit="м/с" if unit == "m/s" else unit,
                observed_at=observed.isoformat(),
                timezone=forecast.timezone,
            ),
        )
