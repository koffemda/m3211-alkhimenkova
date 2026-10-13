"""get_weather против локальной подмены Open-Meteo: без реальных HTTP-запросов (сценарии 1–3)."""

import asyncio

import pytest
from aiohttp import web
from mcp import Client

from mcp_server.schedule import ScheduleSource
from mcp_server.server import create_server
from mcp_server.weather import OpenMeteoClient, WeatherError
from tests.conftest import SECRET

SPB = {
    "name": "Санкт-Петербург",
    "latitude": 59.94,
    "longitude": 30.31,
    "country": "Россия",
    "admin1": "Санкт-Петербург",
    "population": 5_351_935,
}
SPB_FLORIDA = {
    "name": "Санкт-Петербург",
    "latitude": 27.77,
    "longitude": -82.68,
    "country": "США",
    "admin1": "Флорида",
    "population": 258_308,
}
FORECAST = {
    "timezone": "Europe/Moscow",
    "utc_offset_seconds": 10800,
    "current_units": {"temperature_2m": "°C", "wind_speed_10m": "m/s"},
    "current": {
        "time": "2026-10-12T09:00",
        "interval": 900,
        "temperature_2m": 7.4,
        "weather_code": 61,
        "wind_speed_10m": 4.1,
    },
}


@pytest.fixture
async def open_meteo(unused_tcp_port):
    """Сервер-подмена: очереди ответов для геокодера и прогноза."""
    queues = {"geocoding": [], "forecast": []}
    requests = []

    def handler(kind):
        async def handle(request: web.Request):
            requests.append((kind, dict(request.query)))
            answer = queues[kind].pop(0)
            return await answer() if callable(answer) else web.json_response(answer)

        return handle

    app = web.Application()
    app.router.add_get("/v1/search", handler("geocoding"))
    app.router.add_get("/v1/forecast", handler("forecast"))
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
    base = f"http://127.0.0.1:{unused_tcp_port}/v1"
    client = OpenMeteoClient(f"{base}/search", f"{base}/forecast", timeout_seconds=0.5)
    yield client, queues, requests
    await runner.cleanup()


async def test_weather_for_existing_city(open_meteo):
    # Arrange
    client, queues, requests = open_meteo
    queues["geocoding"].append({"results": [SPB]})
    queues["forecast"].append(FORECAST)
    # Act
    result = await client.current("Санкт-Петербург")
    # Assert
    assert result.status == "ok"
    assert result.location.name == "Санкт-Петербург" and result.location.country == "Россия"
    assert result.current.model_dump() == {
        "temperature_c": 7.4,
        "condition_code": 61,
        "condition": "небольшой дождь",
        "wind_speed": 4.1,
        "wind_speed_unit": "м/с",
        "observed_at": "2026-10-12T09:00:00+03:00",
        "timezone": "Europe/Moscow",
    }
    assert requests[0][1]["name"] == "Санкт-Петербург"
    assert requests[1][1]["latitude"] == "59.94"


async def test_city_not_found(open_meteo):
    # Arrange: геокодер без поля results означает «ничего не найдено».
    client, queues, _ = open_meteo
    queues["geocoding"].append({"generationtime_ms": 0.5})
    # Act
    result = await client.current("Нетакойгород")
    # Assert
    assert result.status == "not_found"
    assert result.candidates == []


async def test_same_name_cities_are_returned_as_candidates(open_meteo):
    # Arrange: два сопоставимых одноимённых города — выбирать автоматически нельзя.
    client, queues, requests = open_meteo
    small_russian = {**SPB, "admin1": "Ленинградская область", "population": 300_000}
    queues["geocoding"].append({"results": [small_russian, SPB_FLORIDA]})
    # Act
    result = await client.current("Санкт-Петербург")
    # Assert
    assert result.status == "ambiguous"
    assert [(c.country, c.admin1) for c in result.candidates] == [
        ("Россия", "Ленинградская область"),
        ("США", "Флорида"),
    ]
    assert [kind for kind, _ in requests] == ["geocoding"]  # прогноз не запрашивался


async def test_dominant_city_is_not_ambiguous(open_meteo):
    # Arrange: 5,3 млн против 258 тыс. — больше чем в 10 раз.
    client, queues, _ = open_meteo
    queues["geocoding"].append({"results": [SPB_FLORIDA, SPB]})
    queues["forecast"].append(FORECAST)
    # Act
    result = await client.current("Санкт-Петербург")
    # Assert
    assert result.status == "ok"
    assert result.location.country == "Россия"


async def test_country_after_comma_selects_candidate(open_meteo):
    # Arrange
    client, queues, requests = open_meteo
    queues["geocoding"].append({"results": [SPB, SPB_FLORIDA]})
    queues["forecast"].append(FORECAST)
    # Act
    result = await client.current("Санкт-Петербург, США")
    # Assert
    assert result.status == "ok"
    assert result.location.admin1 == "Флорида"
    assert requests[0][1]["name"] == "Санкт-Петербург"


async def test_timeout_after_retry(open_meteo):
    # Arrange
    client, queues, requests = open_meteo

    async def slow():
        await asyncio.sleep(2)
        return web.json_response({"results": [SPB]})

    queues["geocoding"] += [slow, slow]
    # Act / Assert
    with pytest.raises(WeatherError) as error:
        await client.current("Санкт-Петербург")
    assert error.value.code == "timeout"
    assert len(requests) == 2  # один повтор для операции чтения


@pytest.mark.parametrize(
    "forecast",
    [
        {"timezone": "Europe/Moscow"},  # нет current
        {**FORECAST, "current": {**FORECAST["current"], "temperature_2m": "тепло"}},
        {**FORECAST, "current": {**FORECAST["current"], "time": "вчера"}},
    ],
    ids=["no-current", "bad-type", "bad-time"],
)
async def test_unexpected_forecast_structure_is_bad_response(open_meteo, forecast):
    # Arrange
    client, queues, _ = open_meteo
    queues["geocoding"].append({"results": [SPB]})
    queues["forecast"].append(forecast)
    # Act / Assert
    with pytest.raises(WeatherError) as error:
        await client.current("Санкт-Петербург")
    assert error.value.code == "bad_response"


async def test_not_json_and_server_errors(open_meteo):
    # Arrange
    client, queues, _ = open_meteo

    async def html():
        return web.Response(text="<html>", content_type="text/html")

    async def down():
        return web.Response(status=503)

    queues["geocoding"] += [html, down, down]
    # Act / Assert
    with pytest.raises(WeatherError) as first:
        await client.current("Казань")
    with pytest.raises(WeatherError) as second:
        await client.current("Казань")
    assert (first.value.code, second.value.code) == ("bad_response", "unavailable")


async def test_weather_error_reaches_client_as_structured_error(
    open_meteo, schedule_file, reminder_store, clock
):
    # Arrange
    client, queues, _ = open_meteo

    async def slow():
        await asyncio.sleep(2)
        return web.json_response({})

    queues["geocoding"] += [slow, slow]
    server = create_server(
        weather=client,
        schedule=ScheduleSource(schedule_file, "Europe/Moscow", clock),
        reminders=reminder_store,
        trust_secret=SECRET,
        clock=clock,
    )
    # Act
    async with Client(server) as mcp:
        result = await mcp.call_tool("get_weather", {"city": "Казань"})
    # Assert
    assert result.is_error is True
    assert result.structured_content["error"]["code"] == "timeout"
