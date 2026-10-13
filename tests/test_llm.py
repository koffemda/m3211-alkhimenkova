"""HTTP-клиент LLM против локального тестового сервера: без настоящего API."""

import asyncio
import logging

import pytest
from aiohttp import web

from app import llm as llm_module
from app.config import Settings
from app.llm import (
    ChatMessage,
    LLMBadResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
    OpenAICompatibleClient,
    parse_completion,
)

API_KEY = "sk-test-very-secret"


def completion(text: str | None, finish: str = "stop") -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


@pytest.fixture
async def server(unused_tcp_port, monkeypatch):
    """Сервер отвечает по очереди заданными обработчиками и запоминает запросы."""
    monkeypatch.setattr(llm_module, "RETRY_DELAY_SECONDS", 0)
    responses: list = []
    requests: list[tuple[dict, dict]] = []

    async def handle(request: web.Request) -> web.StreamResponse:
        requests.append((dict(request.headers), await request.json()))
        return await responses.pop(0)()

    app = web.Application()
    app.router.add_post("/api/v1/chat/completions", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
    settings = Settings(
        bot_token="unused",
        postgres_password="unused",
        openrouter_api_key=API_KEY,
        openrouter_base_url=f"http://127.0.0.1:{unused_tcp_port}/api/v1",
        openrouter_model="test/model",
        llm_timeout_seconds=1,
        llm_max_tokens=321,
    )
    client = OpenAICompatibleClient(settings)
    yield client, responses, requests
    await client.close()
    await runner.cleanup()


def reply_json(body, status: int = 200):
    async def respond():
        return web.json_response(body, status=status)

    return respond


async def test_request_contains_model_roles_and_parameters(server):
    # Arrange
    client, responses, requests = server
    responses.append(reply_json(completion("  Ответ  ")))
    messages = [ChatMessage("system", "инструкция"), ChatMessage("user", "вопрос")]
    # Act
    result = await client.complete(messages, temperature=0.7)
    # Assert
    headers, payload = requests[0]
    assert headers["Authorization"] == f"Bearer {API_KEY}"
    assert payload == {
        "model": "test/model",
        "messages": [
            {"role": "system", "content": "инструкция"},
            {"role": "user", "content": "вопрос"},
        ],
        "temperature": 0.7,
        "max_tokens": 321,
    }
    assert result.text == "Ответ"
    assert (result.prompt_tokens, result.completion_tokens, result.total_tokens) == (11, 7, 18)
    assert result.truncated is False


async def test_temporary_error_is_retried_once(server):
    # Arrange
    client, responses, requests = server
    responses += [reply_json({"error": "busy"}, status=503), reply_json(completion("ok"))]
    # Act
    result = await client.complete([ChatMessage("user", "q")], temperature=0.0)
    # Assert
    assert result.text == "ok"
    assert len(requests) == 2


async def test_retry_limit_is_respected(server):
    # Arrange
    client, responses, requests = server
    responses += [reply_json({}, status=502), reply_json({}, status=502), reply_json({})]
    # Act / Assert
    with pytest.raises(LLMUnavailableError) as error:
        await client.complete([ChatMessage("user", "q")], temperature=0.0)
    assert error.value.status == 502
    assert len(requests) == 2


async def test_auth_error_is_not_retried_and_key_is_not_logged(server, caplog):
    # Arrange
    client, responses, requests = server
    responses.append(reply_json({"error": {"message": f"bad key {API_KEY}"}}, status=401))
    # Act
    with caplog.at_level(logging.INFO), pytest.raises(LLMUnavailableError) as error:
        await client.complete([ChatMessage("user", "секретный вопрос")], temperature=0.0)
    # Assert
    assert error.value.status == 401
    assert len(requests) == 1
    assert API_KEY not in caplog.text
    assert "секретный вопрос" not in caplog.text
    assert "test/model" in caplog.text


async def test_timeout_raises_timeout_error(server):
    # Arrange
    client, responses, _ = server

    async def slow():
        await asyncio.sleep(3)
        return web.json_response(completion("поздно"))

    responses.append(slow)
    # Act / Assert
    with pytest.raises(LLMTimeoutError):
        await client.complete([ChatMessage("user", "q")], temperature=0.0)


async def test_invalid_json_is_bad_response(server):
    # Arrange
    client, responses, _ = server

    async def broken():
        return web.Response(text="<html>gateway</html>", content_type="text/html")

    responses.append(broken)
    # Act / Assert
    with pytest.raises(LLMBadResponseError):
        await client.complete([ChatMessage("user", "q")], temperature=0.0)


async def test_unreachable_service_is_unavailable(unused_tcp_port, monkeypatch):
    # Arrange: на порту никто не слушает.
    monkeypatch.setattr(llm_module, "RETRY_DELAY_SECONDS", 0)
    client = OpenAICompatibleClient(
        Settings(
            bot_token="unused",
            postgres_password="unused",
            openrouter_api_key=API_KEY,
            openrouter_base_url=f"http://127.0.0.1:{unused_tcp_port}/v1",
        )
    )
    try:
        # Act / Assert
        with pytest.raises(LLMUnavailableError):
            await client.complete([ChatMessage("user", "q")], temperature=0.0)
    finally:
        await client.close()


@pytest.mark.parametrize(
    "body",
    [
        [],
        {},
        {"choices": []},
        {"choices": [{"message": None}]},
        completion(None),
        completion("   "),
        {"choices": [{"message": {"content": [{"type": "text"}]}}]},
    ],
    ids=["list", "empty", "no-choices", "no-message", "null", "blank", "not-string"],
)
def test_response_without_text_is_rejected(body):
    with pytest.raises(LLMBadResponseError):
        parse_completion(body)


def test_error_in_successful_http_body_is_unavailable():
    # OpenRouter может вернуть ошибку провайдера в теле ответа с кодом 200.
    with pytest.raises(LLMUnavailableError):
        parse_completion({"error": {"code": 502, "message": "provider down"}})


def test_length_finish_reason_marks_truncated_answer():
    assert parse_completion(completion("часть", finish="length")).truncated is True


async def test_tools_are_sent_and_tool_calls_parsed(server):
    # Arrange
    client, responses, requests = server
    responses.append(
        reply_json(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "Казань"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
    )
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    # Act
    result = await client.complete([ChatMessage("user", "погода")], temperature=0.0, tools=tools)
    # Assert
    _, payload = requests[0]
    assert payload["tools"] == tools
    assert (payload["tool_choice"], payload["parallel_tool_calls"]) == ("auto", False)
    assert result.text == ""
    assert [(c.id, c.name, c.arguments) for c in result.tool_calls] == [
        ("call_1", "get_weather", '{"city": "Казань"}')
    ]


def test_tool_messages_are_serialized_with_roles():
    # Arrange
    from app.llm import ToolCall

    call = ToolCall("call_1", "get_schedule", '{"date": "2026-10-13"}')
    # Act
    assistant = ChatMessage("assistant", "", tool_calls=(call,)).as_dict()
    tool = ChatMessage("tool", '{"lessons": []}', tool_call_id="call_1").as_dict()
    # Assert
    assert assistant == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_schedule", "arguments": '{"date": "2026-10-13"}'},
            }
        ],
    }
    assert tool == {"role": "tool", "tool_call_id": "call_1", "content": '{"lessons": []}'}


@pytest.mark.parametrize(
    "message",
    [
        {"tool_calls": "x"},
        {"tool_calls": [{"function": {"name": "f"}}]},
        {"tool_calls": [{"id": "1", "function": {"name": 5, "arguments": "{}"}}]},
    ],
)
def test_malformed_tool_calls_are_rejected(message):
    with pytest.raises(LLMBadResponseError):
        parse_completion({"choices": [{"message": message}]})
