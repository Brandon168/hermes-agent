from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from agent.vercel_ai_gateway_client import (
    GATEWAY_PROTOCOL_VERSION,
    LANGUAGE_MODEL_SPECIFICATION_VERSION,
    VercelAIGatewayClient,
    VercelAIGatewayError,
    VercelAIGatewayProtocolError,
)

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "vercel_ai_gateway" / "sdk_oracle_v4.json"


@pytest.fixture
def fixture() -> dict:
    return json.loads(_FIXTURE.read_text())


def _client(handler) -> VercelAIGatewayClient:
    return VercelAIGatewayClient(
        api_key="test-key",
        base_url="https://ai-gateway.vercel.sh/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_nonstream_request_uses_exact_v4_contract(fixture) -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=fixture["response"])

    client = _client(handler)
    try:
        response = client.chat.completions.create(**fixture["request"])
    finally:
        client.close()

    request = captured["request"]
    assert str(request.url) == "https://ai-gateway.vercel.sh/v4/ai/language-model"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["ai-gateway-auth-method"] == "api-key"
    assert request.headers["ai-gateway-protocol-version"] == GATEWAY_PROTOCOL_VERSION
    assert request.headers["ai-language-model-specification-version"] == LANGUAGE_MODEL_SPECIFICATION_VERSION
    assert request.headers["ai-language-model-id"] == "openai/gpt-5.6-luna"
    assert request.headers["ai-language-model-streaming"] == "false"
    assert captured["body"] == {k: v for k, v in fixture["request"].items() if k != "model"}
    assert response["content"][-1]["text"].startswith("The official page")


def test_stream_projects_text_and_retains_provider_search_metadata(fixture) -> None:
    sse = "".join(f"data: {json.dumps(event)}\n\n" for event in fixture["stream_events"])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["ai-language-model-streaming"] == "true"
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    client = _client(handler)
    try:
        stream = client.chat.completions.create(**fixture["request"], stream=True)
        chunks = list(stream)
    finally:
        client.close()

    text = "".join(
        chunk.choices[0].delta.content or ""
        for chunk in chunks
        if chunk.choices
    )
    assert text == "Vercel AI Gateway"
    final = next(chunk for chunk in chunks if chunk.usage is not None)
    assert final.choices[0].finish_reason == "stop"
    assert final.usage.prompt_tokens == 120
    metadata = final.usage._vercel_provider_data
    assert metadata["provider_tool_calls"][0]["providerExecuted"] is True
    assert metadata["provider_tool_results"][0]["providerExecuted"] is True
    assert metadata["provider_metadata"]["gateway"]["cost"] == "0.0004"


def test_stream_projects_only_local_tool_calls(fixture) -> None:
    events = [
        {"type": "stream-start", "warnings": []},
        {
            "type": "tool-call",
            "toolCallId": "provider-1",
            "toolName": "web_search",
            "input": {"query": "hidden"},
            "providerExecuted": True,
        },
        {
            "type": "tool-call",
            "toolCallId": "local-1",
            "toolName": "read_file",
            "input": {"path": "/tmp/a"},
        },
        {
            "type": "finish",
            "finishReason": {"unified": "tool-calls", "raw": "tool_calls"},
            "usage": {"inputTokens": {"total": 1}, "outputTokens": {"total": 1}},
            "providerMetadata": {},
        },
    ]
    sse = "".join(f"data: {json.dumps(event)}\n\n" for event in events)

    client = _client(lambda request: httpx.Response(200, text=sse))
    try:
        chunks = list(client.chat.completions.create(**fixture["request"], stream=True))
    finally:
        client.close()

    tool_chunks = [c for c in chunks if c.choices and c.choices[0].delta.tool_calls]
    assert len(tool_chunks) == 1
    call = tool_chunks[0].choices[0].delta.tool_calls[0]
    assert call.id == "local-1"
    assert call.function.name == "read_file"
    assert json.loads(call.function.arguments) == {"path": "/tmp/a"}


def test_unknown_stream_event_fails_closed(fixture) -> None:
    sse = 'data: {"type":"future-critical-event"}\n\n'
    client = _client(lambda request: httpx.Response(200, text=sse))
    try:
        with pytest.raises(VercelAIGatewayProtocolError, match="Unknown Vercel v4 SSE event"):
            list(client.chat.completions.create(**fixture["request"], stream=True))
    finally:
        client.close()


def test_http_error_preserves_status_for_hermes_retry_classifier(fixture) -> None:
    client = _client(
        lambda request: httpx.Response(
            429,
            json={"error": {"code": "rate_limit", "message": "slow down"}},
        )
    )
    try:
        with pytest.raises(VercelAIGatewayError, match="slow down") as caught:
            client.chat.completions.create(**fixture["request"])
    finally:
        client.close()
    assert caught.value.status_code == 429
    assert caught.value.code == "rate_limit"


def test_rejects_non_vercel_host_before_sending() -> None:
    with pytest.raises(ValueError, match="exact HTTPS host"):
        VercelAIGatewayClient(
            api_key="test-key",
            base_url="https://lookalike.example/v1",
        )


def test_openai_message_compatibility_does_not_inject_search() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "finishReason": {"unified": "stop", "raw": "stop"},
                "usage": {"inputTokens": {"total": 1}, "outputTokens": {"total": 1}},
            },
        )

    client = _client(handler)
    try:
        client.chat.completions.create(
            model="openai/gpt-5.6-luna",
            messages=[{"role": "user", "content": "hello"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "probe",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )
    finally:
        client.close()

    assert [tool["name"] for tool in captured["tools"]] == ["probe"]
