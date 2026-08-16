from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import httpx

from agent.vercel_ai_gateway_client import VercelAIGatewayClient
from run_agent import AIAgent

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "vercel_ai_gateway" / "sdk_oracle_v4.json"


def _agent(*, enabled_toolsets: list[str]) -> AIAgent:
    return AIAgent(
        api_key="test-key",
        base_url="https://ai-gateway.vercel.sh/v1",
        model="openai/gpt-5.6-luna",
        api_mode="vercel_ai_gateway",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=enabled_toolsets,
        reasoning_config={"enabled": True, "effort": "low"},
    )


def test_agent_initializes_native_client_and_builds_v4_request() -> None:
    agent = _agent(enabled_toolsets=["web"])
    try:
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "Search."}])
        assert isinstance(agent.client, VercelAIGatewayClient)
        transport = agent._get_transport()
        assert transport is not None
        assert transport.api_mode == "vercel_ai_gateway"
        assert kwargs["prompt"] == [
            {"role": "user", "content": [{"type": "text", "text": "Search."}]}
        ]
        assert kwargs["reasoning"] == "low"
        assert kwargs["tools"][-1]["id"] == "gateway.exa_search"
    finally:
        cast(VercelAIGatewayClient, agent.client).close()


def test_agent_respects_web_toolset_opt_out() -> None:
    agent = _agent(enabled_toolsets=[])
    try:
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "No web."}])
        assert "tools" not in kwargs
        assert "toolChoice" not in kwargs
    finally:
        cast(VercelAIGatewayClient, agent.client).close()


def test_nonstreaming_conversation_loop_normalizes_fixture(monkeypatch) -> None:
    fixture = json.loads(_FIXTURE.read_text())
    agent = _agent(enabled_toolsets=["web"])
    old_client = agent.client

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).endswith("/v4/ai/language-model")
        return httpx.Response(200, json=fixture["response"])

    agent.client = VercelAIGatewayClient(
        api_key="test-key",
        base_url="https://ai-gateway.vercel.sh/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    setattr(agent, "_disable_streaming", True)
    # Request-local clients are created inside the interruptible worker. Make
    # those deterministic as well, while retaining the real Hermes loop.
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kwargs: agent.client)
    try:
        result = agent.run_conversation("Search for the official page.")
        assert result["final_response"] == "The official page is titled Vercel AI Gateway."
        assert result["last_reasoning"] == "We need search."
        assert result["api_calls"] == 1
    finally:
        cast(VercelAIGatewayClient, old_client).close()
        cast(VercelAIGatewayClient, agent.client).close()


def test_streaming_conversation_loop_uses_sse_and_returns_final_text(monkeypatch) -> None:
    fixture = json.loads(_FIXTURE.read_text())
    sse = "".join(
        f"data: {json.dumps(event)}\n\n" for event in fixture["stream_events"]
    )
    agent = _agent(enabled_toolsets=["web"])
    old_client = agent.client

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["ai-language-model-streaming"] == "true"
        return httpx.Response(
            200,
            text=sse,
            headers={"content-type": "text/event-stream"},
        )

    agent.client = VercelAIGatewayClient(
        api_key="test-key",
        base_url="https://ai-gateway.vercel.sh/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kwargs: agent.client)
    try:
        result = agent.run_conversation("Search for the official page.")
        assert result["final_response"] == "Vercel AI Gateway"
        assert result["api_calls"] == 1
    finally:
        cast(VercelAIGatewayClient, old_client).close()
        cast(VercelAIGatewayClient, agent.client).close()


def test_local_tool_round_trip_replays_result_in_v4_prompt(monkeypatch) -> None:
    fixture_path = str(_FIXTURE)
    agent = _agent(enabled_toolsets=["file"])
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(
                200,
                json={
                    "content": [
                        {
                            "type": "tool-call",
                            "toolCallId": "read-fixture",
                            "toolName": "read_file",
                            "input": {"path": fixture_path, "offset": 1, "limit": 1},
                        }
                    ],
                    "finishReason": {"unified": "tool-calls", "raw": "tool_calls"},
                    "usage": {
                        "inputTokens": {"total": 10},
                        "outputTokens": {"total": 5},
                    },
                },
            )
        tool_messages = [m for m in body["prompt"] if m["role"] == "tool"]
        assert len(tool_messages) == 1
        result_part = tool_messages[0]["content"][0]
        assert result_part["toolCallId"] == "read-fixture"
        assert result_part["toolName"] == "read_file"
        assert result_part["output"]["type"] == "json"
        assert result_part["output"]["value"]["total_lines"] == 223
        assert result_part["output"]["value"]["content"].startswith("1|{")
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "The fixture was read."}],
                "finishReason": {"unified": "stop", "raw": "stop"},
                "usage": {
                    "inputTokens": {"total": 20},
                    "outputTokens": {"total": 5},
                },
            },
        )

    def request_client(**kwargs):
        return VercelAIGatewayClient(
            api_key="test-key",
            base_url="https://ai-gateway.vercel.sh/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

    setattr(agent, "_disable_streaming", True)
    monkeypatch.setattr(agent, "_create_request_openai_client", request_client)
    try:
        result = agent.run_conversation("Read the protocol fixture.")
        assert result["final_response"] == "The fixture was read."
        assert result["api_calls"] == 2
        assert len(calls) == 2
    finally:
        cast(VercelAIGatewayClient, agent.client).close()
