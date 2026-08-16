from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.transports import get_transport
from agent.transports.vercel_ai_gateway import (
    API_MODE,
    DEFAULT_SEARCH_CONFIG,
    _NATIVE_SEARCH_SAFETY_INSTRUCTION,
    _load_search_config,
    VercelAIGatewayTransport,
)

_FIXTURE = Path(__file__).parents[2] / "fixtures" / "vercel_ai_gateway" / "sdk_oracle_v4.json"


@pytest.fixture
def fixture() -> dict:
    return json.loads(_FIXTURE.read_text())


@pytest.fixture
def transport(monkeypatch) -> VercelAIGatewayTransport:
    config = json.loads(json.dumps(DEFAULT_SEARCH_CONFIG))
    config["enabled"] = True
    monkeypatch.setattr(
        "agent.transports.vercel_ai_gateway._load_search_config",
        lambda: config,
    )
    return VercelAIGatewayTransport()


def test_transport_is_registered() -> None:
    assert isinstance(get_transport(API_MODE), VercelAIGatewayTransport)


def test_converts_messages_and_local_tool_result(transport) -> None:
    messages = [
        {"role": "system", "content": "Be exact."},
        {"role": "user", "content": "Use the probe."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "probe_tool", "arguments": '{"value":7}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": '{"ok":true}'},
    ]

    prompt = transport.convert_messages(messages)

    assert prompt[0] == {"role": "system", "content": "Be exact."}
    assert prompt[2]["content"][0] == {
        "type": "tool-call",
        "toolCallId": "call-1",
        "toolName": "probe_tool",
        "input": {"value": 7},
    }
    assert prompt[3]["content"][0] == {
        "type": "tool-result",
        "toolCallId": "call-1",
        "toolName": "probe_tool",
        "output": {"type": "json", "value": {"ok": True}},
    }


def test_replaces_local_web_tools_with_provider_exa(transport) -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "local search",
                "parameters": {"type": "object"},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_extract",
                "description": "local extraction",
                "parameters": {"type": "object"},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        },
    ]

    converted = transport.convert_tools(tools)

    assert [tool["name"] for tool in converted] == ["read_file", "web_search"]
    assert converted[-1]["type"] == "provider"
    assert converted[-1]["id"] == "gateway.exa_search"
    assert converted[-1]["args"]["numResults"] == 5


def test_native_search_injected_without_local_web_backend(transport) -> None:
    converted = transport.convert_tools([])
    assert converted == [
        {
            "type": "provider",
            "name": "web_search",
            "id": "gateway.exa_search",
            "args": DEFAULT_SEARCH_CONFIG["exa"],
        }
    ]


def test_native_search_is_disabled_without_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    assert _load_search_config()["enabled"] is False
    assert VercelAIGatewayTransport().convert_tools([]) == []


def test_config_loader_failure_fails_closed(monkeypatch) -> None:
    def fail():
        raise RuntimeError("config unavailable")

    monkeypatch.setattr("hermes_cli.config.load_config", fail)
    config = _load_search_config()
    assert config["enabled"] is False
    assert config["zero_data_retention"] is True


def test_provider_search_adds_untrusted_content_system_boundary(transport) -> None:
    kwargs = transport.build_kwargs(
        "openai/gpt-5.6-luna",
        [{"role": "user", "content": "Search."}],
        tools=[],
    )
    assert kwargs["prompt"][0] == {
        "role": "system",
        "content": _NATIVE_SEARCH_SAFETY_INSTRUCTION,
    }


def test_provider_search_appends_boundary_to_existing_system_message(transport) -> None:
    kwargs = transport.build_kwargs(
        "openai/gpt-5.6-luna",
        [
            {"role": "system", "content": "Be exact."},
            {"role": "user", "content": "Search."},
        ],
        tools=[],
    )
    system = kwargs["prompt"][0]
    assert system["content"].startswith("Be exact.\n\n")
    assert system["content"].endswith(_NATIVE_SEARCH_SAFETY_INSTRUCTION)


def test_web_toolset_opt_out_does_not_inject_search(transport) -> None:
    assert transport.convert_tools([], native_search_enabled=False) == []


def test_build_kwargs_maps_low_reasoning_and_zdr(transport) -> None:
    kwargs = transport.build_kwargs(
        "openai/gpt-5.6-luna",
        [{"role": "user", "content": "Search."}],
        tools=[],
        max_tokens=256,
        reasoning_config={"enabled": True, "effort": "low"},
    )

    assert kwargs["model"] == "openai/gpt-5.6-luna"
    assert kwargs["reasoning"] == "low"
    assert kwargs["maxOutputTokens"] == 256
    assert kwargs["providerOptions"]["gateway"]["zeroDataRetention"] is True
    assert kwargs["tools"][0]["id"] == "gateway.exa_search"


def test_normalizes_provider_executed_search_without_local_call(transport, fixture) -> None:
    normalized = transport.normalize_response(fixture["response"])

    assert normalized.content == "The official page is titled Vercel AI Gateway."
    assert normalized.reasoning == "We need search."
    assert normalized.tool_calls is None
    assert normalized.finish_reason == "stop"
    assert normalized.usage.prompt_tokens == 120
    assert normalized.usage.completion_tokens == 24
    data = normalized.provider_data["vercel_ai_gateway"]
    assert data["provider_tool_calls"][0]["providerExecuted"] is True
    assert data["provider_tool_results"][0]["output"]["value"]["results"][0]["title"] == "Vercel AI Gateway"


def test_normalizes_local_tool_call_for_hermes_dispatch(transport, fixture) -> None:
    normalized = transport.normalize_response(fixture["local_tool_response"])

    assert normalized.finish_reason == "tool_calls"
    assert len(normalized.tool_calls) == 1
    assert normalized.tool_calls[0].id == "tool_fixture_local"
    assert normalized.tool_calls[0].name == "probe_tool"
    assert json.loads(normalized.tool_calls[0].arguments) == {"value": 7}


def test_fixture_pins_sdk_oracle_versions(fixture) -> None:
    assert fixture["oracle"] == {
        "captured_at": "2026-08-16",
        "ai": "7.0.66",
        "@ai-sdk/gateway": "4.0.52",
        "@ai-sdk/provider": "4.0.7",
        "language_model_specification_version": "4",
        "gateway_protocol_version": "0.0.1",
        "endpoint": "https://ai-gateway.vercel.sh/v4/ai/language-model",
    }
    package = json.loads(
        (Path(__file__).parents[3] / "scripts" / "vercel_ai_gateway_oracle" / "package.json").read_text()
    )
    assert package["dependencies"] == {
        "ai": fixture["oracle"]["ai"],
        "@ai-sdk/gateway": fixture["oracle"]["@ai-sdk/gateway"],
        "@ai-sdk/provider": fixture["oracle"]["@ai-sdk/provider"],
    }
