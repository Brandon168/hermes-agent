"""Vercel AI Gateway Language Model v4 transport.

This transport is intentionally narrow: it implements the public AI SDK v4
message/tool/result types Hermes needs and replaces the logical ``web_search``
tool with Vercel's provider-executed Exa search.  The HTTP runtime lives in
:mod:`agent.vercel_ai_gateway_client`; keeping conversion here lets the normal
Hermes conversation loop continue to own local tool approval and execution.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any
from urllib.parse import urlparse

from agent.transports import register_transport
from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, ToolCall, Usage

API_MODE = "vercel_ai_gateway"
LANGUAGE_MODEL_SPECIFICATION_VERSION = "4"
GATEWAY_PROTOCOL_VERSION = "0.0.1"
DEFAULT_V4_BASE_URL = "https://ai-gateway.vercel.sh/v4/ai"
DEFAULT_SEARCH_CONFIG: dict[str, Any] = {
    "enabled": True,
    "provider": "exa",
    "exclude_web_extract": True,
    "zero_data_retention": True,
    "exa": {
        "type": "fast",
        "numResults": 5,
        "contents": {
            "text": {"maxCharacters": 8000},
            "highlights": {"maxCharacters": 800},
        },
    },
}
_PROVIDER_TOOL_IDS = {
    "exa": "gateway.exa_search",
    "parallel": "gateway.parallel_search",
    "perplexity": "gateway.perplexity_search",
}
_FINISH_REASONS = {
    "stop": "stop",
    "length": "length",
    "content-filter": "content_filter",
    "tool-calls": "tool_calls",
    "error": "stop",
    "other": "stop",
}


def _json_value(value: Any) -> Any:
    """Return a JSON-safe value without importing provider SDK types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_value(value.model_dump(warnings=False))
        except TypeError:
            return _json_value(value.model_dump())
    if hasattr(value, "__dict__"):
        return {
            str(k): _json_value(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return str(value)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _load_search_config() -> dict[str, Any]:
    """Load ``web.vercel_ai_gateway`` and merge it over safe defaults."""
    config = deepcopy(DEFAULT_SEARCH_CONFIG)
    try:
        from hermes_cli.config import load_config

        raw = ((load_config() or {}).get("web") or {}).get("vercel_ai_gateway") or {}
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        return config

    for key in ("enabled", "provider", "exclude_web_extract", "zero_data_retention"):
        if key in raw:
            config[key] = raw[key]
    provider = str(config.get("provider") or "exa").strip().lower()
    provider_options = raw.get(provider)
    if isinstance(provider_options, dict):
        config[provider] = _camelize_provider_config(provider_options)
    gateway_options = raw.get("gateway")
    if isinstance(gateway_options, dict):
        config["gateway"] = _camelize_provider_config(gateway_options)
    return config


def _camelize_provider_config(value: Any) -> Any:
    """Accept Hermes-style snake_case while emitting AI SDK field names."""
    aliases = {
        "num_results": "numResults",
        "user_location": "userLocation",
        "include_domains": "includeDomains",
        "exclude_domains": "excludeDomains",
        "start_published_date": "startPublishedDate",
        "end_published_date": "endPublishedDate",
        "max_characters": "maxCharacters",
        "include_html_tags": "includeHtmlTags",
        "include_sections": "includeSections",
        "exclude_sections": "excludeSections",
        "max_age_hours": "maxAgeHours",
        "livecrawl_timeout": "livecrawlTimeout",
        "subpage_target": "subpageTarget",
        "image_links": "imageLinks",
        "zero_data_retention": "zeroDataRetention",
    }
    if isinstance(value, dict):
        return {
            aliases.get(str(k), str(k)): _camelize_provider_config(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_camelize_provider_config(v) for v in value]
    return value


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") in {"text", "input_text"}:
            parts.append(str(part.get("text") or ""))
    return "\n".join(p for p in parts if p)


def _content_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        if content is None:
            return []
        return [{"type": "text", "text": str(content)}]

    converted: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            converted.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        ptype = str(part.get("type") or "")
        if ptype in {"text", "input_text"}:
            converted.append({"type": "text", "text": str(part.get("text") or "")})
            continue
        if ptype in {"image_url", "input_image"}:
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            if not url:
                url = part.get("image_url") or part.get("url")
            if not isinstance(url, str) or not url:
                continue
            media_type = "image"
            if url.startswith("data:") and ";base64," in url:
                prefix, data = url.split(",", 1)
                media_type = prefix[5:].split(";", 1)[0] or "image"
                file_data: dict[str, Any] = {"type": "data", "data": data}
            else:
                file_data = {"type": "url", "url": url}
            converted.append({"type": "file", "data": file_data, "mediaType": media_type})
            continue
        if ptype in {"file", "input_file"}:
            url = part.get("url") or part.get("file_url")
            data = part.get("data")
            media_type = str(part.get("media_type") or part.get("mediaType") or "application/octet-stream")
            if isinstance(url, str) and url:
                converted.append({"type": "file", "data": {"type": "url", "url": url}, "mediaType": media_type})
            elif isinstance(data, str) and data:
                converted.append({"type": "file", "data": {"type": "data", "data": data}, "mediaType": media_type})
    return converted


def _tool_function(tool: Any) -> dict[str, Any]:
    if not isinstance(tool, dict):
        return {}
    function = tool.get("function")
    return function if isinstance(function, dict) else tool


def _tool_name(tool: Any) -> str:
    return str(_tool_function(tool).get("name") or "")


def _parse_tool_input(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return _json_value(arguments)


def _tool_output(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return {"type": "text", "value": content}
        return {"type": "json", "value": _json_value(parsed)}
    if isinstance(content, (dict, list, int, float, bool)) or content is None:
        return {"type": "json", "value": _json_value(content)}
    return {"type": "text", "value": str(content)}


class VercelAIGatewayTransport(ProviderTransport):
    @property
    def api_mode(self) -> str:
        return API_MODE

    def convert_messages(self, messages: list[dict[str, Any]], **_: Any) -> list[dict[str, Any]]:
        call_names: dict[str, str] = {}
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                call_id = str(_get(call, "id") or _get(call, "call_id") or "")
                function = _get(call, "function", {})
                name = str(_get(function, "name") or _get(call, "name") or "")
                if call_id and name:
                    call_names[call_id] = name

        prompt: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "")
            if role == "system":
                text = _content_to_text(message.get("content"))
                if text:
                    prompt.append({"role": "system", "content": text})
                continue
            if role == "user":
                parts = _content_parts(message.get("content"))
                if parts:
                    prompt.append({"role": "user", "content": parts})
                continue
            if role == "assistant":
                parts = _content_parts(message.get("content"))
                reasoning = message.get("reasoning_content") or message.get("reasoning")
                if isinstance(reasoning, str) and reasoning:
                    parts.insert(0, {"type": "reasoning", "text": reasoning})
                for call in message.get("tool_calls") or []:
                    call_id = str(_get(call, "id") or _get(call, "call_id") or "")
                    function = _get(call, "function", {})
                    name = str(_get(function, "name") or _get(call, "name") or "")
                    arguments = _get(function, "arguments", _get(call, "arguments", {}))
                    if call_id and name:
                        parts.append({
                            "type": "tool-call",
                            "toolCallId": call_id,
                            "toolName": name,
                            "input": _parse_tool_input(arguments),
                        })
                if parts:
                    prompt.append({"role": "assistant", "content": parts})
                continue
            if role == "tool":
                call_id = str(message.get("tool_call_id") or "")
                name = str(message.get("name") or message.get("tool_name") or call_names.get(call_id) or "tool")
                if call_id:
                    prompt.append({
                        "role": "tool",
                        "content": [{
                            "type": "tool-result",
                            "toolCallId": call_id,
                            "toolName": name,
                            "output": _tool_output(message.get("content")),
                        }],
                    })
        return prompt

    def convert_tools(
        self,
        tools: list[dict[str, Any]] | None,
        *,
        native_search_enabled: bool = True,
        **_: Any,
    ) -> list[dict[str, Any]]:
        config = _load_search_config()
        converted: list[dict[str, Any]] = []
        saw_web_search = False
        for tool in tools or []:
            name = _tool_name(tool)
            if name == "web_search":
                saw_web_search = True
                continue
            if name == "web_extract" and bool(config.get("exclude_web_extract", True)):
                continue
            function = _tool_function(tool)
            if not name:
                continue
            item: dict[str, Any] = {
                "type": "function",
                "name": name,
                "inputSchema": deepcopy(function.get("parameters") or function.get("inputSchema") or {"type": "object", "properties": {}}),
            }
            if function.get("description"):
                item["description"] = str(function["description"])
            if "strict" in function:
                item["strict"] = bool(function["strict"])
            converted.append(item)

        # Provider-native search does not need the local web backend's key or
        # registry availability. Inject it whenever this transport is active;
        # if the local schema was present, it is replaced rather than duplicated.
        if native_search_enabled and bool(config.get("enabled", True)):
            provider = str(config.get("provider") or "exa").strip().lower()
            tool_id = _PROVIDER_TOOL_IDS.get(provider)
            if tool_id is None:
                raise ValueError(
                    "web.vercel_ai_gateway.provider must be one of: "
                    + ", ".join(sorted(_PROVIDER_TOOL_IDS))
                )
            args = config.get(provider) if isinstance(config.get(provider), dict) else {}
            converted.append({
                "type": "provider",
                "name": "web_search",
                "id": tool_id,
                "args": deepcopy(args),
            })
        elif saw_web_search:
            # Explicitly disabled means no silent fallback to Firecrawl/Tavily
            # inside a Vercel-native session.
            pass
        return converted

    def build_kwargs(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        config = _load_search_config()
        request: dict[str, Any] = {
            "model": model,
            "prompt": self.convert_messages(messages),
        }
        converted_tools = self.convert_tools(
            tools,
            native_search_enabled=bool(params.get("native_search_enabled", True)),
        )
        if converted_tools:
            request["tools"] = converted_tools
            request["toolChoice"] = {"type": "auto"}

        max_tokens = params.get("max_tokens")
        if isinstance(max_tokens, int) and max_tokens > 0:
            request["maxOutputTokens"] = max_tokens

        reasoning_config = params.get("reasoning_config")
        if isinstance(reasoning_config, dict):
            enabled = reasoning_config.get("enabled", True)
            effort = str(reasoning_config.get("effort") or "").strip().lower()
            if enabled is False:
                request["reasoning"] = "none"
            elif effort in {"none", "minimal", "low", "medium", "high", "xhigh"}:
                request["reasoning"] = effort

        overrides = params.get("request_overrides")
        if isinstance(overrides, dict):
            option_map = {
                "temperature": "temperature",
                "top_p": "topP",
                "topP": "topP",
                "top_k": "topK",
                "topK": "topK",
                "seed": "seed",
            }
            for source, target in option_map.items():
                if source in overrides:
                    request[target] = overrides[source]

        gateway_options = deepcopy(config.get("gateway") or {})
        gateway_options.setdefault(
            "zeroDataRetention", bool(config.get("zero_data_retention", True))
        )
        request["providerOptions"] = {"gateway": gateway_options}
        request["headers"] = {"user-agent": "hermes-agent/vercel-ai-gateway-v4"}
        timeout = params.get("timeout")
        if timeout is not None:
            request["timeout"] = timeout
        return request

    def normalize_response(self, response: Any, **_: Any) -> NormalizedResponse:
        if isinstance(response, NormalizedResponse):
            return response
        if not isinstance(response, dict) or "content" not in response:
            # Streaming is projected through OpenAI-shaped chunks so Hermes can
            # retain its mature stream/cancellation machinery.
            from agent.transports.chat_completions import ChatCompletionsTransport

            normalized = ChatCompletionsTransport().normalize_response(response)
            usage = _get(response, "usage")
            provider_data = _get(usage, "_vercel_provider_data")
            if isinstance(provider_data, dict):
                normalized.provider_data = {
                    **(normalized.provider_data or {}),
                    "vercel_ai_gateway": provider_data,
                }
            return normalized

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        local_calls: list[ToolCall] = []
        provider_calls: list[dict[str, Any]] = []
        provider_results: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []
        other_content: list[dict[str, Any]] = []

        for part in response.get("content") or []:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text" and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
            elif ptype == "reasoning" and isinstance(part.get("text"), str):
                reasoning_parts.append(part["text"])
            elif ptype == "tool-call":
                if part.get("providerExecuted") is True:
                    provider_calls.append(_json_value(part))
                else:
                    raw_input = part.get("input")
                    arguments: str
                    if isinstance(raw_input, str):
                        arguments = raw_input
                    else:
                        arguments = json.dumps(
                            _json_value(raw_input), separators=(",", ":")
                        )
                    local_calls.append(ToolCall(
                        id=str(part.get("toolCallId") or "") or None,
                        name=str(part.get("toolName") or ""),
                        arguments=arguments,
                        provider_data={
                            "vercel_ai_gateway": {
                                k: _json_value(v)
                                for k, v in part.items()
                                if k not in {"type", "toolCallId", "toolName", "input"}
                            }
                        },
                    ))
            elif ptype == "tool-result":
                provider_results.append(_json_value(part))
            elif ptype == "source":
                sources.append(_json_value(part))
            else:
                other_content.append(_json_value(part))

        finish = response.get("finishReason")
        unified = finish.get("unified") if isinstance(finish, dict) else finish
        finish_reason = _FINISH_REASONS.get(str(unified or "stop"), "stop")
        if local_calls:
            finish_reason = "tool_calls"

        raw_usage = response.get("usage") or {}
        input_usage = raw_usage.get("inputTokens") or {}
        output_usage = raw_usage.get("outputTokens") or {}
        prompt_tokens = int(input_usage.get("total") or 0)
        completion_tokens = int(output_usage.get("total") or 0)
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cached_tokens=int(input_usage.get("cacheRead") or 0),
        )
        provider_data = {
            "vercel_ai_gateway": {
                "specification_version": LANGUAGE_MODEL_SPECIFICATION_VERSION,
                "protocol_version": GATEWAY_PROTOCOL_VERSION,
                "provider_metadata": _json_value(response.get("providerMetadata") or {}),
                "warnings": _json_value(response.get("warnings") or []),
                "provider_tool_calls": provider_calls,
                "provider_tool_results": provider_results,
                "sources": sources,
                "other_content": other_content,
                "raw_usage": _json_value(raw_usage),
            }
        }
        return NormalizedResponse(
            content="".join(text_parts) or None,
            tool_calls=local_calls or None,
            finish_reason=finish_reason,
            reasoning="".join(reasoning_parts) or None,
            usage=usage,
            provider_data=provider_data,
        )


register_transport(API_MODE, VercelAIGatewayTransport)
