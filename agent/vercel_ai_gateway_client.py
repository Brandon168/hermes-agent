"""Direct Python client for Vercel AI Gateway's Language Model v4 wire.

The client deliberately exposes the small ``client.chat.completions.create``
facade Hermes already knows how to manage.  Requests and responses remain v4;
the facade only lets Hermes reuse its request-local client, socket-abort,
stream-staleness, retry, and WebUI token-streaming machinery.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Iterator
from urllib.parse import urlparse

import httpx

from agent.transports.vercel_ai_gateway import (
    GATEWAY_PROTOCOL_VERSION,
    LANGUAGE_MODEL_SPECIFICATION_VERSION,
    VercelAIGatewayTransport,
    _json_value,
)

_LANGUAGE_MODEL_PATH = "/v4/ai/language-model"
_KNOWN_STREAM_TYPES = {
    "stream-start",
    "response-metadata",
    "text-start",
    "text-delta",
    "text-end",
    "reasoning-start",
    "reasoning-delta",
    "reasoning-end",
    "tool-input-start",
    "tool-input-delta",
    "tool-input-end",
    "tool-approval-request",
    "tool-call",
    "tool-result",
    "file",
    "reasoning-file",
    "source",
    "custom",
    "raw",
    "finish",
    "error",
}


class VercelAIGatewayError(RuntimeError):
    """HTTP or protocol failure with the attributes Hermes retry logic reads."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: Any = None,
        response: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.response = response
        self.code = _error_code(body)


class VercelAIGatewayProtocolError(VercelAIGatewayError):
    pass


def _error_code(body: Any) -> str | None:
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return str(error.get("code") or error.get("type") or "") or None
        if error:
            return str(error)
    return None


def _endpoint(base_url: str) -> str:
    parsed = urlparse((base_url or "").strip())
    if parsed.scheme != "https" or parsed.hostname != "ai-gateway.vercel.sh":
        raise ValueError(
            "vercel_ai_gateway transport requires the exact HTTPS host "
            "ai-gateway.vercel.sh"
        )
    return f"{parsed.scheme}://{parsed.netloc}{_LANGUAGE_MODEL_PATH}"


def _timeout_value(timeout: Any) -> Any:
    if timeout is None:
        return None
    if isinstance(timeout, (int, float, httpx.Timeout)):
        return timeout
    return None


def _as_v4_request(kwargs: dict[str, Any]) -> tuple[str, dict[str, Any], bool, Any]:
    values = dict(kwargs)
    stream = bool(values.pop("stream", False))
    values.pop("stream_options", None)
    timeout = _timeout_value(values.pop("timeout", None))
    model = str(values.pop("model", "") or "")
    if not model:
        raise ValueError("Vercel AI Gateway v4 request requires a model")

    if "prompt" not in values and "messages" in values:
        transport = VercelAIGatewayTransport()
        values["prompt"] = transport.convert_messages(values.pop("messages") or [])
        if "tools" in values:
            values["tools"] = transport.convert_tools(
                values.get("tools"), native_search_enabled=False
            )

    # Auxiliary callers can still pass OpenAI spellings. Keep this mapper
    # intentionally narrow; unknown fields fail locally instead of drifting
    # silently into Vercel's private protocol.
    aliases = {
        "max_tokens": "maxOutputTokens",
        "max_completion_tokens": "maxOutputTokens",
        "top_p": "topP",
        "top_k": "topK",
        "presence_penalty": "presencePenalty",
        "frequency_penalty": "frequencyPenalty",
        "stop": "stopSequences",
        "tool_choice": "toolChoice",
        "response_format": "responseFormat",
    }
    for source, target in aliases.items():
        if source in values and target not in values:
            values[target] = values.pop(source)

    tool_choice = values.get("toolChoice")
    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "none", "required"}:
            values["toolChoice"] = {"type": tool_choice}
    elif isinstance(tool_choice, dict) and "function" in tool_choice:
        function = tool_choice.get("function") or {}
        values["toolChoice"] = {
            "type": "tool",
            "toolName": function.get("name"),
        }

    extra_body = values.pop("extra_body", None)
    if isinstance(extra_body, dict):
        for key in ("providerOptions", "reasoning", "responseFormat"):
            if key in extra_body and key not in values:
                values[key] = extra_body[key]

    reasoning_effort = values.pop("reasoning_effort", None)
    if reasoning_effort and "reasoning" not in values:
        values["reasoning"] = str(reasoning_effort)

    # These OpenAI client controls are never protocol body fields.
    for local_only in ("user", "metadata"):
        values.pop(local_only, None)

    allowed = {
        "maxOutputTokens",
        "temperature",
        "topP",
        "topK",
        "presencePenalty",
        "frequencyPenalty",
        "stopSequences",
        "seed",
        "reasoning",
        "tools",
        "toolChoice",
        "responseFormat",
        "prompt",
        "providerOptions",
        "headers",
    }
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise VercelAIGatewayProtocolError(
            "Unsupported Vercel v4 request fields: " + ", ".join(unknown)
        )
    if not isinstance(values.get("prompt"), list):
        raise VercelAIGatewayProtocolError("Vercel v4 prompt must be a list")
    return model, _json_value(values), stream, timeout


def _response_error(response: httpx.Response) -> VercelAIGatewayError:
    try:
        body = response.json()
    except Exception:
        body = response.text[:4000]
    message = f"Vercel AI Gateway request failed with HTTP {response.status_code}"
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            message = str(error["message"])
        elif error:
            message = str(error)
    return VercelAIGatewayError(
        message,
        status_code=response.status_code,
        body=body,
        response=response,
    )


def _usage_object(usage: Any, provider_data: dict[str, Any]) -> SimpleNamespace:
    usage = usage if isinstance(usage, dict) else {}
    input_tokens = usage.get("inputTokens") or {}
    output_tokens = usage.get("outputTokens") or {}
    prompt = int(input_tokens.get("total") or 0)
    completion = int(output_tokens.get("total") or 0)
    details = SimpleNamespace(
        cached_tokens=int(input_tokens.get("cacheRead") or 0),
        cache_write_tokens=int(input_tokens.get("cacheWrite") or 0),
    )
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=details,
        _vercel_provider_data=provider_data,
    )


def _chunk(
    *,
    model: str,
    content: str | None = None,
    reasoning: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
    usage: Any = None,
) -> SimpleNamespace:
    delta = SimpleNamespace(
        role="assistant",
        content=content,
        reasoning_content=reasoning,
        tool_calls=tool_calls,
    )
    choices = []
    if any((content is not None, reasoning is not None, tool_calls, finish_reason is not None)):
        choices = [SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)]
    return SimpleNamespace(model=model, choices=choices, usage=usage)


class VercelV4Stream:
    """SSE iterator projected to OpenAI-like chunks for Hermes' stream loop."""

    final_response = None

    def __init__(
        self,
        *,
        http_client: httpx.Client,
        endpoint: str,
        headers: dict[str, str],
        body: dict[str, Any],
        model: str,
        timeout: Any,
    ) -> None:
        self._context = http_client.stream(
            "POST",
            endpoint,
            headers=headers,
            json=body,
            timeout=timeout,
        )
        self.response = self._context.__enter__()
        self._closed = False
        if self.response.status_code >= 400:
            try:
                self.response.read()
                raise _response_error(self.response)
            finally:
                self.close()
        self._model = model
        self.provider_data: dict[str, Any] = {
            "specification_version": LANGUAGE_MODEL_SPECIFICATION_VERSION,
            "protocol_version": GATEWAY_PROTOCOL_VERSION,
            "provider_tool_calls": [],
            "provider_tool_results": [],
            "sources": [],
            "warnings": [],
            "response_metadata": {},
        }
        self._local_tool_index = 0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._context.__exit__(None, None, None)

    def __iter__(self) -> Iterator[SimpleNamespace]:
        try:
            for line in self.response.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise VercelAIGatewayProtocolError(
                        "Malformed JSON in Vercel v4 SSE event"
                    ) from exc
                if not isinstance(event, dict):
                    raise VercelAIGatewayProtocolError(
                        "Vercel v4 SSE event must be an object"
                    )
                event_type = str(event.get("type") or "")
                if event_type not in _KNOWN_STREAM_TYPES:
                    raise VercelAIGatewayProtocolError(
                        f"Unknown Vercel v4 SSE event type: {event_type or '(missing)'}"
                    )

                if event_type == "error":
                    raise VercelAIGatewayError(
                        f"Vercel v4 stream error: {event.get('error')}",
                        body=event.get("error"),
                        response=self.response,
                    )
                if event_type == "stream-start":
                    self.provider_data["warnings"] = _json_value(event.get("warnings") or [])
                elif event_type == "response-metadata":
                    self.provider_data["response_metadata"] = _json_value(event)
                    if event.get("modelId"):
                        self._model = str(event["modelId"])
                elif event_type == "tool-call":
                    if event.get("providerExecuted") is True:
                        self.provider_data["provider_tool_calls"].append(_json_value(event))
                    else:
                        tool_call = SimpleNamespace(
                            index=self._local_tool_index,
                            id=str(event.get("toolCallId") or ""),
                            function=SimpleNamespace(
                                name=str(event.get("toolName") or ""),
                                arguments=(
                                    event.get("input")
                                    if isinstance(event.get("input"), str)
                                    else json.dumps(_json_value(event.get("input")), separators=(",", ":"))
                                ),
                            ),
                        )
                        self._local_tool_index += 1
                        yield _chunk(model=self._model, tool_calls=[tool_call])
                        continue
                elif event_type == "tool-result":
                    self.provider_data["provider_tool_results"].append(_json_value(event))
                elif event_type == "source":
                    self.provider_data["sources"].append(_json_value(event))
                elif event_type == "text-delta":
                    yield _chunk(model=self._model, content=str(event.get("delta") or ""))
                    continue
                elif event_type == "reasoning-delta":
                    yield _chunk(model=self._model, reasoning=str(event.get("delta") or ""))
                    continue
                elif event_type == "finish":
                    finish = event.get("finishReason")
                    unified = finish.get("unified") if isinstance(finish, dict) else finish
                    mapped = {
                        "content-filter": "content_filter",
                        "tool-calls": "tool_calls",
                    }.get(str(unified), str(unified or "stop"))
                    self.provider_data["provider_metadata"] = _json_value(
                        event.get("providerMetadata") or {}
                    )
                    self.provider_data["raw_usage"] = _json_value(event.get("usage") or {})
                    usage = _usage_object(event.get("usage"), self.provider_data)
                    yield _chunk(
                        model=self._model,
                        finish_reason=mapped,
                        usage=usage,
                    )
                    continue

                # Metadata/provider-tool events are still emitted as empty
                # chunks so Hermes' stream liveness watchdog sees activity.
                yield _chunk(model=self._model)
        finally:
            self.close()


class _Completions:
    def __init__(self, client: "VercelAIGatewayClient") -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        model, body, stream, timeout = _as_v4_request(kwargs)
        headers = self._client._headers(model=model, stream=stream)
        if stream:
            return VercelV4Stream(
                http_client=self._client._http,
                endpoint=self._client._endpoint,
                headers=headers,
                body=body,
                model=model,
                timeout=timeout or self._client._timeout,
            )
        response = self._client._http.post(
            self._client._endpoint,
            headers=headers,
            json=body,
            timeout=timeout or self._client._timeout,
        )
        if response.status_code >= 400:
            raise _response_error(response)
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise VercelAIGatewayProtocolError(
                "Vercel v4 response was not valid JSON",
                status_code=response.status_code,
                response=response,
            ) from exc
        if not isinstance(data, dict) or not isinstance(data.get("content"), list):
            raise VercelAIGatewayProtocolError(
                "Vercel v4 response is missing the content array",
                status_code=response.status_code,
                body=data,
                response=response,
            )
        return data


class VercelAIGatewayClient:
    """Small synchronous client managed by Hermes like its OpenAI clients."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://ai-gateway.vercel.sh/v1",
        default_headers: dict[str, str] | None = None,
        timeout: Any = None,
        http_client: httpx.Client | None = None,
        **_: Any,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Vercel AI Gateway API key is required")
        self.api_key = api_key.strip()
        self.base_url = base_url
        self._endpoint = _endpoint(base_url)
        self._default_headers = dict(default_headers or {})
        self._timeout = timeout or httpx.Timeout(1800.0, connect=30.0)
        self._owns_http = http_client is None
        self._http = http_client or httpx.Client()
        self.chat = SimpleNamespace(completions=_Completions(self))
        self.is_closed = False

    def _headers(self, *, model: str, stream: bool) -> dict[str, str]:
        headers = dict(self._default_headers)
        headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            "ai-gateway-auth-method": "api-key",
            "ai-gateway-protocol-version": GATEWAY_PROTOCOL_VERSION,
            "ai-language-model-id": model,
            "ai-language-model-specification-version": LANGUAGE_MODEL_SPECIFICATION_VERSION,
            "ai-language-model-streaming": "true" if stream else "false",
            "User-Agent": "hermes-agent/vercel-ai-gateway-v4",
        })
        return headers

    def close(self) -> None:
        if self.is_closed:
            return
        self.is_closed = True
        # A keepalive client injected by Hermes is request-owned just like the
        # one injected into OpenAI(). Close it from the owning worker thread.
        self._http.close()

    def __enter__(self) -> "VercelAIGatewayClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
