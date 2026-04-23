"""Provider profile base class.

A ProviderProfile declares everything about an inference provider in one place:
auth, endpoints, client quirks, request-time quirks. The transport reads this
instead of receiving 20+ boolean flags.

Provider profiles are DECLARATIVE — they describe the provider's behavior.
They do NOT own client construction, credential rotation, or streaming.
Those stay on AIAgent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Sentinel for "omit temperature entirely" (Kimi: server manages it)
OMIT_TEMPERATURE = object()


@dataclass
class ProviderProfile:
    """Base provider profile — subclass or instantiate with overrides."""

    # ── Identity ─────────────────────────────────────────────
    name: str
    api_mode: str = "chat_completions"
    aliases: tuple = ()

    # ── Auth ─────────────────────────────────────────────────
    env_vars: tuple = ()
    base_url: str = ""
    auth_type: str = (
        "api_key"  # api_key | oauth_device_code | oauth_external | copilot | aws
    )

    # ── Client-level quirks (set once at client construction) ─
    default_headers: dict[str, str] = field(default_factory=dict)

    # ── Request-level quirks ─────────────────────────────────
    # Temperature: None = use caller's default, OMIT_TEMPERATURE = don't send
    fixed_temperature: Any = None
    default_max_tokens: int | None = None

    # ── Hooks (override in subclass for complex providers) ───

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Provider-specific message preprocessing.

        Called AFTER codex field sanitization, BEFORE developer role swap.
        Default: pass-through.
        """
        return messages

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        """Provider-specific extra_body fields.

        Merged into the API kwargs extra_body. Default: empty dict.
        """
        return {}

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Provider-specific kwargs split between extra_body and top-level api_kwargs.

        Returns (extra_body_additions, top_level_kwargs).
        The transport merges extra_body_additions into extra_body, and
        top_level_kwargs directly into api_kwargs.

        This split exists because some providers put reasoning config in
        extra_body (OpenRouter: extra_body.reasoning) while others put it
        as top-level api_kwargs (Kimi: api_kwargs.reasoning_effort).

        Default: ({}, {}).
        """
        return {}, {}

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch the live model list from the provider's /v1/models endpoint.

        Returns a list of model ID strings, or None if the fetch failed or
        the provider does not support live model listing.

        The default implementation calls GET {self.base_url}/models with
        Bearer auth if api_key is provided. Override to customise the
        request, filter the response, or disable live listing entirely.

        Callers must always fall back to the static _PROVIDER_MODELS list
        when this returns None.
        """
        if not self.base_url:
            return None

        import json
        import urllib.error
        import urllib.request

        url = self.base_url.rstrip("/") + "/models"
        req = urllib.request.Request(url)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Accept", "application/json")
        for k, v in self.default_headers.items():
            req.add_header(k, v)

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            items = data if isinstance(data, list) else data.get("data", [])
            return [m["id"] for m in items if isinstance(m, dict) and "id" in m]
        except Exception as exc:
            logger.debug("fetch_models(%s): %s", self.name, exc)
            return None
