# Vercel AI Gateway native web search

Hermes can attach Vercel-hosted Exa search to the primary model request through
the AI Language Model v4 protocol. Production remains Python-only.

## Configuration

Use an explicit named Vercel provider connection. The same `key_env` can still
be used by image generation and other Vercel-backed features; no credential is
copied into the web configuration.

```yaml
providers:
  vercel-example:
    api: https://ai-gateway.vercel.sh/v1
    key_env: AI_GATEWAY_API_KEY
    transport: vercel_ai_gateway
    models:
      - openai/gpt-5.6-luna

web:
  vercel_ai_gateway:
    enabled: true
    provider: exa
    exclude_web_extract: true
    zero_data_retention: true
    exa:
      type: fast
      num_results: 5
      contents:
        text:
          max_characters: 8000
        highlights:
          max_characters: 800
```

Select `custom:vercel-example/openai/gpt-5.6-luna` in Hermes or WebUI. The
runtime resolves the named provider and activates `vercel_ai_gateway` for that
session.

## Behavior

- `web_search` is replaced with `gateway.exa_search` in the model request.
- Vercel executes Exa; Hermes never calls Firecrawl, Tavily, or Exa directly.
- Provider-executed calls and results remain provider metadata, not local tool
  calls. Ordinary Hermes tools retain their normal approval and execution loop.
- `web_extract` is withheld by default. Exa result content is available to the
  model, but the Gateway tool is not an arbitrary-URL fetch API.
- Request-level Gateway ZDR is enabled by default.
- Native search is fail-closed and requires `enabled: true`; missing, malformed,
  or unreadable configuration does not activate a third-party search request.
- Provider-native results are consumed inside the model request, so Hermes
  cannot apply its post-retrieval `<untrusted_tool_result>` wrapper. Every
  native-search request carries a system-level instruction to treat result
  content as untrusted evidence, but this remains a weaker trust boundary than
  a locally executed and wrapped `web_search` result.
- Disabling the `web` toolset suppresses native search entirely.
- The transport refuses non-Vercel hosts and unknown critical stream events.
- Ambiguous HTTP failures are left to Hermes' outer retry/fallback policy; the
  client does not perform hidden SDK retries.

The transport currently pins:

- Language Model specification `4`
- Gateway protocol `0.0.1`
- Endpoint `/v4/ai/language-model`

The exact SDK oracle and regeneration procedure live in
`scripts/vercel_ai_gateway_oracle/`.

## WebUI

WebUI uses the same Gateway runtime and model resolver as the CLI. Direct
in-process `AIAgent` construction resolves the named provider's configured
transport, so WebUI cannot silently fall back to `chat_completions` while
retaining the same account name and base URL. No WebUI patch or second service
is required. Streaming v4 SSE events are projected into Hermes' existing stream
lifecycle, preserving cancellation, liveness checks, local tool calls, and
token delivery to WebUI.

## Rollback

Immediate configuration rollback:

```yaml
providers:
  vercel-example:
    transport: chat_completions
```

Code checkpoints:

- `vercel-native-search-v0-baseline`: upstream baseline before the feature
- `vercel-native-search-v1-protocol`: frozen SDK/wire fixture
- `vercel-native-search-v1-implementation`: transport and deterministic tests
- `vercel-native-search-v1-live`: deployed and live-QA-verified state

The feature is isolated behind `api_mode == "vercel_ai_gateway"`; changing the
provider transport restores the prior `/v1/chat/completions` path without
altering credentials or models.
