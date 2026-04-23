"""OpenCode provider profiles (Zen + Go).

Both use per-model api_mode routing:
  - OpenCode Zen: Claude → anthropic_messages, GPT-5/Codex → codex_responses,
    everything else → chat_completions (this profile)
  - OpenCode Go: MiniMax → anthropic_messages, GLM/Kimi → chat_completions
    (this profile)
"""

from providers import register_provider
from providers.base import ProviderProfile

opencode_zen = ProviderProfile(
    name="opencode-zen",
    aliases=("opencode", "opencode_zen", "zen"),
    env_vars=("OPENCODE_ZEN_API_KEY",),
    base_url="https://opencode.ai/zen/v1",
)

opencode_go = ProviderProfile(
    name="opencode-go",
    aliases=("opencode_go",),
    env_vars=("OPENCODE_GO_API_KEY",),
    base_url="https://opencode.ai/zen/go/v1",
)

register_provider(opencode_zen)
register_provider(opencode_go)
