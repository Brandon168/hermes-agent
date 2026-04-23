"""ZAI / GLM provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

zai = ProviderProfile(
    name="zai",
    aliases=("glm", "z-ai", "z.ai", "zhipu"),
    env_vars=("ZAI_API_KEY",),
    base_url="https://api.z.ai/api/paas/v4",
)

register_provider(zai)
