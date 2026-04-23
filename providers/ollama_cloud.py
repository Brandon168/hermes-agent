"""Ollama Cloud provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

ollama_cloud = ProviderProfile(
    name="ollama-cloud",
    aliases=("ollama_cloud",),
    env_vars=("OLLAMA_CLOUD_API_KEY",),
    base_url="https://ollama.com/v1",
)

register_provider(ollama_cloud)
