"""Backward-compatibility shim.

CopilotACPClient has moved to acp_adapter/copilot_client.py.
This module re-exports it so existing callers continue to work.
"""
from acp_adapter.copilot_client import CopilotACPClient  # noqa: F401

__all__ = ["CopilotACPClient"]
