"""Model-discovery probe helpers; ``hermes_cli.models.probe_api_models`` is the entry point."""

from __future__ import annotations

from typing import Optional

from agent.anthropic_endpoints import _requires_bearer_auth


def probe_auth_headers(api_key: Optional[str], api_mode: Optional[str], base_url: str) -> dict[str, str]:
    """The credential headers the route's client sends, so discovery and inference agree. An Anthropic
    Messages route always carries ``anthropic-version`` and sends the key as ``Authorization: Bearer``
    where the adapter does (``_requires_bearer_auth``), else as ``x-api-key``; other routes use Bearer."""
    if not api_key:
        return {}
    if api_mode != "anthropic_messages":
        return {"Authorization": f"Bearer {api_key}"}
    if _requires_bearer_auth(base_url):
        return {"Authorization": f"Bearer {api_key}", "anthropic-version": "2023-06-01"}
    return {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
