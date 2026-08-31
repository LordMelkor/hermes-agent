"""Regression tests for dynamic Anthropic model discovery.

A successful ``/v1/models`` response is the authenticated provider's catalog
and must remain authoritative. Hermes' curated list is only a fallback for
missing credentials or failed discovery; merging it into a successful response
can advertise models the account cannot select.
"""

import json
import urllib.error
from email.message import Message
from unittest.mock import patch

from hermes_cli import models as M


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_anthropic_live_catalog_is_authoritative_when_available():
    """Fallback aliases absent from /v1/models are not advertised."""
    curated = M._PROVIDER_MODELS["anthropic"]
    assert "claude-fable-5" in curated  # sanity: the alias is curated
    assert "claude-sonnet-5" in curated  # newest Sonnet alias is curated

    # Live catalog the authenticated API actually returns — no fallback aliases.
    live = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
    with patch.object(M, "_fetch_anthropic_models", return_value=live):
        result = M.provider_model_ids("anthropic")

    assert result == live


def test_anthropic_live_only_model_is_preserved():
    """Discovery accepts provider models unknown to Hermes."""
    live = ["claude-future-9-99"]
    with patch.object(M, "_fetch_anthropic_models", return_value=live):
        result = M.provider_model_ids("anthropic")

    assert result == live


def test_anthropic_falls_back_to_curated_when_live_unavailable():
    """No creds / live failure -> curated list verbatim (alias still present)."""
    with patch.object(M, "_fetch_anthropic_models", return_value=None):
        result = M.provider_model_ids("anthropic")

    assert result == list(M._PROVIDER_MODELS["anthropic"])
    assert "claude-fable-5" in result


def test_anthropic_catalog_retries_pool_key_after_auto_oauth_401():
    """A stale auto-discovered OAuth token must not suppress live discovery."""
    unauthorized = urllib.error.HTTPError(
        "https://api.anthropic.com/v1/models",
        401,
        "Unauthorized",
        Message(),
        None,
    )
    with patch(
        "agent.anthropic_adapter.resolve_anthropic_token",
        return_value="stale-oauth-token",
    ), patch(
        "agent.anthropic_adapter._is_oauth_token",
        side_effect=lambda token: token == "stale-oauth-token",
    ), patch.object(
        M,
        "_resolve_anthropic_pool_catalog_credentials",
        return_value=("valid-api-key", "https://api.anthropic.com"),
    ), patch.object(
        M,
        "_urlopen_model_catalog_request",
        side_effect=[
            unauthorized,
            _Response({"data": [{"id": "claude-account-model"}]}),
        ],
    ) as opener:
        result = M._fetch_anthropic_models()

    assert result == ["claude-account-model"]
    retry_request = opener.call_args_list[1].args[0]
    assert retry_request.get_header("X-api-key") == "valid-api-key"
