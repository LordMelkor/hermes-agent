"""Reasoning override propagation for ``hermes -z``."""

import ast
import inspect
import logging

import pytest


def test_run_oneshot_forwards_explicit_reasoning(monkeypatch):
    from hermes_cli import oneshot

    seen = {}

    def fake_run_agent(prompt, **kwargs):
        seen.update(kwargs)
        return "done", {"final_response": "done"}

    monkeypatch.setattr(oneshot, "_run_agent", fake_run_agent)

    try:
        rc = oneshot.run_oneshot(
            "review this",
            model="example-model",
            reasoning="xhigh",
        )
    finally:
        logging.disable(logging.NOTSET)

    assert rc == 0
    assert seen["reasoning"] == "xhigh"


@pytest.mark.parametrize(
    ("reasoning", "expected"),
    [
        ("xhigh", {"enabled": True, "effort": "xhigh"}),
        (None, {"enabled": True, "effort": "high"}),
    ],
)
def test_run_agent_reasoning_precedence(monkeypatch, reasoning, expected):
    from hermes_cli import config, mcp_startup, models, oneshot, runtime_provider, tools_config
    import run_agent

    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._session_messages = []

        def run_conversation(self, prompt):
            return {"final_response": "done"}

        def shutdown_memory_provider(self, *args):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {
            "model": {"default": "example-model", "provider": "example"},
            "agent": {"reasoning_effort": "high"},
        },
    )
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **kwargs: {
            "provider": "example",
            "requested_provider": "example",
            "api_key": "test-key",
            "base_url": "https://example.invalid",
            "api_mode": "openai-chat",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(models, "detect_provider_for_model", lambda *args: None)
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda *args: set())
    monkeypatch.setattr(mcp_startup, "ensure_mcp_discovery_before_agent_build", lambda **kwargs: None)
    monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.setattr(oneshot, "get_fallback_chain", lambda cfg: [])
    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)

    response, _result = oneshot._run_agent(
        "review this",
        model="example-model",
        provider="example",
        reasoning=reasoning,
        toolsets=[],
    )

    assert response == "done"
    assert captured["reasoning_config"] == expected


def test_top_level_oneshot_dispatch_forwards_reasoning(monkeypatch):
    from hermes_cli import main, oneshot

    seen = {}

    class ExitCalled(Exception):
        pass

    def fake_run_oneshot(prompt, **kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(oneshot, "run_oneshot", fake_run_oneshot)
    monkeypatch.setattr(main, "_cleanup_oneshot_runtime", lambda: None)
    monkeypatch.setattr(
        main,
        "_exit_after_oneshot",
        lambda rc: (_ for _ in ()).throw(ExitCalled(rc)),
    )

    with pytest.raises(ExitCalled):
        main._run_and_exit_oneshot("review this", reasoning="xhigh")

    assert seen["reasoning"] == "xhigh"


def test_every_oneshot_dispatch_passes_reasoning():
    from hermes_cli import main

    tree = ast.parse(inspect.getsource(main))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "_run_and_exit_oneshot"
    ]

    assert calls, "no one-shot dispatch sites found"
    missing = [
        call.lineno
        for call in calls
        if "reasoning" not in {kw.arg for kw in call.keywords}
    ]
    assert not missing, f"one-shot dispatch drops --reasoning at lines {missing}"
