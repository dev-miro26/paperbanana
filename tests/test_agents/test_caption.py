from __future__ import annotations

from paperbanana.agents.caption import CaptionAgent


def test_strip_response_plain():
    assert CaptionAgent._strip_response("  Hello world.  ") == "Hello world."


def test_strip_response_fenced():
    raw = "```\nLine one. Line two.\n```"
    out = CaptionAgent._strip_response(raw)
    assert "Line one" in out
    assert "```" not in out
