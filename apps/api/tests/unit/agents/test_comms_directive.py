"""Parsing comms' non-text turn outcomes (SILENCE / REACT) vs an ordinary reply."""

import pytest

from app.agents.core.comms_directive import interpret_comms_output
from app.constants.comms import CommsDirectiveKind

pytestmark = pytest.mark.unit


class TestInterpretCommsOutput:
    def test_plain_text_is_a_reply(self) -> None:
        d = interpret_comms_output("Booked your 9am flight to Tokyo.")
        assert d.kind == CommsDirectiveKind.REPLY
        assert d.payload == "Booked your 9am flight to Tokyo."

    def test_silence_directive(self) -> None:
        d = interpret_comms_output("SILENCE: background calendar refresh, nothing new")
        assert d.kind == CommsDirectiveKind.SILENCE
        assert d.payload == "background calendar refresh, nothing new"

    def test_react_directive(self) -> None:
        d = interpret_comms_output("REACT: 👍")
        assert d.kind == CommsDirectiveKind.REACT
        assert d.payload == "👍"

    def test_directive_is_case_and_whitespace_tolerant(self) -> None:
        d = interpret_comms_output("  react:   ✅  ")
        assert d.kind == CommsDirectiveKind.REACT
        assert d.payload == "✅"

    def test_multiline_message_is_never_a_directive(self) -> None:
        # A real reply that merely starts with the word must not be mis-silenced —
        # the safe failure is "treat as reply", never drop a real message.
        text = "SILENCE: is golden.\nBut here is the actual answer you asked for."
        d = interpret_comms_output(text)
        assert d.kind == CommsDirectiveKind.REPLY
        assert d.payload == text

    def test_react_without_emoji_falls_back_to_reply(self) -> None:
        d = interpret_comms_output("REACT:")
        assert d.kind == CommsDirectiveKind.REPLY
