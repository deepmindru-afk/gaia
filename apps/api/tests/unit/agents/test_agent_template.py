"""Unit tests for the experiment variants in agent_template."""

from app.agents.templates.agent_template import (
    get_comms_static_prompt,
    get_executor_prompt,
)

OPENUI_MARKER = "---OpenUI Lang (Rich UI Components)---"
MARKDOWN_MARKER = "---Output Format---"


class TestOpenuiVariants:
    def test_web_on_has_openui_off_has_markdown_fallback(self) -> None:
        on = get_comms_static_prompt("web", openui_enabled=True)
        off = get_comms_static_prompt("web", openui_enabled=False)
        assert on != off
        assert OPENUI_MARKER in on
        assert MARKDOWN_MARKER not in on
        assert MARKDOWN_MARKER in off
        assert OPENUI_MARKER not in off

    def test_default_is_on_for_backwards_compat(self) -> None:
        assert get_comms_static_prompt("web") == get_comms_static_prompt(
            "web", openui_enabled=True
        )

    def test_desktop_keeps_desktop_context_in_both_variants(self) -> None:
        on = get_comms_static_prompt("desktop", openui_enabled=True)
        off = get_comms_static_prompt("desktop", openui_enabled=False)
        assert "Desktop Context" in on
        assert "Desktop Context" in off
        assert OPENUI_MARKER in on
        assert MARKDOWN_MARKER in off
        assert OPENUI_MARKER not in off

    def test_text_channels_ignore_the_flag(self) -> None:
        for source in ("whatsapp", "telegram", "discord", "slack"):
            assert get_comms_static_prompt(source, openui_enabled=True) == (
                get_comms_static_prompt(source, openui_enabled=False)
            )

    def test_unknown_source_falls_back_to_web(self) -> None:
        assert get_comms_static_prompt("nope") == get_comms_static_prompt("web")
        assert get_comms_static_prompt(None, openui_enabled=False) == (
            get_comms_static_prompt("web", openui_enabled=False)
        )


class TestExecutorPrompt:
    def test_teaches_activation(self) -> None:
        prompt = get_executor_prompt()
        assert "activate_integration" in prompt
        assert "You activate one, then do" in prompt

    def test_default_matches_env_template(self) -> None:
        from app.agents.templates.agent_template import EXECUTOR_PROMPT_TEMPLATE

        assert get_executor_prompt() == EXECUTOR_PROMPT_TEMPLATE
