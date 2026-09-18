"""Agnost turn events. Never raises; missing org id is a silent no-op."""

import agnost
from agnost import Interaction

from app.config.settings import settings
from app.constants.agents import COMMS_AGENT_NAME
from shared.py.wide_events import log


def _configured() -> bool:
    return bool((settings.AGNOST_ORG_ID or "").strip())


_disabled_logged = False


def _log_disabled_once() -> None:
    global _disabled_logged
    if not _disabled_logged:
        _disabled_logged = True
        log.info("agnost_disabled", reason="AGNOST_ORG_ID unset; turns will not report")


def begin_turn(
    *,
    user_id: str,
    conversation_id: str,
    user_input: str,
    agent_name: str = COMMS_AGENT_NAME,
    properties: dict[str, str | bool | None] | None = None,
) -> Interaction | None:
    """Open an Agnost interaction for this turn, or None when disabled/failing."""
    if not user_id:
        return None
    if not _configured():
        _log_disabled_once()
        return None
    try:
        props = {k: v for k, v in (properties or {}).items() if v is not None}
        return agnost.begin(
            user_id=user_id,
            agent_name=agent_name,
            input=user_input,
            conversation_id=conversation_id,
            properties=props,
        )
    except Exception as exc:
        log.warning(
            "agnost_begin_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            conversation_id=conversation_id,
        )
        return None


def end_turn(
    interaction: Interaction | None,
    *,
    output: str,
    success: bool,
    properties: dict[str, str | bool | None] | None = None,
) -> None:
    """Close an Agnost interaction. No-op when interaction is None. Never raises."""
    if interaction is None:
        return
    try:
        props = {k: v for k, v in (properties or {}).items() if v is not None}
        if props:
            interaction.set_properties(props)
    except Exception as exc:
        # A properties failure must not skip the close below: an un-ended
        # interaction leaks the turn from every dashboard.
        log.warning(
            "agnost_properties_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
    try:
        interaction.end(output=output, success=success)
    except Exception as exc:
        log.warning(
            "agnost_end_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
