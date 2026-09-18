"""One fan-out for turn telemetry across Agnost, Latitude, and Laminar.

Every agent entry point (streaming chat, silent background, narrator,
executor, HIL-approval classifier) opens the three vendor scopes together
and closes them with one shared outcome, so a turn reads identically in
every dashboard: user-cancelled stays separable from failure everywhere,
errors carry the same exception, properties match.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, TypedDict, cast

from app.config.settings import settings
from app.services import agnost_service, laminar_service, latitude_service
from shared.py.wide_events import log

if TYPE_CHECKING:
    from agnost import Interaction

    from app.services.laminar_service import TurnScope
    from app.services.latitude_service import TurnCapture


class TurnOutcome(StrEnum):
    """Terminal outcome of a turn, mapped explicitly per vendor.

    Cancelled is its own outcome, not a failure: a user hitting Stop and a
    provider outage must read differently in every dashboard (and in Laminar
    signals), or "is the model getting worse or are users just impatient?"
    becomes unanswerable months from now.

    Mapping note: Laminar/Latitude end cancelled turns as OK with a
    ``cancelled`` tag; Agnost has no cancelled state so it records
    ``success=False`` with ``properties.outcome="cancelled"``. Filter on
    ``properties.outcome``, not on success alone, to split user-stops from
    outages in Agnost.
    """

    SUCCESS = "success"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TurnError(Exception):
    """Synthesized turn failure when no exception object exists.

    Yielded error frames (stream) and classifier/error-result paths have a
    user-facing message but no traceback. Wrapping it in a bare
    ``Exception``/``RuntimeError`` fabricates an error identity; this type
    names it honestly: the turn failed, the stack did not exist.
    """


class TurnHandles(TypedDict):
    """One vendor scope per telemetry backend; any may be None when disabled."""

    agnost: "Interaction | None"
    latitude: "TurnCapture | None"
    laminar: "TurnScope | None"


# Logged once per process when a turn opens zero scopes, so "telemetry is
# deliberately off" is greppable and distinct from per-backend failure
# warnings. A rotated-but-typo'd key otherwise reads as weeks of silence.
_disabled_logged = False


# Closed value sets (repo-owned): a typo compiles nowhere. Keep the tier
# members in sync with COMMS_AGENT_NAME / EXECUTOR_TIER_NAME /
# NARRATOR_TIER_NAME in app/constants/agents.py (canonical spellings for
# non-telemetry code); the Literals are what the fan-out enforces.
TurnMode = Literal["interactive", "background"]
TurnTier = Literal["comms_agent", "executor", "narrator"]


@dataclass(frozen=True)
class TurnSpec:
    """What identifies a turn across all three backends.

    A single value (instead of seven parallel arguments) so every entry
    point — streaming, silent, narrator, executor, HIL-approval — opens its
    turn the same way. ``source``/``mode``/``tier``/``env`` are owned by the
    fan-out (uniformity is the point); anything else rides in ``properties``.

    ``mode`` has deliberately NO default: every turn knows whether a user is
    waiting on it, and a default would let a future reader (or mutant) blur
    interactive and background turns into each other silently.
    """

    user_id: str
    conversation_id: str
    user_input: str
    mode: TurnMode
    source: str | None = None
    tier: TurnTier = "comms_agent"
    properties: dict[str, str | bool | None] | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("interactive", "background"):
            raise ValueError(f"invalid TurnSpec.mode: {self.mode!r}")
        if self.tier not in ("comms_agent", "executor", "narrator"):
            raise ValueError(f"invalid TurnSpec.tier: {self.tier!r}")


def begin_turn_all(spec: TurnSpec) -> TurnHandles:
    """Open all three vendor scopes. Never raises (each service guards)."""
    try:
        # Reserved keys win by application order alone — a caller key colliding
        # with one is overwritten below, so no separate filter is needed.
        # env splits shared dashboards (one org/project across dev/staging/prod).
        props = {
            **(spec.properties or {}),
            "source": (spec.source or "unknown").strip() or "unknown",
            "mode": spec.mode,
            "tier": spec.tier,
            "env": settings.ENV,
        }
        handles: TurnHandles = {
            "agnost": agnost_service.begin_turn(
                user_id=spec.user_id,
                conversation_id=spec.conversation_id,
                user_input=spec.user_input,
                agent_name=spec.tier,
                properties=props,
            ),
            "latitude": latitude_service.begin_turn(
                user_id=spec.user_id,
                conversation_id=spec.conversation_id,
                agent_name=spec.tier,
                properties=props,
            ),
            "laminar": laminar_service.begin_turn(
                user_id=spec.user_id,
                conversation_id=spec.conversation_id,
                agent_name=spec.tier,
                user_input=spec.user_input,
                properties=props,
            ),
        }
    except BaseException as exc:
        log.warning(
            "turn_telemetry_begin_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return {"agnost": None, "latitude": None, "laminar": None}
    global _disabled_logged
    if not _disabled_logged and all(v is None for v in handles.values()):
        _disabled_logged = True
        if not spec.user_id:
            log.info("turn_telemetry_no_scopes", reason="anonymous turn; no user to attribute")
        else:
            log.info("turn_telemetry_no_scopes", reason="keys unset or all begins failed")
    return handles


def end_turn_all(
    handles: TurnHandles | None,
    *,
    output: str,
    error: Exception | None = None,
    cancelled: bool = False,
) -> None:
    """Close all three scopes with one outcome. None handles is a no-op. Never raises."""
    if handles is None:
        return
    try:
        # An explicit error dominates: a turn that both errored and saw a cancel
        # flag failed — the exception is what needs debugging.
        outcome = (
            TurnOutcome.FAILED
            if error is not None
            else TurnOutcome.CANCELLED
            if cancelled
            else TurnOutcome.SUCCESS
        )
        outcome_value = outcome.value
        properties: dict[str, str | bool | None] = {
            "cancelled": outcome is TurnOutcome.CANCELLED,
            "has_error": error is not None,
            "outcome": outcome_value,
        }
        # Malformed handles (a partial dict) must not raise out of an except
        # handler and mask the turn's real error.
        agnost_scope = handles.get("agnost") if isinstance(handles, dict) else None
        latitude_scope = handles.get("latitude") if isinstance(handles, dict) else None
        laminar_scope = handles.get("laminar") if isinstance(handles, dict) else None
        agnost_service.end_turn(
            cast("Interaction | None", agnost_scope),
            output=output,
            success=outcome is TurnOutcome.SUCCESS,
            properties=properties,
        )
        latitude_service.end_turn(
            cast("TurnCapture | None", latitude_scope),
            error=error,
            cancelled=outcome is TurnOutcome.CANCELLED,
        )
        laminar_service.end_turn(
            cast("TurnScope | None", laminar_scope),
            output=output,
            error=error,
            cancelled=outcome is TurnOutcome.CANCELLED,
        )
    except BaseException as exc:
        log.warning(
            "turn_telemetry_end_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
