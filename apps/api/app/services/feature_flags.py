"""Per-user feature flags backed by PostHog, with env defaults.

Why PostHog instead of Infisical: env/Infisical values are read once at boot
(``get_settings`` is ``lru_cache``d), so changing one needs a redeploy and it
applies to every user at once. PostHog evaluates per ``distinct_id`` at
request time, so targeting and rollouts change from the dashboard with no
deploy. The settings value stays as the default and kill-switch: when PostHog
is unreachable or unconfigured, evaluation fails open to it.

Every call evaluates live — deliberately no cache. A dashboard flip applies
on the very next turn, and PostHog's own ``$feature_flag_called`` stays a
complete exposure record. The cost is one ``/decide`` call per evaluation,
run in a worker thread (the SDK is sync) so the event loop never blocks.

The backend owns flags only. Experiments (control/test arms, goals,
significance) are built in the PostHog dashboard on top of these flags —
there is no experiment concept in this module.

This module is the single source for flags: the ``FeatureFlag`` members name
the PostHog flag keys, ``_default`` owns the env fallback for each, and
``is_enabled`` is the only evaluator. Call sites never touch PostHog or
``settings.ENABLE_*`` directly for a registered flag.
"""

import asyncio
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from app.config.settings import settings
from app.core.lazy_loader import providers
from app.services.analytics_service import AnalyticsEvents, capture_event
from shared.py.wide_events import log


class FeatureFlag(StrEnum):
    """PostHog flag keys GAIA evaluates. The member name is the code handle;
    the value is the flag key in the PostHog dashboard."""

    COMMS_OPENUI = "COMMS_OPENUI"
    CODE_MODE = "CODE_MODE"
    HIL_LEDGER = "HIL_LEDGER"
    HIL_JEV_JUDGE = "HIL_JEV_JUDGE"


# Human description per flag, kept next to the key so the dashboard setup and
# the code cannot drift apart.
FEATURE_FLAG_DESCRIPTIONS: dict[FeatureFlag, str] = {
    FeatureFlag.COMMS_OPENUI: (
        "Include the OpenUI component reference in the comms prompt on "
        "renderable channels; off serves the markdown fallback."
    ),
    FeatureFlag.CODE_MODE: (
        "Bash runs seed the `gaia.execute` client and mint a per-invocation "
        "token; off runs bash with no GAIA_EXECUTE_* env."
    ),
    FeatureFlag.HIL_LEDGER: (
        "Gated calls register PENDING in the approval ledger and return "
        "instead of parking the run; off keeps the interrupt barrier."
    ),
    FeatureFlag.HIL_JEV_JUDGE: (
        "Auto mode classifies with the JEV choice judge first, falling back "
        "to the LLM intent judge on transport failure; off keeps the LLM judge."
    ),
}


def _default(flag: FeatureFlag) -> bool:
    """Env default and kill-switch for ``flag``, read at call time so tests
    can override ``settings``."""
    match flag:
        case FeatureFlag.COMMS_OPENUI:
            return bool(settings.ENABLE_COMMS_OPENUI)
        case FeatureFlag.CODE_MODE:
            return bool(settings.ENABLE_CODE_MODE)
        case FeatureFlag.HIL_LEDGER:
            return bool(settings.ENABLE_HIL_LEDGER)
        case FeatureFlag.HIL_JEV_JUDGE:
            return bool(settings.ENABLE_HIL_JEV_JUDGE)


def _coerce_result(result: Any, default: bool) -> bool:  # noqa: ANN401 -- posthog SDK returns untyped flag values; validated here
    """Interpret a PostHog flag value. ``None`` means unevaluated (no
    targeting matched, error upstream) so it falls back to the default. Any
    non-control variant string counts as enabled."""
    if result is None:
        return default
    if isinstance(result, bool):
        return result
    if isinstance(result, str):
        return result.strip().lower() not in ("", "false", "off", "disabled", "control")
    return bool(result)


def _get_posthog_client() -> Any | None:  # noqa: ANN401 -- posthog SDK is untyped at the boundary
    """Return the shared PostHog client, or ``None`` when unconfigured."""
    try:
        if not providers.is_available("posthog"):
            log.debug("PostHog client not available, flag falls back to default")
            return None
        return providers.get("posthog")
    except Exception as e:
        log.debug(
            "PostHog provider lookup failed, flag falls back to default",
            error=str(e),
            error_type=type(e).__name__,
        )
        return None


async def is_enabled(
    flag: FeatureFlag, user_id: str | None, default: bool | None = None
) -> bool:
    """Evaluate ``flag`` for ``user_id``, live on every call. No user means
    no evaluation: the default applies with no I/O. The PostHog SDK call is
    sync so it runs in a worker thread. Any failure fails open to the
    default — a flag fetch must never fail a turn.

    Successful evaluations need no event from us: the SDK auto-emits
    ``$feature_flag_called`` (``send_feature_flag_events`` defaults true).
    We emit ``feature_flag:evaluated`` only for the paths that never touch
    the SDK — unconfigured client, evaluation error, unevaluated flag — so
    fallback users are counted instead of silently missing from the
    denominator.
    """
    fallback = _default(flag) if default is None else default
    if not user_id:
        return fallback

    client = _get_posthog_client()
    if client is None:
        _track_evaluation(user_id, flag, fallback, fallback_reason="posthog_unconfigured")
        return fallback

    try:
        result = await asyncio.to_thread(client.get_feature_flag, flag.value, user_id)
    except Exception as e:
        log.warning(
            "Feature flag evaluation failed, falling back to default",
            flag=flag.value,
            error=str(e),
            error_type=type(e).__name__,
        )
        _track_evaluation(user_id, flag, fallback, fallback_reason="evaluation_error")
        return fallback

    evaluated = result is not None
    enabled = _coerce_result(result, fallback)
    log.set(flags={flag.value: enabled})
    if not evaluated:
        _track_evaluation(user_id, flag, enabled, fallback_reason="flag_unevaluated")
    return enabled


def _track_evaluation(
    user_id: str, flag: FeatureFlag, enabled: bool, fallback_reason: str | None
) -> None:
    """Emit one fallback event per user/flag/day. Called only for paths the
    SDK never sees, so this never duplicates ``$feature_flag_called``.
    Best-effort and enqueue-only: telemetry must never break or slow a turn,
    so any failure is swallowed with a debug line. The per-day dedupe key
    makes repeats collapse instead of double-counting."""
    try:
        capture_event(
            user_id,
            AnalyticsEvents.FEATURE_FLAG_EVALUATED,
            {
                "flag": flag.value,
                "enabled": enabled,
                **({"fallback_reason": fallback_reason} if fallback_reason else {}),
            },
            dedupe_key=(
                f"feature-flag-evaluated:{flag.value}:{user_id}:"
                f"{datetime.now(UTC).date().isoformat()}"
            ),
        )
    except Exception as e:
        log.debug(
            "Feature flag evaluation event skipped",
            flag=flag.value,
            error=str(e),
            error_type=type(e).__name__,
        )


async def is_comms_openui_enabled(user_id: str | None) -> bool:
    """Whether ``user_id`` gets the full OpenUI component reference in the
    comms prompt on renderable channels. Off serves the markdown fallback."""
    return await is_enabled(FeatureFlag.COMMS_OPENUI, user_id)


async def is_code_mode_enabled(user_id: str | None) -> bool:
    """Whether ``user_id``'s bash runs get the ``gaia.execute`` client and a
    per-invocation token. Off runs bash with no GAIA_EXECUTE_* env."""
    return await is_enabled(FeatureFlag.CODE_MODE, user_id)


async def is_hil_ledger_enabled(user_id: str | None) -> bool:
    """Whether ``user_id``'s gated calls register PENDING in the approval
    ledger and return instead of parking the run. Off keeps the barrier."""
    return await is_enabled(FeatureFlag.HIL_LEDGER, user_id)


async def is_jev_judge_enabled(user_id: str | None) -> bool:
    """Whether ``user_id``'s auto mode classifies with JEV first (LLM fallback
    on transport failure). Off keeps the LLM intent judge for every decision."""
    return await is_enabled(FeatureFlag.HIL_JEV_JUDGE, user_id)
