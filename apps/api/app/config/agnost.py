from typing import Any

from app.config.settings import settings
from app.core.lazy_loader import MissingKeyStrategy, lazy_provider
from shared.py.wide_events import log

try:
    import agnost as _agnost_module

    agnost: Any = _agnost_module
    _AGNOST_AVAILABLE = True
except ImportError:
    agnost = None
    _AGNOST_AVAILABLE = False

_initialized = False


@lazy_provider(
    name="agnost",
    required_keys=[settings.AGNOST_ORG_ID],
    auto_initialize=True,
    is_global_context=True,
    strategy=MissingKeyStrategy.SILENT,
)
def init_agnost() -> bool:
    """Initialize the Agnost conversation SDK once per process.

    No-op when AGNOST_ORG_ID is unset so local dev without keys stays quiet.
    Raises on failure so the loader retries instead of marking ready.
    """
    if not _AGNOST_AVAILABLE:
        raise RuntimeError("agnost package not installed")
    ok = agnost.init(
        settings.AGNOST_ORG_ID or "",
        endpoint=settings.AGNOST_ENDPOINT or "https://api.agnost.ai",
    )
    if ok:
        global _initialized
        _initialized = True
        log.info("agnost_ready", endpoint=settings.AGNOST_ENDPOINT)
    else:
        log.warning("agnost_init_failed", hint="turn telemetry will be dropped")
        raise RuntimeError("agnost.init returned False")
    return True


async def flush_agnost() -> None:
    """Shut down the Agnost SDK on process exit so a restart loses no turns."""
    if not _initialized:
        return
    try:
        # shutdown() flushes, then joins the worker and closes sessions —
        # flush() alone drains but leaves the thread and HTTP sessions to
        # interpreter exit (and is idempotent, as is the SDK atexit handler).
        agnost.shutdown()
    except Exception as exc:
        log.warning(
            "agnost_flush_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
