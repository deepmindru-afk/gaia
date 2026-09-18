import os
import signal

from latitude_telemetry import Latitude
from latitude_telemetry.env import env as latitude_env
from latitude_telemetry.env.env import get_exporter_url
from openinference.instrumentation.langchain import LangChainInstrumentor

from app.config.settings import settings
from app.core.lazy_loader import MissingKeyStrategy, lazy_provider
from shared.py.wide_events import log

_client: Latitude | None = None


def resolve_exporter_endpoint() -> str:
    """Point the Latitude SDK at the settings-configured ingest.

    Settings wins over process env, always: the SDK freezes EXPORTER_URL at
    import time (latitude_telemetry.env builds Env on import, before this
    runs), so assigning only when unset would silently keep pointing at
    Latitude Cloud while settings names the self-hosted ingest. Rebind the
    frozen value after assigning so both agree. Split out for direct tests —
    this is the line that decides self-hosted vs Cloud.
    """
    os.environ["LATITUDE_TELEMETRY_URL"] = settings.LATITUDE_TELEMETRY_URL
    latitude_env.EXPORTER_URL = get_exporter_url()
    return os.environ["LATITUDE_TELEMETRY_URL"]


@lazy_provider(
    name="latitude",
    required_keys=[settings.LATITUDE_API_KEY],
    auto_initialize=True,
    is_global_context=True,
    strategy=MissingKeyStrategy.SILENT,
)
def init_latitude() -> bool:
    """Initialize Latitude telemetry once per process.

    Spans come from OpenInference's LangChain instrumentor (maintained against
    LangChain 1.x — the OTel-contrib one latitude-telemetry bundles no longer
    patches current internals, verified: zero LLM spans), registered against
    Latitude's provider. When no global provider exists Latitude owns it;
    when one does (e.g. Sentry init in prod) the SDK piggybacks on it and
    spans traverse that pipeline too — dev/prod differ here by construction.
    No-op when LATITUDE_API_KEY is unset. Idempotent: a second call returns
    True without attaching a duplicate processor.
    """
    global _client
    if _client is not None:
        return True
    # Settings wins over process env, always: see resolve_exporter_endpoint.
    endpoint = resolve_exporter_endpoint()

    try:
        prev_sigterm = signal.getsignal(signal.SIGTERM)
        prev_sigint = signal.getsignal(signal.SIGINT)
        signals_saved = True
    except ValueError:
        # Non-main thread: no signal handlers to save/restore.
        signals_saved = False
    try:
        latitude = Latitude(
            api_key=settings.LATITUDE_API_KEY or "",
            project=settings.LATITUDE_PROJECT,
        )
    finally:
        # The SDK hijacks SIGTERM/SIGINT process-wide (sys.exit instead of
        # uvicorn's drain path, so unified_shutdown never runs). Restore the
        # host's handlers; the SDK's atexit flush still runs.
        if signals_saved:
            try:
                signal.signal(signal.SIGTERM, prev_sigterm)
                signal.signal(signal.SIGINT, prev_sigint)
            except ValueError:
                pass
    LangChainInstrumentor().instrument(tracer_provider=latitude.provider)
    _client = latitude
    log.info(
        "latitude_ready",
        project=settings.LATITUDE_PROJECT,
        endpoint=endpoint,
    )
    return True


async def flush_latitude() -> None:
    """Flush queued Latitude spans on shutdown so a restart loses no traces."""
    if _client is None:
        return
    try:
        _client.flush()
    except Exception as exc:
        log.warning(
            "latitude_flush_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
