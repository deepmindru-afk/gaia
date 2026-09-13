"""Unit tests for turn latency histograms."""

from prometheus_client import REGISTRY


def test_histograms_registered_and_observe():
    from app.services import latency_metrics as m

    m.observe_chat_ttft(0.5, source="web", voice_mode=False, status="success")
    assert "chat_ttft_seconds" in REGISTRY._names_to_collectors


def test_span_yields_elapsed_seconds():
    from app.services import latency_metrics as m

    with m.span() as elapsed:
        pass
    assert elapsed() >= 0.0
