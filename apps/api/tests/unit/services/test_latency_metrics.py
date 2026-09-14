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


def test_llm_call_and_graph_node_observe():
    from prometheus_client import REGISTRY

    from app.services import latency_metrics as m

    m.observe_llm_call(1.5, model="test-model", agent="test-agent")
    assert (
        REGISTRY.get_sample_value(
            "llm_call_seconds_count", {"model": "test-model", "agent": "test-agent"}
        )
        == 1.0
    )
    m.observe_graph_node(0.05, node="test-node", agent="test-agent")
    assert (
        REGISTRY.get_sample_value(
            "graph_node_seconds_count", {"node": "test-node", "agent": "test-agent"}
        )
        == 1.0
    )
