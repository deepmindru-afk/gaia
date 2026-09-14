"""Unit tests for LLMTtftCallback — true provider time-to-first-token."""

import time
from uuid import uuid4

from langchain_core.outputs import LLMResult
from prometheus_client import REGISTRY

from app.agents.llm.ttft import LLMTtftCallback


def _count(model: str, lane: str, agent: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "llm_ttft_seconds_count", {"model": model, "lane": lane, "agent": agent}
        )
        or 0.0
    )


def _sum(model: str, lane: str, agent: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "llm_ttft_seconds_sum", {"model": model, "lane": lane, "agent": agent}
        )
        or 0.0
    )


def test_first_token_observes_ttft_once() -> None:
    cb = LLMTtftCallback(agent="ttft-test-agent")
    run_id = uuid4()
    before = _count("ttft-model-a", "ttft-lane-a", "ttft-test-agent")
    sum_before = _sum("ttft-model-a", "ttft-lane-a", "ttft-test-agent")
    cb.on_chat_model_start(
        {},
        [],
        run_id=run_id,
        metadata={"lane_model": "ttft-model-a", "lane_provider": "ttft-lane-a"},
    )
    time.sleep(0.05)
    cb.on_llm_new_token("hello", run_id=run_id)
    assert _count("ttft-model-a", "ttft-lane-a", "ttft-test-agent") == before + 1
    assert _sum("ttft-model-a", "ttft-lane-a", "ttft-test-agent") - sum_before >= 0.05
    # Second token of the same run is not a new TTFT sample.
    cb.on_llm_new_token(" world", run_id=run_id)
    assert _count("ttft-model-a", "ttft-lane-a", "ttft-test-agent") == before + 1
    cb.on_llm_end(LLMResult(generations=[]), run_id=run_id)


def test_non_streaming_call_emits_no_sample() -> None:
    cb = LLMTtftCallback(agent="ttft-test-agent")
    run_id = uuid4()
    before = _count("ttft-model-b", "ttft-lane-b", "ttft-test-agent")
    cb.on_chat_model_start(
        {},
        [],
        run_id=run_id,
        metadata={"lane_model": "ttft-model-b", "lane_provider": "ttft-lane-b"},
    )
    cb.on_llm_end(LLMResult(generations=[]), run_id=run_id)
    assert _count("ttft-model-b", "ttft-lane-b", "ttft-test-agent") == before


def test_error_cleans_up_run_state() -> None:
    cb = LLMTtftCallback(agent="ttft-test-agent")
    run_id = uuid4()
    cb.on_chat_model_start(
        {},
        [],
        run_id=run_id,
        metadata={"lane_model": "ttft-model-c", "lane_provider": "ttft-lane-c"},
    )
    cb.on_llm_error(ValueError("boom"), run_id=run_id)
    # A later run reusing no state still records: the errored run left nothing behind.
    before = _count("ttft-model-c", "ttft-lane-c", "ttft-test-agent")
    run_id2 = uuid4()
    cb.on_chat_model_start(
        {},
        [],
        run_id=run_id2,
        metadata={"lane_model": "ttft-model-c", "lane_provider": "ttft-lane-c"},
    )
    cb.on_llm_new_token("tok", run_id=run_id2)
    assert _count("ttft-model-c", "ttft-lane-c", "ttft-test-agent") == before + 1
    cb.on_llm_end(LLMResult(generations=[]), run_id=run_id2)
