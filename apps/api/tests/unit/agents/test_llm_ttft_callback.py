"""Unit tests for LLMTtftCallback — true provider time-to-first-token."""

import time
from typing import Any
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import LLMResult
from prometheus_client import REGISTRY

from app.agents.llm.client import ainvoke_llm
from app.agents.llm.ttft import LLMTtftCallback
from app.constants.llm import LLM_LABEL_METADATA_KEY


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
    cb = LLMTtftCallback()
    run_id = uuid4()
    before = _count("ttft-model-a", "ttft-lane-a", "ttft-test-agent")
    sum_before = _sum("ttft-model-a", "ttft-lane-a", "ttft-test-agent")
    cb.on_chat_model_start(
        {},
        [],
        run_id=run_id,
        metadata={
            "lane_model": "ttft-model-a",
            "lane_provider": "ttft-lane-a",
            LLM_LABEL_METADATA_KEY: "ttft-test-agent",
        },
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
    cb = LLMTtftCallback()
    run_id = uuid4()
    before = _count("ttft-model-b", "ttft-lane-b", "ttft-test-agent")
    cb.on_chat_model_start(
        {},
        [],
        run_id=run_id,
        metadata={
            "lane_model": "ttft-model-b",
            "lane_provider": "ttft-lane-b",
            LLM_LABEL_METADATA_KEY: "ttft-test-agent",
        },
    )
    cb.on_llm_end(LLMResult(generations=[]), run_id=run_id)
    assert _count("ttft-model-b", "ttft-lane-b", "ttft-test-agent") == before


def test_error_cleans_up_run_state() -> None:
    cb = LLMTtftCallback()
    run_id = uuid4()
    cb.on_chat_model_start(
        {},
        [],
        run_id=run_id,
        metadata={
            "lane_model": "ttft-model-c",
            "lane_provider": "ttft-lane-c",
            LLM_LABEL_METADATA_KEY: "ttft-test-agent",
        },
    )
    cb.on_llm_error(ValueError("boom"), run_id=run_id)
    # A later run reusing no state still records: the errored run left nothing behind.
    before = _count("ttft-model-c", "ttft-lane-c", "ttft-test-agent")
    run_id2 = uuid4()
    cb.on_chat_model_start(
        {},
        [],
        run_id=run_id2,
        metadata={
            "lane_model": "ttft-model-c",
            "lane_provider": "ttft-lane-c",
            LLM_LABEL_METADATA_KEY: "ttft-test-agent",
        },
    )
    cb.on_llm_new_token("tok", run_id=run_id2)
    assert _count("ttft-model-c", "ttft-lane-c", "ttft-test-agent") == before + 1
    cb.on_llm_end(LLMResult(generations=[]), run_id=run_id2)


def test_each_call_is_labelled_by_its_own_label_not_the_runs_agent() -> None:
    """One turn makes several streaming calls under one callback list — the
    user-facing comms call plus title, follow-up and memory side calls. The
    agent label must be the CALL's, or the side calls pollute the comms TTFT
    p95 and the decomposition the metric exists for is lost."""
    cb = LLMTtftCallback()
    before = _count("ttft-model-d", "ttft-lane-d", "follow_up_actions")
    run_id = uuid4()
    cb.on_chat_model_start(
        {},
        [],
        run_id=run_id,
        metadata={
            "lane_model": "ttft-model-d",
            "lane_provider": "ttft-lane-d",
            LLM_LABEL_METADATA_KEY: "follow_up_actions",
        },
    )
    cb.on_llm_new_token("tok", run_id=run_id)
    assert _count("ttft-model-d", "ttft-lane-d", "follow_up_actions") == before + 1
    cb.on_llm_end(LLMResult(generations=[]), run_id=run_id)


class _StartRecorder(BaseCallbackHandler):
    def __init__(self) -> None:
        self.metadata: dict[str, Any] | None = None

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        metadata: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        self.metadata = metadata


async def test_ainvoke_llm_stamps_its_label_onto_the_call_metadata() -> None:
    """The label every call site already passes is what reaches the callbacks —
    the one seam every provider call goes through, so no site can forget it."""
    recorder = _StartRecorder()
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    await ainvoke_llm(
        model,
        "hi",
        label="follow_up_actions",
        config={"callbacks": [recorder], "metadata": {"lane_model": "m"}},
    )
    assert recorder.metadata is not None
    assert recorder.metadata[LLM_LABEL_METADATA_KEY] == "follow_up_actions"
    assert recorder.metadata["lane_model"] == "m"
