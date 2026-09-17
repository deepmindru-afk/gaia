#!/usr/bin/env python
"""Check a Vercel AI Gateway key against TypeSafe's Jev before wiring it into the browser lane.

    uv run python scripts/jev_probe.py vck_...            # one or more keys
    uv run python scripts/jev_probe.py                    # BROWSER_USE_JEV_GATEWAY_API_KEY / AI_GATEWAY_API_KEY

Sends the same evaluation request the browser policy sends (an element table and
an operation question) through ``JevGatewayClient`` and prints the decision, so a
green result here means ``BROWSER_USE_JEV_ENABLED`` will work end to end. The
gateway's own reason is printed on failure — an invalid key, or a team that still
needs a credit card on file before it serves requests.
"""

from __future__ import annotations

import asyncio
import os
import sys

from app.services.browser.jev.gateway import (
    JevChoiceQuestion,
    JevEvaluationRequest,
    JevGatewayClient,
    JevGatewayError,
)

DEFAULT_BASE_URL = "https://ai-gateway.vercel.sh/v4/ai"
DEFAULT_MODEL = "typesafe-ai/jev"

PROBE = JevEvaluationRequest(
    state={
        "page": {
            "url": "https://example.com/flights",
            "title": "Flights",
            "text": "Where to? Search",
        },
        "elements": [
            {
                "index": "1",
                "label": "Where to?",
                "role": "combobox",
                "value": "",
                "operations": ["CLICK", "TYPE_TEXT"],
            },
            {"index": "2", "label": "Search", "role": "button", "operations": ["CLICK"]},
        ],
        "recent_actions": [],
    },
    questions={
        "operation": JevChoiceQuestion(
            instructions={"goal": "Search flights to London", "rules": "Pick the next operation."},
            criteria={
                "CLICK": "click an element",
                "TYPE_TEXT": "type into a field",
                "DONE": "goal satisfied",
            },
        ),
        "type_text_target": JevChoiceQuestion(
            instructions="Pick the field to type into.",
            criteria={"1": {"element": "[1] Where to?", "current_value": ""}},
        ),
    },
)


async def probe(key: str) -> bool:
    client = JevGatewayClient(
        api_key=key,
        model=os.environ.get("BROWSER_USE_JEV_MODEL", DEFAULT_MODEL),
        base_url=os.environ.get("BROWSER_USE_JEV_GATEWAY_BASE_URL", DEFAULT_BASE_URL),
    )
    try:
        evaluation = await client.evaluate(PROBE)
    except JevGatewayError as exc:
        print(f"{key[:12]}…  FAIL  {exc}")
        return False
    finally:
        await client.aclose()
    operation = evaluation.answers["operation"]
    usage = evaluation.usage
    print(
        f"{key[:12]}…  OK    operation={operation.choice} p={operation.probabilities.get(operation.choice)}"
        f" tokens_in={usage.input_tokens if usage else '?'} latency={evaluation.latency_ms}ms"
    )
    return True


async def main(keys: list[str]) -> int:
    if not keys:
        env_key = os.environ.get("BROWSER_USE_JEV_GATEWAY_API_KEY") or os.environ.get(
            "AI_GATEWAY_API_KEY"
        )
        if not env_key:
            print("usage: jev_probe.py KEY [KEY ...]  (or set BROWSER_USE_JEV_GATEWAY_API_KEY)")
            return 2
        keys = [env_key]
    results = [await probe(key) for key in keys]
    return 0 if any(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
