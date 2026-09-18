"""Lanes: the two implementations of "decide and execute this task's steps".

The seam and its shared types live in ``base``; ``browser_use`` is the shipped
Browser-Use agent and ``ultrafast`` the ported jev-ultrafast loop. The runner
picks one from ``BrowserRunConfig.agent_loop`` and knows nothing else about them.
"""

from app.services.browser.lanes.base import (
    ActionResultsFn,
    BrowserLane,
    BrowserRunConfig,
    LaneHooks,
    LaneOutcome,
    LaneUsage,
    StepClock,
    StepFrame,
)
from app.services.browser.lanes.browser_use import BrowserUseLane
from app.services.browser.lanes.ultrafast import UltrafastLane

__all__ = [
    "ActionResultsFn",
    "BrowserLane",
    "BrowserRunConfig",
    "BrowserUseLane",
    "LaneHooks",
    "LaneOutcome",
    "LaneUsage",
    "StepClock",
    "StepFrame",
    "UltrafastLane",
]
