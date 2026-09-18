"""Browser automation capability: gaia-browser-host + an agent loop.

Layered so each concern is swappable and independently testable:

  * ``session``    — browser-host session lifecycle (create / live-view / release)
  * ``llm``        — the strong, vision-capable model that drives the agent
  * ``classify``   — LLM gate deciding which steps need human approval
  * ``handoff``    — Redis bridge a paused run blocks on
  * ``runner``     — one task's orchestration: progress, handoff, budgets, metering
  * ``agent_run``  — the Browser-Use agent loop that decides and executes the steps
  * ``bot_delivery`` — mirrors progress + screenshots to messaging bots

The agent tool (``app/agents/tools/browser_tool.py``) is the only place these
are wired together; the executor sees a single tool, not the host or Browser-Use.
"""
