"""The executor prompt doctrine: integrations are activated in-context.

The executor's doctrine: there is no handoff and no per-integration subagent, so every passage that teaches "hand this to the
gmail subagent" has to teach "activate gmail, then do the work yourself".

This rewrites those passages rather than forking the prompt. Two copies of a
450-line prompt drift, and the copy people actually tune would be the other one,
so the executor under the experiment would quietly fall behind on every prompt
fix that lands.

Every rewrite is anchored to text in EXECUTOR_AGENT_PROMPT. If an anchor stops
matching because that prompt was edited, that one rewrite is skipped with a
warning and the rest still apply: this builder runs at import time, so raising
would keep the whole API from starting over one edited sentence. The anchor
tests pin every rewrite, so a skipped one still fails loudly in CI.
"""

from app.agents.prompts.comms_prompts import EXECUTOR_AGENT_PROMPT
from shared.py.wide_events import log

_DELEGATION_MODEL = """DELEGATION MODEL

Integrations are not separate agents you hand work to. You activate one, then do
the work yourself with its tools in your own hands.

activate_integration(integration_id) loads an integration into THIS conversation:
its most-used tools arrive as schemas in the reply (run them via execute, never
by name), its helpers bind immediately, the rest become retrievable, and its
operating notes and the user's standing preferences for it land in your context,
and its skills become readable. No second
agent, no separate context window, no cold start. You keep everything you have
already gathered this turn, which is exactly what handing work to a subagent used
to throw away.

The flow is always the same:
  1. activate_integration(integration_id="gmail")
  2. run the preloaded tools through execute(task_description=..., tool_name=...,
     data=...) built from the schemas in the reply; call bound helpers directly
  3. retrieve_tools (which searches the active integration too) for anything
     else, then execute those the same way

Activate once per integration per turn. A second activation of the same one is
wasted work: its tools are already preloaded and retrievable and its notes are already in your
context. Activating several DIFFERENT integrations in a turn is normal and cheap,
so when a task spans gmail and calendar, activate both up front rather than
discovering the second one halfway through.

If the integration is not connected, activation returns the connect prompt and
shows the user a connect card. Relay that and stop. Do not try to route around it.

- Third-party work (gmail, googlecalendar, notion, slack, linear, github, etc.): activate, then act.
- Unknown integration ids: discover first with retrieve_tools.
- CONNECTED INTEGRATIONS LIST: your context carries a live "CONNECTED INTEGRATIONS" block listing the user's currently connected accounts, each with its integration_id in parentheses. Trust it over retrieve_tools for what is connected this turn. If the user asks for an integration that is NOT listed, STILL call activate_integration on it: that call is what renders the connect card. Telling the user to connect without calling it leaves them hunting for a button nobody rendered. Built-in integrations (reminders, todos, gaia_knowledge_guide, docgen) are always available and are not listed.

Per-user integrations (custom MCP connections, and any integration whose tools are issued per user) cannot be pulled in-context. When you call activate_integration on one, it tells you to delegate with handoff(subagent_id="<id>", task=...) instead, which runs it in its own per-user graph and hands back the result. That is the ONLY thing handoff is for in this mode; every other integration you activate and act on yourself.

"""

_WORKING_CONTRACT = """Working an activated integration
- Hold the user's objective as-is. Do not narrow it into your own smaller script.
- Finish one integration's whole objective before moving to the next, so related items batch into one pass instead of scattering.
- NEVER use one integration's tools to do another's work (do not try to read Gmail with Slack tools). Activate the right one instead.
- The notes activation returns encode the user's standing preferences for that integration. They beat your defaults; read them before acting.

spawn_subagent (context isolation)
- A spawn is a fresh worker with no memory of this conversation. It inherits the tools you have bound (the helpers activation bound, not the preloaded schemas). A spawn that needs an integration tool either needs its schema pasted into its task text or must re-discover it itself with retrieve_tools, which searches your active integrations.
- Spawns run ONE AT A TIME, not side by side. Issuing several does not make them finish sooner, it just adds turns. Spawning is for keeping bulky intermediate work out of your context, never for speed.
- Use it when a step produces far more output than its answer is worth: mining a large file, extracting from a long document, scanning many items to report a few.
- It returns once, and only what it returns survives. Put everything it needs in the task text, and require it to hand back every finding, id, and path.
- Do NOT spawn for a call you could make yourself. A spawn costs a whole model turn; a direct tool call does not.
- Default to acting yourself with the activated tools via execute. Reach for a spawn when the output would bury you, not by habit.

"""

#: (anchor, replacement). Anchors are the smallest distinctive slice of the
#: passage, so ordinary edits elsewhere in the prompt do not break the swap.
_PHRASE_REWRITES: tuple[tuple[str, str], ...] = (
    (
        "TODO (handoff to subagent:todos)",
        'TODO (activate_integration("todos"), then act)',
    ),
    (
        "(create_tracked_todo, a direct tool, no handoff)",
        "(create_tracked_todo, a direct tool, no activation)",
    ),
    (
        "3. act on them yourself or delegate (handoff/spawn_subagent)",
        "3. act on them yourself (bound tools by name, integration tools via execute), "
        "or spawn_subagent to parallelise independent chunks",
    ),
    (
        "→ handoff directly to subagent:gaia_knowledge_guide. Always available, "
        "no retrieve_tools needed.",
        '→ activate_integration("gaia_knowledge_guide"), then answer with its tools. '
        "Always available.",
    ),
    (
        "→ handoff to subagent:docgen. Always available, no retrieve_tools needed.",
        '→ activate_integration("docgen"), then use its tools. Always available.',
    ),
    (
        "(Google Docs/Sheets/Slides, Notion → their own subagents)",
        "(Google Docs/Sheets/Slides, Notion → activate those integrations)",
    ),
    (
        "the task just needs a `read`/`write`/`edit`, a handoff, or another tool",
        "the task just needs a `read`/`write`/`edit`, an activated integration's tool, "
        "or another tool",
    ),
    (
        "- Use these directly (not handoff):",
        "- Use these directly (no activation needed):",
    ),
    # Discovery vocabulary, not delegation vocabulary — but just as load-bearing.
    # Left unrewritten, these send the model hunting for a `subagent:` entry that
    # no longer exists in this mode; retrieve_tools returns none and it falls back
    # to fetch_webpages. Observed live: a Hacker News request scraped the web
    # instead of activating the integration.
    (
        "there is almost always a dedicated tool or subagent (e.g. subagent:hackernews, "
        "fetch_webpages, web_search_tool) that is better than hand-rolling it. Do NOT curl "
        "an API or scrape a site in bash when a tool/subagent covers it.",
        "there is almost always a dedicated INTEGRATION or tool (e.g. the hackernews "
        "integration, fetch_webpages, web_search_tool) that is better than hand-rolling it. "
        "When retrieve_tools surfaces an integration for the source, activate_integration it "
        "and use its tools: that beats scraping the same data off the public web, which is "
        "the mistake this rule exists to stop. Do NOT curl an API or scrape a site in bash "
        "when an integration or tool covers it.",
    ),
    (
        'The mistake is querying "fetch webpage content" for a Hacker News request and '
        "missing subagent:hackernews.",
        'The mistake is querying "fetch webpage content" for a Hacker News request and '
        "missing the hackernews integration.",
    ),
    (
        "NEVER route a reminder\n     to subagent:todos.",
        'NEVER route a reminder\n     to the "todos" integration.',
    ),
    (
        "never name a provider in a task you send to subagent:todos.",
        'never name a provider in work you do with the "todos" integration.',
    ),
    (
        "subagent:todos is GAIA's list and nothing else.",
        'The "todos" integration is GAIA\'s list and nothing else.',
    ),
    (
        "each provider subagent reloads its whole toolset (~20s of pure overhead) and usually repeats the same outcome",
        "each spawn repeats the same setup cost and usually repeats the same outcome",
    ),
    (
        "Spawning the same provider subagent more than once for a single request is almost always a mistake",
        "Spawning the same worker more than once for a single request is almost always a mistake",
    ),
    (
        "hand off to THAT provider's subagent instead, and",
        "activate THAT provider's integration instead, and",
    ),
    (
        "Emails ALWAYS go through the draft flow (the gmail subagent drafts → user confirms → then send), never compose-and-send in one shot.",
        "Emails ALWAYS go through the draft flow (draft with the gmail integration's tools → user confirms → then send), never compose-and-send in one shot.",
    ),
)

#: (start marker, end marker, replacement). The end marker is the heading that
#: follows the section and is preserved.
_SECTION_REWRITES: tuple[tuple[str, str, str], ...] = (
    (
        "DELEGATION MODEL",
        "RESEARCH EFFORT LADDER",
        _DELEGATION_MODEL,
    ),
    (
        "Handoff contract (strict)",
        "YOUR OUTPUT (INTERNAL",
        _WORKING_CONTRACT,
    ),
)


class ActivationPromptAnchorError(RuntimeError):
    """An anchor no longer matches EXECUTOR_AGENT_PROMPT, so the rewrite is stale."""


def _replace_section(prompt: str, start: str, end: str, replacement: str) -> str:
    start_idx = prompt.find(start)
    if start_idx == -1:
        raise ActivationPromptAnchorError(f"section start {start!r} not found")
    end_idx = prompt.find(end, start_idx + len(start))
    if end_idx == -1:
        raise ActivationPromptAnchorError(f"section end {end!r} not found after {start!r}")
    return prompt[:start_idx] + replacement + prompt[end_idx:]


def build_activation_executor_prompt() -> str:
    """EXECUTOR_AGENT_PROMPT rewritten to teach activation instead of handoff.

    A stale anchor degrades instead of raising: the one rewrite is skipped
    with a warning and the rest still apply. Raising here would run at import
    time (agent_template builds _EXECUTOR_BASE on import), so one edited
    sentence in the source prompt would keep the whole API from starting. A
    skipped rewrite leaves a single handoff-era passage behind, which the
    anchor tests flag in CI — a missing prompt nuance, never an outage.
    """
    prompt = EXECUTOR_AGENT_PROMPT
    for start, end, replacement in _SECTION_REWRITES:
        try:
            prompt = _replace_section(prompt, start, end, replacement)
        except ActivationPromptAnchorError as e:
            log.warning("activation_prompt.stale_section_anchor_skipped", error=str(e))
    for anchor, replacement in _PHRASE_REWRITES:
        if anchor not in prompt:
            log.warning("activation_prompt.stale_phrase_anchor_skipped", anchor=anchor[:80])
            continue
        prompt = prompt.replace(anchor, replacement)
    return prompt
