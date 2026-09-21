"""Background executor constants.

Shared key names and internal markers used by the background executor run and
its handoff to the comms agent. Centralized so the executor runner, capture,
and any future consumers reference a single source of truth.
"""

# SSE frame key carrying the executor's narrated answer for voice-mode TTS.
# Must match VOICE_TTS_KEY in apps/voice-agent/src/constants.py — the voice
# agent matches on this exact string to decide what to speak.
VOICE_TTS_KEY = "voice_tts"
# SSE frame key carrying the saved bot message's id alongside the voice answer.
# Must match MESSAGE_ID_KEY in apps/voice-agent/src/constants.py.
MESSAGE_ID_KEY = "message_id"

# User-facing error text when the executor exhausts its recursion budget
# (GraphRecursionError). Handed to comms as the error result so it's re-voiced in
# GAIA's persona instead of leaking the raw LangGraph traceback string.
EXECUTOR_STEP_LIMIT_MESSAGE = "This task hit its step limit — try breaking it into smaller pieces."

# result_type for a run that stopped on a HIL approval instead of finishing. Such
# a run has nothing to deliver and KEEPS the busy lock: its thread is checkpointed
# with pending work, so no queued task may run on it until the approval resolves.
EXECUTOR_PAUSED = "paused"

# User-facing text when a run paused for approval but its resume context could not be
# recorded, so no decision could ever restart it. Handed to comms as the error result
# rather than parking the conversation behind a lock nothing will ever release.
EXECUTOR_APPROVAL_LOST_MESSAGE = (
    "I couldn't set up the approval for that action, so I've stopped. Please try again."
)

# User-facing text when comms narration of a finished run is unavailable. The
# executor's own terminal text is never substituted: it is internal monologue,
# and on the error path can be a raw exception string.
EXECUTOR_NARRATION_FAILED_MESSAGE = (
    "I finished that task, but I couldn't write up the result. Please ask me again."
)
EXECUTOR_NARRATION_FAILED_ERROR_MESSAGE = (
    "That task didn't finish, and I couldn't write up what went wrong. Please try again."
)

# Task text for the wake-up turn queued when background-subagent work lands after
# the executor rested. Landed results arrive through the executor inbox on their
# own; the run only has to report what is new.
EXECUTOR_COLLECTION_TASK = "Background subagent results have landed in this conversation. Summarize what is new for the user."

# Dedup marker: at most one queued collection turn per conversation at a time.
# TTL is crash insurance so a lost run can't suppress wake-ups forever.
EXECUTOR_COLLECT_MARKER_PREFIX = "executor:collect_queued:"
EXECUTOR_COLLECT_MARKER_TTL = 600


# Stamped onto an injected inbox message so a later drain pass recognises it as
# already committed to the thread. This is the whole basis of the drain's
# idempotency: the thread itself is the record of what has been delivered, so no
# cursor has to be kept in sync with it.
INBOX_ENTRY_ID = "inbox_entry_id"

# What a stopped run tells the run that follows it. Carries no instruction of its
# own, which is why it never counts as work (see ``ExecutorInbox.announce_interruption``).
INTERRUPTION_NOTICE = (
    "The task you were working on was INTERRUPTED by the user. Do not "
    "resume it, retry it, or finish what it left half-done unless the "
    "user asks for it again."
)
