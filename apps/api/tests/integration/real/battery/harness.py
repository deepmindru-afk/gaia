"""Drive one real browser task the way a Telegram user does, and read what really happened.

Nothing here is faked: the message goes through the bot harness (the real bot
pipeline emulating Telegram) into the running API, the comms and executor
agents, the ARQ worker, the browser host and the engine, against real sites.
Assertions read the run's own artifacts: the job state in Redis, the task
history in Mongo, the handoff record, the saved logins, and the transcript of
what the bot delivered (text, photos).

Requires the native dev stack (see the `driving-gaia` skill and
``~/.cache/gaia-browser/run/boot.sh``) and ``GAIA_BROWSER_BATTERY=1``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any
import uuid

import httpx
import pymongo
import redis

from app.constants.browser import (
    BROWSER_HANDOFF_CONV_KEY_PREFIX,
    BROWSER_HANDOFF_KEY_PREFIX,
    BROWSER_JOB_STATE_PREFIX,
)

API_URL = os.environ.get("GAIA_BATTERY_API_URL", "http://localhost:8480")
HOST_URL = os.environ.get("BROWSER_HOST_URL", "http://localhost:8930")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
BATTERY_USER = os.environ.get("GAIA_BATTERY_USER", "aryan@heygaia.io")
REPO_ROOT = Path(__file__).resolve().parents[6]
HARNESS_DIR = REPO_ROOT / "apps" / "bots" / "harness"

#: How long one scenario may take end to end before it is a failure in itself.
RUN_TIMEOUT_SECONDS = 720.0
_POLL_SECONDS = 2.0


def battery_enabled() -> bool:
    return os.environ.get("GAIA_BROWSER_BATTERY") == "1"


def stack_answers() -> bool:
    try:
        return httpx.get(f"{API_URL}/api/v1/todos", timeout=5).status_code == 200 and (
            httpx.get(f"{HOST_URL}/healthz", timeout=5).json().get("chromium_up") is True
        )
    except Exception:
        return False


@dataclass
class Transcript:
    """What the bot delivered for one turn, from the harness's JSONL."""

    events: list[dict[str, Any]]

    @property
    def texts(self) -> list[str]:
        return [
            str(e.get("text", ""))
            for e in self.events
            if e.get("type") in ("send", "outbound-delivery") and e.get("text")
        ]

    @property
    def photos(self) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") in ("rich", "outbound-attachment")]

    def texts_matching(self, pattern: str) -> list[str]:
        return [t for t in self.texts if re.search(pattern, t, re.I | re.S)]


@dataclass
class RunOutcome:
    """One finished (or stopped) browser run, read back from the stack."""

    job_id: str
    state: dict[str, Any]
    task_record: dict[str, Any] | None
    transcript: Transcript
    handoffs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return str(((self.state.get("result") or {}).get("summary")) or "")

    @property
    def status(self) -> str:
        return str((self.state.get("result") or {}).get("status", ""))

    @property
    def success(self) -> bool | None:
        result = self.state.get("result") or {}
        return result.get("success") if result else None

    @property
    def step_count(self) -> int:
        return int((self.task_record or {}).get("steps") or 0)


class Battery:
    """The stack, as one scenario sees it."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        self.mongo = pymongo.MongoClient(MONGO_URL)["GAIA"]
        self.api = httpx.Client(base_url=f"{API_URL}/api/v1", timeout=30)
        self.last_channel: str | None = None

    # -- sending -----------------------------------------------------------

    def send(
        self, message: str, *, settle_ms: int, channel: str | None = None
    ) -> subprocess.Popen[str]:
        """Inject one Telegram message through the real bot pipeline; returns the live process.

        Every scenario gets a channel of its own, hence a fresh conversation: in
        one conversation a repeated task is answered from the earlier result
        instead of driving the browser again, which is right for a user and
        wrong for a test. Pass the same channel to continue a conversation.
        """
        out = self.out_dir / f"{uuid.uuid4().hex[:8]}.jsonl"
        self.last_channel = channel or f"battery-{uuid.uuid4().hex[:10]}"
        cmd = [
            "infisical",
            "run",
            "--env=development",
            "--",
            "pnpm",
            "tsx",
            "src/cli.ts",
            "send",
            "--emulate",
            "telegram",
            "--user",
            BATTERY_USER,
            "--api",
            API_URL,
            "--channel",
            self.last_channel,
            "--settle",
            str(settle_ms),
            "--out",
            str(out),
            message,
        ]
        proc = subprocess.Popen(
            cmd, cwd=HARNESS_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        proc.transcript_path = out  # type: ignore[attr-defined]
        return proc

    def transcript_of(self, proc: subprocess.Popen[str]) -> Transcript:
        path: Path = proc.transcript_path  # type: ignore[attr-defined]
        events = []
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    events.append(json.loads(line))
        return Transcript(events)

    # -- reading the run ----------------------------------------------------

    def job_ids(self) -> list[str]:
        return [
            k.removeprefix(BROWSER_JOB_STATE_PREFIX)
            for k in self.redis.scan_iter(f"{BROWSER_JOB_STATE_PREFIX}*")
            if ":" not in k.removeprefix(BROWSER_JOB_STATE_PREFIX)
        ]

    def job_state(self, job_id: str) -> dict[str, Any]:
        raw = self.redis.get(f"{BROWSER_JOB_STATE_PREFIX}{job_id}")
        return json.loads(raw) if raw else {}

    def wait_for_job(
        self, known: set[str], *, timeout: float = 240.0, proc: subprocess.Popen[str] | None = None
    ) -> str:
        """Return the one job that appears after `known`: one browser per conversation, one scenario at a time."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            new = [j for j in self.job_ids() if j not in known]
            if new:
                return new[0]
            if proc is not None and proc.poll() is not None:
                break
            time.sleep(_POLL_SECONDS)
        said = self.transcript_of(proc).texts if proc is not None else []
        raise AssertionError(f"no browser job was enqueued for the message; the bot said {said}")

    def wait_for_status(self, job_id: str, statuses: set[str], *, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.job_state(job_id)
            if state.get("status") in statuses:
                return state
            time.sleep(_POLL_SECONDS)
        raise AssertionError(f"job {job_id} never reached {statuses}: {self.job_state(job_id)}")

    def task_record(self, session_id: str | None) -> dict[str, Any] | None:
        """Return the task-history row for this run's browser session (written when the run ends)."""
        if not session_id:
            return None
        return self.mongo["browser_tasks"].find_one({"session_id": session_id})

    # -- handoffs ----------------------------------------------------------

    def pending_handoff(self, conversation_id: str) -> tuple[str, dict[str, Any]] | None:
        handoff_id = self.redis.get(f"{BROWSER_HANDOFF_CONV_KEY_PREFIX}{conversation_id}")
        if not handoff_id:
            return None
        raw = self.redis.get(f"{BROWSER_HANDOFF_KEY_PREFIX}{handoff_id}")
        record = json.loads(raw) if raw else {}
        return (handoff_id, record) if record.get("status") == "pending" else None

    def wait_for_handoff(
        self, conversation_id: str, *, timeout: float = 300.0
    ) -> tuple[str, dict[str, Any]]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            found = self.pending_handoff(conversation_id)
            if found:
                return found
            time.sleep(_POLL_SECONDS)
        raise AssertionError(f"no handoff was raised for conversation {conversation_id}")

    def decide_handoff(self, handoff_id: str, decision: str, message: str | None = None) -> str:
        response = self.api.post(
            f"/browser/handoffs/{handoff_id}/decision",
            json={"decision": decision, "message": message},
        )
        response.raise_for_status()
        return str(response.json()["status"])

    # -- saved logins ------------------------------------------------------

    def saved_login_domains(self) -> list[str]:
        response = self.api.get("/browser/logins")
        response.raise_for_status()
        return [str(row.get("domain", "")) for row in response.json()]

    def forget_logins(self) -> None:
        self.api.delete("/browser/logins").raise_for_status()

    # -- the live browser during a handoff --------------------------------

    async def type_into_live_session(self, session_id: str, script: str) -> Any:
        """Run JS in the run's own page, the way a person acts in live view during a handoff."""
        from browser_use import Browser

        ws = HOST_URL.replace("http", "ws", 1) + f"/cdp/{session_id}"
        browser = Browser(cdp_url=ws)
        await browser.start()
        try:
            cdp = await browser.get_or_create_cdp_session(focus=False)
            result = await cdp.cdp_client.send.Runtime.evaluate(
                params={"expression": script, "awaitPromise": True, "returnByValue": True},
                session_id=cdp.session_id,
            )
            return result.get("result", {}).get("value")
        finally:
            await browser.stop()

    # -- one whole scenario -------------------------------------------------

    def run(self, message: str, *, settle_ms: int = 60_000, on_running=None) -> RunOutcome:
        """Send a message, follow its run to the end, and return everything it produced.

        on_running(job_id, state, battery) is called once the run is RUNNING, for
        scenarios that act mid-run (a handoff to answer, a stop to send).
        """
        known = set(self.job_ids())
        proc = self.send(message, settle_ms=settle_ms)
        try:
            job_id = self.wait_for_job(known, proc=proc)
            state = self.wait_for_status(job_id, {"running", "done"}, timeout=240.0)
            # The terminal state drops the session id; the history row is keyed on it.
            session_id = state.get("session_id")
            handoffs: list[dict[str, Any]] = []
            if on_running is not None and state.get("status") == "running":
                handoffs = on_running(job_id, state, self) or []
            state = self.wait_for_status(job_id, {"done"}, timeout=RUN_TIMEOUT_SECONDS)
        finally:
            try:
                proc.wait(timeout=settle_ms / 1000 + 60)
            except subprocess.TimeoutExpired:
                proc.kill()
        # The history row is written at the end of the run; give it a moment.
        record = None
        for _ in range(15):
            record = self.task_record(session_id)
            if record:
                break
            time.sleep(1)
        return RunOutcome(job_id, state, record, self.transcript_of(proc), handoffs)


def fetch_text(url: str) -> str:
    """Ground truth from the site itself, fetched at test time."""
    return httpx.get(
        url, timeout=30, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}
    ).text


def hn_front_page_titles() -> list[str]:
    html = fetch_text("https://news.ycombinator.com/")
    return [
        re.sub(r"<[^>]+>", "", m)
        for m in re.findall(r'<span class="titleline"><a[^>]*>(.*?)</a>', html)
    ]


def run_async(coro):
    return asyncio.run(coro)
