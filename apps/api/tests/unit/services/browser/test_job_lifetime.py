"""A browser job's lifetime: one number every clock that must outlast a run is derived from."""

import pytest

from app.config.settings import settings
from app.constants.browser import (
    BROWSER_AGENT_GUIDANCE_MAX,
    BROWSER_AGENT_GUIDANCE_TIMEOUT_SECONDS,
    BROWSER_JOB_TASK,
    MAX_HANDOFFS_PER_TASK,
)
from app.services.browser.job_lifetime import (
    browser_job_deadline_seconds,
    browser_job_ttl_seconds,
)
from app.workers.config.worker_settings import (
    ARQ_BACKSTOP_GRACE_SECONDS,
    WORKER_JOB_TIMEOUT_SECONDS,
)

pytestmark = pytest.mark.unit


def test_the_deadline_outlasts_every_window_a_run_may_legitimately_wait_in() -> None:
    longest_legitimate_run = (
        settings.BROWSER_USE_TASK_TIMEOUT_SECONDS
        + MAX_HANDOFFS_PER_TASK * settings.BROWSER_USE_HANDOFF_TIMEOUT_SECONDS
        + BROWSER_AGENT_GUIDANCE_MAX * BROWSER_AGENT_GUIDANCE_TIMEOUT_SECONDS
    )

    assert browser_job_deadline_seconds() > longest_legitimate_run
    assert browser_job_deadline_seconds() > WORKER_JOB_TIMEOUT_SECONDS


def test_the_deadline_follows_the_settings_that_bound_a_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = browser_job_deadline_seconds()
    monkeypatch.setattr(
        settings,
        "BROWSER_USE_HANDOFF_TIMEOUT_SECONDS",
        settings.BROWSER_USE_HANDOFF_TIMEOUT_SECONDS + 100,
    )
    monkeypatch.setattr(
        settings, "BROWSER_USE_TASK_TIMEOUT_SECONDS", settings.BROWSER_USE_TASK_TIMEOUT_SECONDS + 7
    )

    assert browser_job_deadline_seconds() == before + MAX_HANDOFFS_PER_TASK * 100 + 7


def test_a_jobs_state_outlives_the_longest_run_whatever_the_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert browser_job_ttl_seconds() > browser_job_deadline_seconds()
    monkeypatch.setattr(settings, "BROWSER_USE_HANDOFF_TIMEOUT_SECONDS", 10_000)

    assert browser_job_ttl_seconds() > browser_job_deadline_seconds()


def test_the_worker_cuts_a_browser_job_off_at_its_deadline() -> None:
    # Read off the module, not WorkerSettings.functions: other tests reset that registry.
    from app.worker import _run_browser_job as job

    assert job.name == BROWSER_JOB_TASK
    assert job.timeout_s == browser_job_deadline_seconds() + ARQ_BACKSTOP_GRACE_SECONDS
