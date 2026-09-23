"""The browser battery: real tasks, real sites, the whole product path, judged on outcomes.

Every test here sends one Telegram message through the real bot pipeline and
asserts on what a user would get: a correct answer, one final message, photos
along the way, a handoff when one is due, a saved login reused, an honest
failure when the task cannot be done. Nothing is mocked; each run bills real
model calls and takes minutes.

Run it against the booted native stack:

    USE_REAL_SERVICES=1 GAIA_BROWSER_BATTERY=1 mise test:one tests/integration/real/battery/test_browser_battery.py -x -rs

One scenario at a time; the stack holds one browser per conversation and the
harness emulates one Telegram user.
"""

from __future__ import annotations

from collections.abc import Iterator
import re
import time

import pytest

from tests.integration.real.battery.harness import (
    Battery,
    RunOutcome,
    battery_enabled,
    hn_front_page_titles,
    run_async,
    stack_answers,
)

pytestmark = [
    pytest.mark.battery,
    pytest.mark.skipif(
        not battery_enabled() or not stack_answers(),
        reason="requires GAIA_BROWSER_BATTERY=1 and the native stack (API :8480, host :8930)",
    ),
    pytest.mark.timeout(2400),
]

_FORM = "https://www.selenium.dev/selenium/web/web-form.html"
_INTERNET = "https://the-internet.herokuapp.com"


@pytest.fixture(scope="module")
def battery(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Battery]:
    stack = Battery(tmp_path_factory.mktemp("battery"))
    yield stack
    stack.close()


#: Lines the run itself sends while it works, none of them its outcome: step
#: captions, stall notes, and the three lines of a handoff prompt.
_PROGRESS_LINE = re.compile(
    r"^(Step \d+ ·|Still (on step|waiting)|Open the live browser:|Reply \"done\")"
)
#: One reply delivered as several messages arrives within this many seconds.
_ONE_REPLY_SECONDS = 2.0


def _one_final_message(outcome: RunOutcome) -> str:
    """Assert the run's outcome reached the user exactly once and return it.

    With the executor joined on the run, the comms agent voices the outcome in
    its own words; unjoined, the worker sends the run's own result line. Either
    way it is the one reply after the run's own progress lines: the streamed
    acknowledgement before the run starts and the step captions, stall notes and
    handoff prompt during it are not outcomes. A reply the bot delivers as two
    messages within a breath is one reply.
    """
    events = outcome.transcript.events
    started = next(
        (
            i
            for i, e in enumerate(events)
            if e.get("type") in ("outbound-delivery", "rich", "outbound-attachment")
        ),
        len(events),
    )
    texts = [
        e
        for e in events[started:]
        if e.get("type") in ("send", "outbound-delivery", "edit") and e.get("text")
    ]
    handoff_prompt = {
        i - 1 for i, e in enumerate(texts) if str(e["text"]).startswith("Open the live browser:")
    }
    finals = [
        e
        for i, e in enumerate(texts)
        if i not in handoff_prompt and not _PROGRESS_LINE.search(str(e["text"]))
    ]
    replies: list[list[dict]] = []
    for e in finals:
        at = float(e.get("t") or 0.0) / 1000
        if replies and at - float(replies[-1][-1].get("t") or 0.0) / 1000 <= _ONE_REPLY_SECONDS:
            replies[-1].append(e)
        else:
            replies.append([e])
    assert len(replies) == 1, (
        f"expected one outcome reply, got {len(replies)}: "
        f"{[[str(e['text'])[:80] for e in r] for r in replies]}"
    )
    return "\n".join(str(e["text"]) for e in replies[0])


def _no_contradiction(outcome: RunOutcome) -> None:
    """Assert the chat never told the user a finished run was still going, or a failed run succeeded."""
    assert not outcome.transcript.texts_matching(r"still going|didn't die|queued right behind"), (
        outcome.transcript.texts
    )


# --------------------------------------------------------------------------
# Ordinary tasks that must simply work
# --------------------------------------------------------------------------


def test_a_form_is_filled_and_submitted_with_every_field(battery: Battery) -> None:
    outcome = battery.run(
        f'Use the browser for this. Go to {_FORM} and fill the form: text input "Aryan", '
        'password "gaia-test-123", textarea "Testing GAIA browser automation", pick "Two" in the '
        'dropdown select, tick "Checkbox 2", choose "Radio 2", set the date to 09/22/2026, then click '
        "Submit and tell me exactly what message the page shows after submitting, and the full URL "
        "of the page you land on.",
    )

    assert outcome.success is True, outcome.summary
    assert "Received!" in outcome.summary and "Form submitted" in outcome.summary, outcome.summary
    # The submitted URL carries every value the form sent: the run reports it as read.
    url = re.search(r"submitted-form\.html\?[^\s)\"]+", outcome.summary)
    assert url, f"the landing URL with the submitted values was not reported: {outcome.summary}"
    for expected in (
        "my-text=Aryan",
        "my-password=gaia-test-123",
        "my-select=2",
        "my-date=09%2F22%2F2026",
    ):
        assert expected in url.group(0), f"{expected} missing from {url.group(0)}"
    assert url.group(0).count("my-check=on") == 2, (
        "Checkbox 2 was not ticked (the first is pre-ticked)"
    )
    assert "my-radio=on" in url.group(0), "no radio was chosen"
    assert outcome.step_count >= 6, "a nine-field form takes at least six actions"
    assert len(outcome.transcript.photos) >= 4, "step photos did not reach Telegram"
    _one_final_message(outcome)
    _no_contradiction(outcome)


def test_a_two_site_research_task_reports_every_part(battery: Battery) -> None:
    outcome = battery.run(
        "Use the browser for this whole task. Go to news.ycombinator.com, find the top 3 front-page "
        "stories that are about AI or LLMs, open each one and give me its title, its points, and a "
        'one-line summary of the linked article. Then go to en.wikipedia.org, search for "Transformer '
        '(deep learning architecture)" and tell me who introduced it and in what year. Report '
        "everything back to me at the end.",
    )

    assert outcome.success is True, outcome.summary
    assert "2017" in outcome.summary and re.search(r"Vaswani|Google", outcome.summary), (
        outcome.summary
    )
    assert len(re.findall(r"\d+ points", outcome.summary)) >= 3, outcome.summary
    titles = hn_front_page_titles()
    named = [t for t in titles if len(t) > 12 and t.lower() in outcome.summary.lower()]
    assert len(named) >= 3, f"fewer than three real front-page stories named: {named}"
    assert 8 <= outcome.step_count <= 40
    _one_final_message(outcome)
    _no_contradiction(outcome)


def test_a_whole_list_is_counted_by_scrolling_to_its_end(battery: Battery) -> None:
    truth = len(hn_front_page_titles())
    outcome = battery.run(
        "Use the browser. Go to https://news.ycombinator.com and tell me exactly how many stories are "
        "listed on the front page (scroll to the bottom and count every numbered story).",
    )

    assert outcome.success is True, outcome.summary
    numbers = [int(n) for n in re.findall(r"\b(\d{1,3})\b", outcome.summary)]
    assert truth in numbers, (
        f"front page has {truth} stories; answer said {numbers}: {outcome.summary}"
    )
    _one_final_message(outcome)


def test_a_fact_from_a_specific_list_position_is_read_not_guessed(battery: Battery) -> None:
    before = hn_front_page_titles()
    outcome = battery.run(
        "Use the browser. Go to https://news.ycombinator.com and tell me the exact title of the 3rd "
        "story on the front page, copied character for character.",
    )
    after = hn_front_page_titles()

    assert outcome.success is True, outcome.summary
    accepted = (
        {before[2].lower(), after[2].lower()} if len(before) > 2 and len(after) > 2 else set()
    )
    assert any(t in outcome.summary.lower() for t in accepted), (
        f"3rd story was {accepted}; answer: {outcome.summary}"
    )
    _one_final_message(outcome)


def test_content_that_appears_after_a_wait_is_read(battery: Battery) -> None:
    outcome = battery.run(
        f"Use the browser. Go to {_INTERNET}/dynamic_loading/2, click Start, wait for the loading to "
        "finish, and tell me exactly the text that appears.",
    )

    assert outcome.success is True, outcome.summary
    assert "Hello World!" in outcome.summary, outcome.summary
    _one_final_message(outcome)


def test_a_link_that_opens_a_new_window_is_followed(battery: Battery) -> None:
    outcome = battery.run(
        f'Use the browser. Go to {_INTERNET}/windows, click the "Click Here" link, and tell me '
        "exactly the heading text on the page that opens.",
    )

    assert outcome.success is True, outcome.summary
    assert "New Window" in outcome.summary, outcome.summary
    _one_final_message(outcome)


def test_a_search_box_is_typed_into_and_the_first_result_reported(battery: Battery) -> None:
    outcome = battery.run(
        'Use the browser. Go to https://duckduckgo.com, search for "Attention Is All You Need paper", '
        "and tell me the title and URL of the first result.",
    )

    assert outcome.success is True, outcome.summary
    assert re.search(r"https?://\S+", outcome.summary), outcome.summary
    assert re.search(r"attention", outcome.summary, re.I), outcome.summary
    _one_final_message(outcome)


# --------------------------------------------------------------------------
# Handoff and saved logins
# --------------------------------------------------------------------------

_LOGIN_JS = """(async () => {
  document.querySelector('#username').value = 'tomsmith';
  document.querySelector('#password').value = 'SuperSecretPassword!';
  document.querySelector('#login').submit();
  await new Promise(r => setTimeout(r, 3000));
  return location.href;
})()"""


def test_a_login_is_handed_to_the_user_then_reused_without_a_second_handoff(
    battery: Battery,
) -> None:
    """One scenario, two runs: the second depends on what the first saved, so it is not a test of its own."""
    battery.forget_logins()

    def act(job_id: str, state: dict, b: Battery) -> list[dict]:
        handoff_id, record = b.wait_for_handoff(b.conversation_id())
        assert re.search(r"password|sign in|log in", record.get("reason", ""), re.I), record
        landed = run_async(b.type_into_live_session(state["session_id"], _LOGIN_JS))
        assert "/secure" in str(landed), f"the live login did not land on the secure area: {landed}"
        assert b.decide_handoff(handoff_id, "continue") == "completed"
        return [record]

    outcome = battery.run(
        f"Use the browser. Go to {_INTERNET}/login and log me in; I'll type my username and "
        "password myself. Then tell me the exact flash message shown after logging in.",
        on_running=act,
    )

    assert outcome.handoffs, "no handoff record was captured"
    assert outcome.success is True, outcome.summary
    assert "You logged into a secure area!" in outcome.summary, outcome.summary
    assert outcome.transcript.texts_matching(r"Open the live browser: https?://"), (
        "no live-view link reached the user"
    )
    assert any(d.endswith("the-internet.herokuapp.com") for d in battery.saved_login_domains()), (
        battery.saved_login_domains()
    )
    _one_final_message(outcome)

    seen_handoff = {"raised": False}

    def watch(job_id: str, state: dict, b: Battery) -> list[dict]:
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline and b.job_state(job_id).get("status") != "done":
            found = b.pending_handoff(b.conversation_id())
            if found:
                seen_handoff["raised"] = True
                b.decide_handoff(found[0], "cancel")
            time.sleep(3)
        return []

    reuse = battery.run(
        f"Use the browser. Go to {_INTERNET}/secure and tell me the exact heading and the flash message on the page.",
        on_running=watch,
    )

    assert not seen_handoff["raised"], (
        "the saved login was not reused: the run asked the user to log in again"
    )
    assert reuse.success is True, reuse.summary
    assert "Secure Area" in reuse.summary, reuse.summary
    _one_final_message(reuse)


def test_a_login_with_credentials_given_in_the_message_needs_no_handoff(battery: Battery) -> None:
    battery.forget_logins()
    outcome = battery.run(
        f"Use the browser. Go to {_INTERNET}/login, log in as tomsmith with password "
        "SuperSecretPassword!, and tell me the exact flash message shown after logging in.",
    )

    assert outcome.success is True, outcome.summary
    assert "You logged into a secure area!" in outcome.summary, outcome.summary
    assert not outcome.transcript.texts_matching(r"Open the live browser"), (
        "the run asked the user for a login it had been given"
    )
    _one_final_message(outcome)


def test_a_captcha_is_handed_over_and_a_cancel_ends_the_run_cleanly(battery: Battery) -> None:
    def act(job_id: str, state: dict, b: Battery) -> list[dict]:
        handoff_id, record = b.wait_for_handoff(b.conversation_id())
        assert b.decide_handoff(handoff_id, "cancel") == "cancelled"
        return [record]

    outcome = battery.run(
        "Use the browser. Go to https://www.google.com/recaptcha/api2/demo, complete the reCAPTCHA "
        "and submit the form, then tell me what the page says.",
        on_running=act,
    )

    assert outcome.handoffs and re.search(
        r"captcha|robot", outcome.handoffs[0].get("reason", ""), re.I
    )
    assert outcome.success is not True
    assert outcome.transcript.texts_matching(r"🛑 Stopped"), outcome.transcript.texts
    _one_final_message(outcome)


# --------------------------------------------------------------------------
# Honesty under failure
# --------------------------------------------------------------------------


def test_an_action_the_page_cannot_offer_is_reported_not_faked(battery: Battery) -> None:
    outcome = battery.run(
        'Use the browser. Go to https://example.com, click the "Buy now" button and tell me the order number it shows.',
    )

    assert outcome.success is not True, outcome.summary
    assert not re.search(
        r"order number is|order #\s*\d|clicked (the )?\"?buy now", outcome.summary, re.I
    ), f"fabricated an action: {outcome.summary}"
    assert re.search(r"no|not|couldn't|could not|unable", outcome.summary, re.I), outcome.summary
    _one_final_message(outcome)
    _no_contradiction(outcome)


def test_a_site_that_does_not_exist_fails_once_and_plainly(battery: Battery) -> None:
    outcome = battery.run(
        "Use the browser. Go to https://this-site-truly-does-not-exist-gaia-battery.invalid and tell me its page title.",
    )

    assert outcome.success is not True
    assert re.search(
        r"couldn't|could not|unreachable|does not exist|failed|not load|no such",
        outcome.summary,
        re.I,
    ), outcome.summary
    _one_final_message(outcome)
    _no_contradiction(outcome)


def test_content_the_engine_cannot_see_is_reported_as_unseen(battery: Battery) -> None:
    """An iframe's contents are a known gap; the answer must say so rather than invent text."""
    truth = "Your content goes here."
    outcome = battery.run(
        f"Use the browser. Go to {_INTERNET}/iframe and tell me the exact text inside the editor frame.",
    )

    honest = truth in outcome.summary or re.search(
        r"could not|couldn't|unable|not (visible|readable|see)", outcome.summary, re.I
    )
    assert honest, f"neither the real text nor an admission: {outcome.summary}"
    _one_final_message(outcome)


def test_a_stop_from_the_user_ends_the_run_with_one_message(battery: Battery) -> None:
    def stop(job_id: str, state: dict, b: Battery) -> list[dict]:
        time.sleep(20)
        proc = b.send("stop", settle_ms=30_000, channel=b.last_channel)
        try:
            proc.wait(timeout=240)
        finally:
            b.stop_sender(proc)
        return []

    outcome = battery.run(
        "Use the browser for this whole task. Go to news.ycombinator.com, open the top 10 stories one "
        "by one and summarise each in one line.",
        on_running=stop,
    )

    assert outcome.success is not True
    assert outcome.status in ("stopped", "cancelled", "failed"), outcome.state
    finals = outcome.transcript.texts_matching(r"🛑 Stopped|⚠️ Couldn't finish")
    assert len(finals) == 1, finals
    _no_contradiction(outcome)


def test_one_message_never_starts_two_browser_runs(battery: Battery) -> None:
    before = set(battery.job_ids())
    outcome = battery.run(
        "Use the browser. Go to https://example.com and tell me its main heading.",
    )
    started = set(battery.job_ids()) - before

    assert started == {outcome.job_id}, f"one message started {len(started)} runs: {started}"
    assert outcome.success is True and "Example Domain" in outcome.summary, outcome.summary
    _one_final_message(outcome)
    _no_contradiction(outcome)
