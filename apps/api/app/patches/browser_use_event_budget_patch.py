"""Give Browser-Use's screenshot and state-read events a budget this engine can meet.

Measured on Obscura 2026-09-19: the first Page.captureScreenshot of a very long
page (de.wikipedia.org/wiki/Berlin, 85,000px, 14,000 nodes) takes 24 to 35s, and
about 2s once rendered. Browser-Use allows a screenshot 15s and the state read
that contains it 30s, so the capture errored, was retried, and the user sat
through a minute of silence for a page that had loaded in 3.5s. A slow render
that completes costs less than a timeout plus a retry.

Browser-Use reads each budget from the environment when it builds the event, so
an operator's own value wins over these defaults.
"""

import os

# The state read carries the screenshot, so its budget has to sit above it.
_EVENT_BUDGET_SECONDS = {
    "TIMEOUT_ScreenshotEvent": "60",
    "TIMEOUT_BrowserStateRequestEvent": "90",
}


def apply() -> None:
    """Set each budget unless the environment already names one."""
    for name, seconds in _EVENT_BUDGET_SECONDS.items():
        os.environ.setdefault(name, seconds)


apply()
