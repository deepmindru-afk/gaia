"""Offline threshold sweep over a hil-judge run journal. Free — no model calls.

Usage: uv run --group backend python -m scripts.evals.sweep_hil_judge [run-id]
Without run-id, sweeps the newest hil-judge run under scripts/evals/runs/.
"""

from __future__ import annotations

import sys

from scripts.evals.core.paths import RUNS_DIR
from scripts.evals.suites.hil_judge import sweep_journal


def _latest_hil_judge_run() -> str:
    candidates = sorted(
        (p for p in RUNS_DIR.iterdir() if p.is_dir() and (p / "journal.jsonl").exists()),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise SystemExit("no runs found under scripts/evals/runs/")
    return candidates[-1].name


def main() -> None:
    run_id = sys.argv[1] if len(sys.argv) > 1 else _latest_hil_judge_run()
    print(f"sweeping run {run_id}")
    print(sweep_journal(RUNS_DIR / run_id))


if __name__ == "__main__":
    main()
