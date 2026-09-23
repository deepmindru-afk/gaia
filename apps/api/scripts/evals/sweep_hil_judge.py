"""Offline threshold sweep over a hil-judge run journal. Free — no model calls.

Usage: uv run --group backend python -m scripts.evals.sweep_hil_judge [run-id]
Without run-id, sweeps the newest hil-judge run under scripts/evals/runs/.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from scripts.evals.core.paths import RUNS_DIR
from scripts.evals.suites.hil_judge import HilJudgeSuite, sweep_journal


def _is_hil_judge_run(run_dir: Path) -> bool:
    meta_path = run_dir / "run.json"
    if not (meta_path.exists() and (run_dir / "journal.jsonl").exists()):
        return False
    return str(json.loads(meta_path.read_text()).get("suite")) == HilJudgeSuite.name


def _latest_hil_judge_run() -> str:
    candidates = sorted(
        (p for p in RUNS_DIR.iterdir() if p.is_dir() and _is_hil_judge_run(p)),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise SystemExit(f"no {HilJudgeSuite.name} runs found under {RUNS_DIR}")
    return candidates[-1].name


def main() -> None:
    run_id = sys.argv[1] if len(sys.argv) > 1 else _latest_hil_judge_run()
    print(f"sweeping run {run_id}")
    print(sweep_journal(RUNS_DIR / run_id))


if __name__ == "__main__":
    main()
