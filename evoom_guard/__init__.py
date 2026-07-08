# ─────────────────────────────────────────────────────────────────────────────
# Copyright (c) 2026 Mana Alharbi (مانع الحربي). All rights reserved.
# Source-available — see LICENSE for permitted use.
# Sole owner & author: Mana Alharbi (مانع الحربي).
# ─────────────────────────────────────────────────────────────────────────────
"""EvoOM Guard — an AI patch verification gate.

Does this patch fix the code **without gaming the tests**? Guard rejects any
edit to the tests or their configuration before the suite runs, and reads the
verdict from a judge-owned JUnit report + the process exit code — never from
stdout — so a forged "9999 passed" cannot flip it.

Extracted as a standalone tool from the EvoOM verification platform.
"""

from evoom_guard.guard import (
    ERROR,
    FAIL,
    PASS,
    REJECTED,
    GuardResult,
    candidate_from_dirs,
    guard,
    guard_from_diff,
    render_report,
    write_json,
)

__version__ = "0.1.0"

__all__ = [
    "PASS",
    "REJECTED",
    "FAIL",
    "ERROR",
    "GuardResult",
    "guard",
    "guard_from_diff",
    "candidate_from_dirs",
    "render_report",
    "write_json",
    "__version__",
]
