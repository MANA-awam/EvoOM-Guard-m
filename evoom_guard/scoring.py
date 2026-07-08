# ─────────────────────────────────────────────────────────────────────────────
# Copyright (c) 2026 Mana Alharbi (مانع الحربي). All rights reserved.
# Source-available — see LICENSE for permitted use.
# Sole owner & author: Mana Alharbi (مانع الحربي).
# ─────────────────────────────────────────────────────────────────────────────
"""The score gradient (extracted from EvoOM's code verifier).

Guard itself only needs the pass/fail verdict, but the judge preserves the
fractional gradient so a partially-passing patch is distinguishable from a
broken one in diagnostics and scores.
"""

from __future__ import annotations

# Gradient anchors. A candidate that runs but passes zero tests stays at the
# 0.25 plateau; partially passing candidates climb from there, always staying
# strictly below a full pass.
PARTIAL_FLOOR = 0.25
PARTIAL_CEIL = 0.95


def partial_score(stderr: str) -> float:
    """Partial credit: a syntax error is worse than a logic error."""
    if "SyntaxError" in stderr:
        return 0.05
    if "NameError" in stderr:
        return 0.10
    return 0.25  # ran, but an assertion failed


def fraction_score(passed: int, total: int, stderr: str = "") -> float:
    """Map a passed/total fraction onto the score gradient.

    Zero passing tests delegates to :func:`partial_score`. Partial passes land
    strictly inside (PARTIAL_FLOOR, PARTIAL_CEIL]; only a full pass reaches 1.0.
    """
    if passed <= 0 or total <= 0:
        return partial_score(stderr)
    if passed >= total:
        return 1.0
    return PARTIAL_FLOOR + (PARTIAL_CEIL - PARTIAL_FLOOR) * (passed / total)
