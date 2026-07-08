# ─────────────────────────────────────────────────────────────────────────────
# Copyright (c) 2026 Mana Alharbi (مانع الحربي). All rights reserved.
# Source-available — see LICENSE for permitted use.
# Sole owner & author: Mana Alharbi (مانع الحربي).
# ─────────────────────────────────────────────────────────────────────────────
"""Blast-radius risk scoring (extracted from EvoOM's ``evoom_patchmin``).

Deterministic and pure: a patch's touched files and line counts map onto a
bounded ``[0, 1]`` score and a coarse low/medium/high level. Standard library
only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import TypeAlias


@dataclass(frozen=True)
class RiskScore:
    """Blast-radius summary of a patch (see :func:`risk_score`)."""

    files_touched: int
    lines_added: int
    lines_removed: int
    protected_hits: list[str]   # touched files matching a protected glob (sorted, unique)
    score: float                # 0..1 blast-radius score
    level: str                  # "low" | "medium" | "high"


def parse_unified_diff(diff: str) -> dict[str, tuple[int, int]]:
    """Parse a unified diff into ``{file_path: (added, removed)}`` line counts.

    Recognizes ``+++ b/<path>`` (or ``+++ <path>``) as the current file and then
    counts content lines starting with ``'+'`` / ``'-'`` for that file, excluding
    the ``+++`` / ``---`` file headers and ``@@`` hunk headers. ``/dev/null``
    targets (pure deletions) are ignored as a destination. Multiple files in one
    diff are handled; counts accumulate per resolved path.

    The leading ``b/`` (or ``a/``) prefix produced by ``git diff`` is stripped so
    the returned paths are repo-relative.
    """
    counts: dict[str, list[int]] = {}
    current: str | None = None

    for line in diff.splitlines():
        if line.startswith("+++"):
            path = _strip_diff_path(line[3:].strip())
            if path == "/dev/null" or path == "":
                current = None
            else:
                current = path
                counts.setdefault(path, [0, 0])
            continue
        if line.startswith("---"):
            continue
        if line.startswith("@@"):
            continue
        if current is None:
            continue
        if line.startswith("+"):
            counts[current][0] += 1
        elif line.startswith("-"):
            counts[current][1] += 1

    return {path: (added, removed) for path, (added, removed) in counts.items()}


def _strip_diff_path(token: str) -> str:
    """Strip a leading ``a/`` or ``b/`` git prefix (but leave ``/dev/null``)."""
    if token == "/dev/null":
        return token
    if token.startswith("a/") or token.startswith("b/"):
        return token[2:]
    return token


# A diff string or a precomputed {file: (added, removed)} mapping.
DiffLike: TypeAlias = "str | Mapping[str, tuple[int, int]]"


def risk_score(
    diff: DiffLike,
    *,
    protected: Sequence[str] = (),
    medium_files: int = 3,
    high_files: int = 8,
    medium_lines: int = 40,
    high_lines: int = 200,
) -> RiskScore:
    """Compute a blast-radius :class:`RiskScore` from a patch.

    ``diff`` is either a unified-diff string (parsed via
    :func:`parse_unified_diff`) or an already-computed
    ``{file: (added, removed)}`` mapping. ``protected`` is a list of fnmatch
    globs; any touched file matching one is a *protected hit* (matched
    case-insensitively against the whole path).

    Score formula (monotone in both files and lines, bounded, then clamped)::

        files_term     = min(1.0, files_touched / high_files)
        lines_term     = min(1.0, total_lines   / high_lines)
        protected_term = 0.25 if protected_hits else 0.0
        score = min(1.0, 0.5 * files_term + 0.5 * lines_term + protected_term)

    ``level`` is ``'high'`` on any protected hit or when either threshold is
    reached; ``'medium'`` at the medium thresholds; else ``'low'``. With
    non-positive thresholds the corresponding term saturates to ``1.0``.
    Deterministic and pure.
    """
    file_counts = diff if isinstance(diff, Mapping) else parse_unified_diff(diff)

    touched = list(file_counts.keys())
    files_touched = len(touched)
    lines_added = sum(added for added, _removed in file_counts.values())
    lines_removed = sum(removed for _added, removed in file_counts.values())
    total_lines = lines_added + lines_removed

    protected_hits = sorted(
        {
            path
            for path in touched
            for glob in protected
            if fnmatch(path.lower(), glob.lower())
        }
    )

    files_term = 1.0 if high_files <= 0 else min(1.0, files_touched / high_files)
    lines_term = 1.0 if high_lines <= 0 else min(1.0, total_lines / high_lines)
    protected_term = 0.25 if protected_hits else 0.0
    score = min(1.0, 0.5 * files_term + 0.5 * lines_term + protected_term)

    if protected_hits or files_touched >= high_files or total_lines >= high_lines:
        level = "high"
    elif files_touched >= medium_files or total_lines >= medium_lines:
        level = "medium"
    else:
        level = "low"

    return RiskScore(
        files_touched=files_touched,
        lines_added=lines_added,
        lines_removed=lines_removed,
        protected_hits=protected_hits,
        score=score,
        level=level,
    )
