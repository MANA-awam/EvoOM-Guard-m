# ─────────────────────────────────────────────────────────────────────────────
# Copyright (c) 2026 Mana Alharbi (مانع الحربي). All rights reserved.
# Source-available — see LICENSE for permitted use.
# Sole owner & author: Mana Alharbi (مانع الحربي).
# ─────────────────────────────────────────────────────────────────────────────
"""The ``evo-guard`` command-line entry point.

Three input shapes, mirroring how a patch reaches you:

    evo-guard [<repo>] --diff <file|->      # a base...HEAD unified diff (recommended)
    evo-guard --base <dir> --head <dir>     # two checkouts (the Action's shape)
    evo-guard <repo> --patch <file|->       # EvoOM <<<FILE>>>/<<<PATCH>>> edit blocks

Exit code 0 only on a clean ``PASS`` — pipe it straight into CI.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evo-guard",
        description=(
            "AI patch verification gate: does this patch fix the code without "
            "gaming the tests? Rejects any edit to the tests or their "
            "configuration; reads the verdict from a judge-owned report, never "
            "from stdout."
        ),
    )
    p.add_argument(
        "repo", nargs="?", default=None,
        help="the repository to verify against (the base); omit when using --base/--head",
    )
    p.add_argument(
        "--patch", default=None,
        help="candidate patch in <<<FILE>>>/<<<PATCH>>> block format ('-' for stdin)",
    )
    p.add_argument("--base", default=None, help="base checkout dir (diff mode, e.g. a PR's target)")
    p.add_argument("--head", default=None, help="head checkout dir (diff mode, e.g. a PR's source)")
    p.add_argument(
        "--diff", default=None,
        help="a base...HEAD unified diff ('-' for stdin), verified against the current "
        "checkout (the repo arg or cwd) by reverse-applying it",
    )
    p.add_argument(
        "--test-command", default=None,
        help="test command run inside the repo copy (default: pytest -q)",
    )
    p.add_argument("--protected", nargs="*", default=[], help="extra globs the patch may not modify")
    p.add_argument("--timeout", type=int, default=120, help="per-run suite timeout in seconds")
    p.add_argument("--json", dest="json_out", default=None, help="write the JSON verdict to this path")
    p.add_argument("--report", default=None, help="write the Markdown report here (else stdout)")
    from evoom_guard import __version__

    p.add_argument("--version", action="version", version=f"evo-guard {__version__}")
    return p


def _read_text(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8") as f:
        return f.read()


def main(argv: list[str] | None = None, *, out: Callable[[str], None] = print) -> int:
    """Run the gate. Returns a process exit code (0 only on PASS)."""
    args = build_parser().parse_args(argv)

    from evoom_guard.guard import (
        candidate_from_dirs,
        guard,
        guard_from_diff,
        render_report,
        write_json,
    )

    test_command = args.test_command.split() if args.test_command else None
    protected = tuple(args.protected)
    deleted: list[str] = []

    if args.diff is not None:
        # A base...HEAD diff verified against the current checkout (repo arg or cwd)
        # by reverse-applying it — so `git diff … | evo-guard --diff -` just works.
        head = args.repo or os.getcwd()
        result, deleted = guard_from_diff(
            head, _read_text(args.diff),
            test_command=test_command, protected=protected, timeout=args.timeout,
        )
    elif args.base and args.head:
        candidate, deleted = candidate_from_dirs(args.base, args.head)
        result = guard(
            args.base, candidate,
            test_command=test_command, protected=protected, timeout=args.timeout,
        )
        result.source = "base/head"
    elif args.repo and args.patch:
        result = guard(
            args.repo, _read_text(args.patch),
            test_command=test_command, protected=protected, timeout=args.timeout,
        )
        result.source = "edit blocks"
    else:
        out(
            "usage: evo-guard <repo> --patch <file|->   |   "
            "evo-guard --base <dir> --head <dir>   |   "
            "evo-guard [<repo>] --diff <file|->"
        )
        return 2

    report = render_report(result, deleted=deleted)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        out(f"wrote {args.report}")
    else:
        out(report)
    if args.json_out:
        write_json(result, args.json_out, deleted=deleted)
    return result.exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
