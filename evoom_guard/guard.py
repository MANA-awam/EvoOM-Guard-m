# ─────────────────────────────────────────────────────────────────────────────
# Copyright (c) 2026 Mana Alharbi (مانع الحربي). All rights reserved.
# Source-available — see LICENSE for permitted use.
# Sole owner & author: Mana Alharbi (مانع الحربي).
# ─────────────────────────────────────────────────────────────────────────────
"""EvoOM Guard — an AI patch verification gate.

Guard answers one question objectively, for a code change produced by anyone (a
human or — the motivating case — an AI agent):

    *Does this patch fix the code, **without gaming the tests**?*

It is a thin, model-free composition of assets extracted from EvoOM:

  * the **reward-hack-resistant repo judge** (:class:`evoom_guard.repo_verifier.RepoVerifier`)
    — applies the patch to a throwaway copy and reads the verdict from a
    *judge-owned* JUnit report + the process exit code, so the patch cannot fake a
    pass by writing to stdout, and is **rejected** outright if it edits the tests or
    their configuration; and
  * the **blast-radius risk score** (:func:`evoom_guard.riskscore.risk_score`).

The result is a single verdict — ``PASS`` / ``REJECTED`` / ``FAIL`` / ``ERROR`` — a
process exit code suitable for CI, and a Markdown report suitable for a PR comment.

Two input shapes:
  * a candidate in EvoOM's edit-block format (``<<<FILE>>>`` / ``<<<PATCH>>>``), the
    same format agents already emit; or
  * a **base** and **head** checkout (the natural shape in a GitHub Action), which
    :func:`candidate_from_dirs` diffs into the block format.

Trust boundary (honest): the judge runs the repo's own test suite in a subprocess
with rlimits and a timeout. That is fine for **trusted** repositories (your own
code, gating a patch). For **untrusted** code, run inside the hardened container
judge — see the README's trust-boundary section. Guard never claims the
subprocess is a security sandbox.
"""

from __future__ import annotations

import difflib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

from evoom_guard.riskscore import risk_score
from evoom_guard.repo_verifier import (
    COPY_IGNORE,
    RepoVerifier,
    is_judge_autoexec,
    is_protected,
    is_protected_config,
    is_safe_relpath,
    parse_file_blocks,
    parse_patch_blocks,
)

# Globs the risk scorer treats as "protected" so a protected hit is visible in the
# blast radius too (mirrors the judge's protected-path convention).
_PROTECTED_GLOBS = (
    "*tests/*", "*test/*", "test_*.py", "*_test.py", "conftest.py",
    "pyproject.toml", "*pytest.ini", "tox.ini", "setup.cfg",
    "*.pth", "sitecustomize.py", "usercustomize.py", "Makefile", "GNUmakefile", "noxfile.py",
)

# Verdicts.
PASS = "PASS"          # tests pass and the harness was untouched
REJECTED = "REJECTED"  # the patch edits the tests / their config (reward-hack)
FAIL = "FAIL"          # the patch applied and ran, but the tests fail
ERROR = "ERROR"        # the patch did not apply / produced no parseable edits


@dataclass
class GuardResult:
    """The outcome of a Guard run."""

    verdict: str
    passed: bool
    reason: str
    files_changed: list[str]
    protected_violations: list[str]
    risk_level: str
    risk_score: float
    tests_passed: int | None = None
    tests_total: int | None = None
    verdict_source: str | None = None
    diagnostics: str = ""
    source: str | None = None              # how the candidate was supplied (e.g. "diff")
    base_reconstruction: str | None = None  # "ok" | "failed" (only for --diff)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "passed": self.passed,
            "reason": self.reason,
            "files_changed": self.files_changed,
            "protected_violations": self.protected_violations,
            "risk_level": self.risk_level,
            "risk_score": round(self.risk_score, 3),
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "verdict_source": self.verdict_source,
            "source": self.source,
            "base_reconstruction": self.base_reconstruction,
            "diagnostics": self.diagnostics[:2000],
        }

    @property
    def exit_code(self) -> int:
        """0 only on a clean PASS; non-zero otherwise (CI-gate friendly)."""
        return 0 if self.verdict == PASS else 1


def _read_repo_file(repo_path: str, rel: str) -> str:
    try:
        with open(os.path.join(repo_path, *rel.split("/")), encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _diff_counts(old: str, new: str) -> tuple[int, int]:
    """(added, removed) line counts between two file contents."""
    added = removed = 0
    for line in difflib.unified_diff(old.splitlines(), new.splitlines(), n=0):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def _risk_map(repo_path: str, candidate: str) -> dict[str, tuple[int, int]]:
    """Build a ``{path: (added, removed)}`` map for the risk scorer.

    For whole-file blocks the count is the real diff against the base file; for
    surgical PATCH blocks it is approximated by the search/replace line counts
    (we do not re-apply to count exactly — risk is a coarse, bounded signal).
    """
    out: dict[str, tuple[int, int]] = {}
    for path, new in parse_file_blocks(candidate).items():
        out[path] = _diff_counts(_read_repo_file(repo_path, path), new)
    for pb in parse_patch_blocks(candidate):
        a, r = len(pb.replace.splitlines()), len(pb.search.splitlines())
        prev_a, prev_r = out.get(pb.path, (0, 0))
        out[pb.path] = (prev_a + a, prev_r + r)
    return out


def changed_paths(candidate: str) -> list[str]:
    """All repo-relative paths a candidate would create or modify."""
    blocks = parse_file_blocks(candidate)
    patches = parse_patch_blocks(candidate)
    return sorted(set(blocks) | {pb.path for pb in patches})


def guard(
    repo_path: str,
    candidate: str,
    *,
    test_command: list[str] | None = None,
    protected: tuple[str, ...] = (),
    timeout: int = 120,
) -> GuardResult:
    """Verify ``candidate`` against ``repo_path`` and return a :class:`GuardResult`.

    The repo at ``repo_path`` is never modified — the judge works on a throwaway
    copy. ``protected`` adds extra globs the patch may not touch (on top of the
    built-in tests/config/auto-exec set).
    """
    changed = changed_paths(candidate)
    violations = sorted(
        p for p in changed
        if is_protected(p, protected) or is_protected_config(p) or is_judge_autoexec(p)
    )

    problem: dict[str, Any] = {"name": "guard", "repo_path": repo_path}
    if test_command:
        problem["test_command"] = test_command
    if protected:
        problem["protected"] = list(protected)

    verdict = RepoVerifier(timeout=timeout).verify(candidate, problem)
    art = verdict.artifact or {}
    risk = risk_score(_risk_map(repo_path, candidate), protected=_PROTECTED_GLOBS + tuple(protected))

    if not changed:
        v, reason = ERROR, (
            "no parseable edit blocks — the patch produced no <<<FILE>>> or "
            "<<<PATCH>>> edits to verify"
        )
    elif violations:
        v, reason = REJECTED, (
            "reward-hack guard: the patch edits the judging tests, their "
            "configuration, or an auto-executed file — fix the source under test, "
            f"not the harness ({', '.join(violations)})"
        )
    elif verdict.passed:
        v, reason = PASS, "all repo tests pass and the patch leaves the test harness untouched"
    elif art.get("tests_total"):
        v, reason = FAIL, (
            f"the repo's tests fail on this patch "
            f"({art.get('tests_passed', 0)}/{art.get('tests_total')} passed)"
        )
    elif verdict.score <= 0.08:
        v, reason = ERROR, "the patch did not apply cleanly (a PATCH anchor did not match)"
    else:
        v, reason = FAIL, "the test session produced no clean verdict (collection/usage error)"

    return GuardResult(
        verdict=v,
        passed=(v == PASS),
        reason=reason,
        files_changed=changed,
        protected_violations=violations,
        risk_level=risk.level,
        risk_score=risk.score,
        tests_passed=art.get("tests_passed"),
        tests_total=art.get("tests_total"),
        verdict_source=art.get("verdict_source"),
        diagnostics=verdict.diagnostics or "",
    )


def candidate_from_dirs(base_dir: str, head_dir: str, *, max_bytes: int = 1_000_000) -> tuple[str, list[str]]:
    """Diff a base and head checkout into an EvoOM ``<<<FILE>>>`` candidate.

    Returns ``(candidate, deleted)`` where ``candidate`` is the block-format patch
    of every file that was added or modified in ``head`` relative to ``base``
    (skipping ``.git`` and the standard ignored dirs), and ``deleted`` lists files
    present in base but absent in head (Guard cannot apply deletions via FILE
    blocks; they are surfaced in the report). Binary/oversized files are skipped.
    """
    base_files = _walk_text_files(base_dir, max_bytes)
    head_files = _walk_text_files(head_dir, max_bytes)
    blocks: list[str] = []
    for rel in sorted(head_files):
        new = head_files[rel]
        if base_files.get(rel) != new:  # added or modified
            blocks.append(f"<<<FILE: {rel}>>>\n{new}\n<<<END FILE>>>")
    deleted = sorted(set(base_files) - set(head_files))
    return "\n".join(blocks), deleted


def _walk_text_files(root: str, max_bytes: int) -> dict[str, str]:
    out: dict[str, str] = {}
    ignore = set(COPY_IGNORE) | {".git"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignore]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            try:
                if os.path.getsize(full) > max_bytes:
                    continue
                with open(full, encoding="utf-8") as f:
                    out[rel] = f.read()
            except (OSError, UnicodeDecodeError):
                continue  # binary or unreadable — not a text patch target
    return out


def _reverse_apply(work_dir: str, diff_file: str) -> bool:
    """Reverse-apply a unified diff in ``work_dir`` (undo it). True on success.

    Tries ``git apply -R`` first (works on a plain directory, no repo needed), then
    falls back to ``patch -R -p1``. Used to reconstruct the BASE tree from the HEAD
    working tree given a base→head diff.
    """
    for cmd in (
        ["git", "apply", "-R", "--whitespace=nowarn", diff_file],
        ["patch", "-R", "-p1", "--no-backup-if-mismatch", "-i", diff_file],
    ):
        if shutil.which(cmd[0]) is None:
            continue
        try:
            r = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0:
            return True
    return False


def _diff_error(reason: str, *, base_reconstruction: str = "failed") -> GuardResult:
    return GuardResult(
        verdict=ERROR, passed=False, reason=reason,
        files_changed=[], protected_violations=[],
        risk_level="low", risk_score=0.0, diagnostics="",
        source="diff", base_reconstruction=base_reconstruction,
    )


def _is_binary_diff(diff_text: str) -> bool:
    """Git marks binary changes with a ``GIT binary patch`` block or a one-line
    ``Binary files a/x and b/x differ`` — Guard cannot verify those."""
    return ("GIT binary patch" in diff_text) or ("\nBinary files " in ("\n" + diff_text))


def _diff_target_paths(diff_text: str) -> list[str]:
    """Every file path a diff targets (both ``---``/``+++`` sides), prefix-stripped.

    ``/dev/null`` (the add/delete marker) is excluded. Used to refuse a diff that
    points outside the repo *before* anything is applied — defence in depth on top
    of ``git apply``'s own unsafe-path guard and the verifier's relpath gate.
    """
    paths: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith(("--- ", "+++ ")):
            tok = line[4:].strip().split("\t", 1)[0]
            if tok in ("/dev/null", ""):
                continue
            if tok.startswith(("a/", "b/")):
                tok = tok[2:]
            paths.append(tok)
    return paths


def guard_from_diff(
    head_dir: str,
    diff_text: str,
    *,
    test_command: list[str] | None = None,
    protected: tuple[str, ...] = (),
    timeout: int = 120,
) -> tuple[GuardResult, list[str]]:
    """Verify a unified diff against the working tree it was produced from.

    ``head_dir`` is the **current** checkout (e.g. the PR head you are standing in);
    ``diff_text`` is a base→head unified diff (e.g. ``git diff main...HEAD``). Guard
    reconstructs the base by **reverse-applying** the diff to a throwaway copy of
    ``head_dir`` — ``head_dir`` itself is **never modified** — then verifies the
    head's changes against that base with the repo's own tests. So
    ``git diff … | evo guard --diff -`` works straight from your tree.

    Returns ``(GuardResult, deleted)``. The verdict is a clear ``ERROR`` (never an
    apply against the real tree) when the diff is empty, binary, references an
    unsafe path (absolute / ``..`` / repo escape), or does not reverse-apply.
    """
    if not (diff_text or "").strip():
        return _diff_error("empty diff — nothing to verify"), []
    if _is_binary_diff(diff_text):
        return _diff_error(
            "binary patches are not supported — Guard verifies text source changes; "
            "the diff contains a binary file change"
        ), []
    unsafe = sorted({p for p in _diff_target_paths(diff_text) if not is_safe_relpath(p)})
    if unsafe:
        return _diff_error(
            "the diff references unsafe path(s) outside the repo (absolute, '..', or "
            f"escaping the root) — refusing to apply: {', '.join(unsafe)}"
        ), []

    workdir = tempfile.mkdtemp(prefix="evo_guard_diff_")
    base = os.path.join(workdir, "base")
    try:
        # base is a copy of head; head_dir is only ever read, never written.
        shutil.copytree(head_dir, base, ignore=shutil.ignore_patterns(*COPY_IGNORE, ".git"))
        diff_file = os.path.join(workdir, "patch.diff")
        with open(diff_file, "w", encoding="utf-8") as f:
            f.write(diff_text if diff_text.endswith("\n") else diff_text + "\n")
        if not _reverse_apply(base, diff_file):
            return _diff_error(
                "the diff did not reverse-apply to the working tree — make sure you "
                "are in the head checkout and the diff is 'base...HEAD' (git/patch needed)"
            ), []
        candidate, deleted = candidate_from_dirs(base, head_dir)
        if not candidate.strip():
            return _diff_error(
                "the diff changed no verifiable source files", base_reconstruction="ok"
            ), deleted
        result = guard(
            base, candidate,
            test_command=test_command, protected=protected, timeout=timeout,
        )
        result.source = "diff"
        result.base_reconstruction = "ok"
        return result, deleted
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


_BADGE = {PASS: "✅ PASS", REJECTED: "⛔ REJECTED", FAIL: "❌ FAIL", ERROR: "⚠️ ERROR"}


def render_report(result: GuardResult, *, deleted: list[str] | None = None, title: str = "EvoOM Guard") -> str:
    """Render a :class:`GuardResult` as a Markdown report (PR-comment ready)."""
    r = result
    tests = (
        f"{r.tests_passed}/{r.tests_total}"
        if r.tests_total is not None else "—"
    )
    lines = [
        f"## {title} — {_BADGE.get(r.verdict, r.verdict)}",
        "",
        f"**{r.reason}**",
        "",
        "| | |",
        "|---|---|",
        f"| Verdict | **{r.verdict}** |",
        f"| Tests passed | {tests} |",
        f"| Files changed | {len(r.files_changed)} |",
        f"| Blast radius | **{r.risk_level}** ({r.risk_score:.2f}) |",
        f"| Verdict source | {r.verdict_source or '—'} |",
    ]
    if r.source:
        lines.append(f"| Input | {r.source} |")
    if r.base_reconstruction:
        lines.append(f"| Base reconstruction | {r.base_reconstruction} |")
    if r.protected_violations:
        lines += [
            "",
            "### ⛔ Reward-hack: the patch tried to edit the judging harness",
            "",
            *[f"- `{p}`" for p in r.protected_violations],
            "",
            "A patch must fix the **source under test**, never the tests or their "
            "configuration. This is rejected before the suite runs.",
        ]
    if deleted:
        lines += [
            "",
            "> Note: these files were **deleted** in head and are not gated by Guard "
            "(it verifies additions/modifications): " + ", ".join(f"`{p}`" for p in deleted),
        ]
    if r.files_changed and not r.protected_violations:
        shown = ", ".join(f"`{p}`" for p in r.files_changed[:15])
        more = "" if len(r.files_changed) <= 15 else f" (+{len(r.files_changed) - 15} more)"
        lines += ["", f"<details><summary>Files changed</summary>\n\n{shown}{more}\n</details>"]
    if r.diagnostics and r.verdict in (FAIL, ERROR):
        diag = r.diagnostics.strip()[:1200]
        lines += ["", "<details><summary>Diagnostics</summary>\n", "```", diag, "```", "</details>"]
    lines += [
        "",
        "<sub>EvoOM Guard reads the verdict from a judge-owned JUnit report + the "
        "process exit code (not stdout), and rejects any edit to the tests or their "
        "config. Trusted-repo subprocess judge; use the container judge for untrusted "
        "code. See the README.</sub>",
    ]
    return "\n".join(lines)


def write_json(result: GuardResult, path: str, *, deleted: list[str] | None = None) -> None:
    payload = result.to_dict()
    if deleted:
        payload["deleted_not_gated"] = deleted
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
