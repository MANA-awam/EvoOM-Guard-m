# ─────────────────────────────────────────────────────────────────────────────
# Copyright (c) 2026 Mana Alharbi (مانع الحربي). All rights reserved.
# Source-available — see LICENSE for permitted use.
# Sole owner & author: Mana Alharbi (مانع الحربي).
# ─────────────────────────────────────────────────────────────────────────────
"""The repo-level judge (extracted from EvoOM).

A candidate is a *set of file edits* applied to a throwaway copy of a real
repository and judged by the repository's own test suite.

Hypothesis format — full-file blocks, not unified diffs (LLM diffs break on
drifted line numbers; whole-file replacement is robust):

    <<<FILE: relative/path/to/file.py>>>
    ...the complete new content of that file...
    <<<END FILE>>>

Any number of blocks. Each block replaces (or creates) one file inside a
throwaway copy of the repo; the original repository is **never** touched.

Surgical-edit format — for changing a *large existing* file without rewriting it
whole (issue #15), a search/replace block applied via
:func:`evoom_guard.patch_applier.apply_patch` with a unique anchor:

    <<<PATCH: relative/path/to/file.py>>>
    <<<SEARCH>>>
    ...a unique anchor copied verbatim from the file...
    <<<REPLACE>>>
    ...its replacement...
    <<<END PATCH>>>

The anchor must occur **exactly once** in the file (else the patch is rejected
with ``AmbiguousMatchError``); a missing anchor is ``NoMatchError``. Both surface
as a precise diagnostic the loop feeds back, so the next generation can fix the
anchor. ``FILE`` and ``PATCH`` blocks may be mixed; patches apply in order, after
the file blocks.

Golden rule, enforced: the candidate may NOT modify the harness that judges it
— neither the tests nor their configuration. Paths under ``tests/``, files named
``test_*.py`` / ``*_test.py`` / ``conftest.py``, and any extra ``protected`` globs
are rejected outright, otherwise the loop would learn to delete its own judge. The
same rejection covers test-runner / build configuration (``pyproject.toml``,
``pytest.ini``, ``tox.ini``, ``setup.cfg``, ``vitest.config.*``, ``foundry.toml``,
…): editing it is a *reward-hack* — a candidate can make a failing suite report
success WITHOUT fixing the code by deselecting the failing tests (e.g. a pytest
``addopts = "-k <passing test>"``). See :func:`is_protected_config`. The
dual-purpose ``package.json`` is not rejected (it carries real dependencies and
source metadata); instead its test-harness fields (``scripts.test`` and embedded
``jest``/``vitest`` config) are restored from the pristine original after a
candidate edit — see :func:`restore_judge_package_json`.

Score gradient (reuses :func:`evoom_guard.scoring.fraction_score`):

    0.02  no parseable file blocks
    0.05  unsafe / protected / config path (absolute, ``..`` escape, test or
          test-config files)
    0.10  test session failed to start (collection/usage error, no tests ran)
    0.25+ tests ran; score climbs with the fraction passed
    1.00  full pass (exit code 0)

SECURITY — the suite runs in a subprocess with a hard timeout and POSIX
rlimits, but it needs the repo's installed dependencies, so the strong ``-I -S``
isolation of ``CodeVerifier`` does not apply here. Treat this as *basic*
isolation: for untrusted targets or unattended VPS operation, run it inside the
network-less hardened container (see ``docker/`` and ``DockerCodeVerifier``).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from fnmatch import fnmatch
from typing import NamedTuple, TypedDict

from evoom_guard.patch_applier import PatchError, apply_patch
from evoom_guard.contracts import VerdictResult
from evoom_guard.scoring import fraction_score

try:  # POSIX-only; absent on Windows.
    import resource
except ImportError:  # pragma: no cover - platform dependent
    resource = None  # type: ignore[assignment]

# Directories never copied into the throwaway working copy.
COPY_IGNORE = (
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    ".evo_runs", ".pytest_cache", ".mypy_cache", "dist", "build",
)

# Test-file basenames the candidate may not touch.
_PROTECTED_BASENAMES = ("test_*.py", "*_test.py", "conftest.py")

# Test-runner / build-configuration basenames the candidate may not touch. Editing
# the *harness configuration* (rather than the source under test) is a reward-hack:
# the candidate can make a failing suite report success WITHOUT fixing the code —
# e.g. a pytest ``[tool.pytest.ini_options] addopts = "-k <passing test>"`` that
# deselects the failing tests, an ``--ignore`` / ``--deselect`` of the failing
# module, or a JS runner's ``include`` narrowed to the passing specs. The
# candidate's job is to fix the SOURCE, never the harness that judges it. Matched
# on the basename anywhere in the tree, case-insensitively. (``conftest.py`` is
# already covered by ``_PROTECTED_BASENAMES`` above.)
_PROTECTED_CONFIG = (
    # pytest / Python test configuration
    "pytest.ini", ".pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml",
    # JS/TS test-runner configuration (``package.json`` is dual-purpose — see
    # ``is_protected_config`` for why it is deliberately not rejected wholesale).
    "vitest.config.*", "vite.config.*", "jest.config.*", "jest.setup.*",
    ".mocharc.*", "karma.conf.*", "cypress.config.*", "playwright.config.*",
    "ava.config.*", ".nycrc", ".nycrc.*",
    # Solidity / fuzzing toolchains
    "foundry.toml", "echidna.yaml", "slither.config.json",
    # Build/test *runners* that redefine how the suite is invoked when the
    # ``test_command`` shells out to them (``make test`` / ``nox`` / ``invoke``).
    # Editing one is the reward-hack equivalent of editing ``addopts``: it lets a
    # candidate point the judge at a passing target without fixing the source.
    "Makefile", "GNUmakefile", "noxfile.py", "Justfile",
)

# Files Python executes *inside the judge process itself* with no test ever naming
# them — so a candidate that writes one runs code in the judge, not in the program
# under test, and can subvert the verdict (force ``sys.exit(0)``, monkey-patch the
# runner, rewrite the report) without touching a single protected test/config file.
#   * ``sitecustomize.py`` / ``usercustomize.py`` are imported automatically during
#     interpreter start-up whenever they are importable on ``sys.path``;
#   * a ``*.pth`` file on the path may carry an executable ``import …`` line that
#     runs at start-up.
# These are rejected outright (the judge owns its own process), matched on the
# basename case-insensitively. See :func:`is_judge_autoexec`.
_PROTECTED_AUTOEXEC = ("sitecustomize.py", "usercustomize.py", "*.pth")

class RepoProblem(TypedDict, total=False):
    """A repo-level problem definition."""

    name: str
    repo_path: str            # root of the target repository (never modified)
    description: str          # the task brief, in natural language
    test_command: list[str]   # judge command (default: pytest -q in the copy)
    target_files: list[str]   # generator hint: files to show the model first
    protected: list[str]      # extra globs the candidate may not modify
    timeout: int              # per-candidate suite timeout (CLI uses this)
    mem_limit_mb: int         # address-space cap for the suite (CLI uses this);
                              # 0 disables the cap — required for node/V8 suites,
                              # whose virtual reservations exceed any sane RLIMIT_AS
    hide_tests: bool          # closed-book mode: the generator must not show the
                              # judging test files' content to the model
    # Container-judge fields (DockerRepoVerifier, S21):
    docker_image: str         # runtime image, e.g. "node:22-slim"
    network: str              # "none" (default) or a docker network name
    judge_env: dict[str, str]  # explicit env passed into the container
    mounts_ro: list[str]      # "host:container" read-only binds
    tmpfs: list[str]          # container paths granted scratch (tmpfs) writes


_BLOCK_RE = re.compile(
    r"<<<FILE:\s*(?P<path>[^>\n]+?)\s*>>>\r?\n(?P<body>.*?)\r?\n?<<<END\s*FILE>>>",
    re.DOTALL,
)

# A surgical-edit block: one search/replace hunk for one file,
# applied with a unique anchor (issue #15). Multiple blocks apply in order.
_PATCH_BLOCK_RE = re.compile(
    r"<<<PATCH:\s*(?P<path>[^>\n]+?)\s*>>>\r?\n"
    r"<<<SEARCH>>>\r?\n(?P<search>.*?)\r?\n"
    r"<<<REPLACE>>>\r?\n(?P<replace>.*?)\r?\n?"
    r"<<<END\s*PATCH>>>",
    re.DOTALL,
)

# Lenient fallbacks — used ONLY when the strict parsers above find nothing.
# Correct, winning solutions have been discarded because a model emitted a
# near-miss format (observed live: single-angle-bracket ``<PATCH>…</PATCH>`` with
# XML-style closers and no file path). Rather than throw away a fix over
# delimiters, recover these variants: any run of angle brackets, optional
# ``/``/``END`` closers, and a path inferred from the task's target file when a
# PATCH omits it. Well-formed (strict) output never reaches these.
_LENIENT_FILE_RE = re.compile(
    r"<+\s*FILE\s*:\s*(?P<path>[^>\n]+?)\s*>+\r?\n?"
    r"(?P<body>.*?)\r?\n?"
    r"<+\s*/?\s*(?:END\s*)?FILE\s*>+",
    re.DOTALL | re.IGNORECASE,
)
_LENIENT_PATCH_RE = re.compile(
    r"<+\s*PATCH\s*(?::\s*(?P<path>[^>\n]*?))?\s*>+\s*"
    r"<+\s*SEARCH\s*>+\r?\n?(?P<search>.*?)\s*(?:<+\s*/\s*SEARCH\s*>+\s*)?"
    r"<+\s*REPLACE\s*>+\r?\n?(?P<replace>.*?)\s*(?:<+\s*/\s*REPLACE\s*>+\s*)?"
    r"<+\s*/?\s*(?:END\s*)?PATCH\s*>+",
    re.DOTALL | re.IGNORECASE,
)

# pytest's summary line, e.g. "2 failed, 3 passed in 0.12s" / "1 error in 0.05s".
_PASSED_RE = re.compile(r"(\d+) passed")
_FAILED_RE = re.compile(r"(\d+) failed")
_ERROR_RE = re.compile(r"(\d+) errors?")

# Lines that carry the *essence* of a failure for the generator:
# test names, assertion diffs, counterexamples, error messages, and the summary.
# Matches both pytest ("FAILED", "E   assert") and vitest/jest ("FAIL", "×",
# "- Expected"/"+ Received", fast-check "Counterexample") vocabularies.
_DIAG_LINE_RE = re.compile(
    r"FAIL|×|✗|Expected|Received|expected|received|Counterexample|"
    r"AssertionError|Error:|assert|Tests\s|Test Files|=== |--- |E\s{3}"
)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def distill_diagnostics(output: str, *, max_chars: int = 1600) -> str:
    """Distill a test run's output to what the generator can act on.

    A raw tail of the output is often all stack trace — the model then never
    sees *which* assertion failed or the expected/received values, and the loop
    goes blind and stagnates (observed live on a vitest+fast-check judge). Keep
    only the failure-essence lines, newest last, within ``max_chars``; fall back
    to the raw tail when nothing matches.
    """
    clean = _ANSI_RE.sub("", output or "")
    picked = [ln.strip() for ln in clean.splitlines() if _DIAG_LINE_RE.search(ln)]
    # Drop pure stack-trace noise that mentions matched words inside file paths.
    picked = [ln for ln in picked if not ln.lstrip().startswith(("❯", "at "))]
    if not picked:
        return clean[-800:]
    text = "\n".join(picked)
    return text[-max_chars:]


def parse_file_blocks(hypothesis: str) -> dict[str, str]:
    """Extract ``{relative_path: content}`` from the hypothesis.

    Tolerant of surrounding prose or markdown fences — only the blocks are read.
    Later blocks for the same path win (the model corrected itself).
    """
    blocks: dict[str, str] = {}
    for m in _BLOCK_RE.finditer(hypothesis or ""):
        blocks[m.group("path").strip()] = m.group("body")
    return blocks


class PatchBlock(NamedTuple):
    """One unique-anchor search/replace edit for one file."""

    path: str
    search: str
    replace: str


def parse_patch_blocks(hypothesis: str) -> list[PatchBlock]:
    """Extract ordered ``<<<PATCH>>>`` edits from the hypothesis.

    Each is one unique-anchor search/replace applied by :func:`apply_patch`.
    Returned in document order; multiple edits (even to the same file) apply in
    sequence. Tolerant of surrounding prose — only the blocks are read.
    """
    return [
        PatchBlock(m.group("path").strip(), m.group("search"), m.group("replace"))
        for m in _PATCH_BLOCK_RE.finditer(hypothesis or "")
    ]


def parse_blocks_lenient(
    hypothesis: str, default_path: str | None = None
) -> tuple[dict[str, str], list[PatchBlock]]:
    """Best-effort recovery of near-miss block formats (see the lenient regexes).

    Returns the same ``(file_blocks, patch_blocks)`` shape as the strict parsers,
    so callers can fall back to it transparently. A PATCH that omits its path
    adopts ``default_path`` (the task's single target file) when one is given;
    otherwise it is dropped, since the file to edit is unknown. Intended to run
    only when the strict parsers find nothing.
    """
    files: dict[str, str] = {}
    for m in _LENIENT_FILE_RE.finditer(hypothesis or ""):
        files[m.group("path").strip()] = m.group("body")
    patches: list[PatchBlock] = []
    for m in _LENIENT_PATCH_RE.finditer(hypothesis or ""):
        path = (m.group("path") or "").strip() or (default_path or "")
        if path:
            patches.append(PatchBlock(path, m.group("search"), m.group("replace")))
    return files, patches


def is_safe_relpath(path: str) -> bool:
    """Is the path safe? Relative, normalized, and unable to escape the repo root."""
    if not path or os.path.isabs(path) or "\\" in path:
        return False
    parts = path.split("/")
    return all(p not in ("", ".", "..") for p in parts)


def is_protected(path: str, extra_globs: tuple[str, ...] = ()) -> bool:
    """Is this one of the files that judge the candidate?

    Protects anything in a ``tests``/``test`` directory segment, standard pytest
    test-file names anywhere in the tree, and caller-supplied globs — all matched
    **case-insensitively** (``TESTS/x.py``, ``Conftest.PY`` are protected too),
    while still comparing whole segments/patterns so look-alikes (``latest/``,
    ``testing/``, ``contest.py``) are not over-matched.
    """
    parts = path.split("/")
    if any(p.lower() in ("tests", "test") for p in parts[:-1]):
        return True
    base = parts[-1]
    if any(fnmatch(base.lower(), pat.lower()) for pat in _PROTECTED_BASENAMES):
        return True
    return any(fnmatch(path.lower(), g.lower()) for g in extra_globs)


def is_protected_config(path: str) -> bool:
    """Is this a test-runner / build-config file the candidate may not modify?

    Editing the harness *configuration* (instead of the source under test) lets a
    candidate game the judge without fixing anything — e.g. a pytest
    ``addopts = "-k <passing test>"`` that deselects the failing tests, or an
    ``--ignore`` / ``--deselect`` of the failing module. The judge owns the test
    harness; the candidate owns only the source. Matched on the basename anywhere
    in the tree, case-insensitively (so ``PyProject.TOML`` is caught too).

    ``package.json`` is intentionally NOT rejected wholesale: it defines the whole
    JS project, so blocking every edit would reject legitimate source/dependency
    fixes. Its test-script / embedded-runner-config vector is a tracked follow-up
    (restore the test-relevant fields rather than reject the file).
    """
    base = path.split("/")[-1].lower()
    return any(fnmatch(base, pat.lower()) for pat in _PROTECTED_CONFIG)


def is_judge_autoexec(path: str) -> bool:
    """Is this a file Python auto-executes inside the judge process?

    ``sitecustomize.py`` / ``usercustomize.py`` are imported automatically at
    interpreter start-up, and a ``*.pth`` file may carry an executable ``import``
    line — none of which any test has to reference. A candidate that writes one
    therefore runs arbitrary code *in the judge's own process* (it can force
    ``sys.exit(0)``, monkey-patch the runner, or rewrite the result report) without
    editing a protected test or config file. The judge owns its process, so these
    are rejected like the tests themselves. Matched on the basename,
    case-insensitively (see :data:`_PROTECTED_AUTOEXEC`).
    """
    base = path.split("/")[-1].lower()
    return any(fnmatch(base, pat.lower()) for pat in _PROTECTED_AUTOEXEC)


# ``package.json`` keys/scripts that configure the JS test harness. A candidate may
# legitimately edit ``package.json`` (dependencies, build/lint scripts, source
# metadata), so it is not rejected wholesale — but it must not redefine the harness
# that judges it (the reward-hack equivalent of pytest's ``addopts``). These fields
# are restored to the original after a candidate edit; everything else is kept.
_PKG_RUNNER_KEYS = ("jest", "vitest", "mocha", "ava", "c8", "nyc")


def _is_judge_script(name: str) -> bool:
    """A ``scripts`` entry that runs/!wraps the test suite (so it judges)."""
    return name == "test" or name.startswith("test:") or name in ("pretest", "posttest")


def restore_judge_package_json(original_text: str | None, candidate_text: str) -> str:
    """Return the candidate ``package.json`` with the test-harness fields restored.

    The candidate must not game a JS judge by editing ``package.json`` instead of
    fixing the code — e.g. pointing ``scripts.test`` at only the passing specs, or
    narrowing an embedded ``jest``/``vitest`` ``include``. This is the dual-purpose
    analogue of :func:`is_protected_config`: rather than reject the whole file
    (which would block legitimate dependency/source fixes), we **restore only the
    judging fields** from the pristine original — the embedded runner-config keys
    (:data:`_PKG_RUNNER_KEYS`) and the test ``scripts`` (:func:`_is_judge_script`) —
    and keep every other candidate edit.

    Pure and defensive: if either text is not valid JSON object, the candidate text
    is returned unchanged (a malformed ``package.json`` fails the run on its own).
    When ``original_text`` is ``None`` (the repo had no ``package.json``), the judge
    fields are *stripped* from the candidate — a candidate cannot introduce a test
    harness where the original defined none. Returns ``candidate_text`` byte-for-byte
    when nothing judging changed, so untouched formatting is preserved.
    """
    try:
        candidate = json.loads(candidate_text)
    except (ValueError, TypeError):
        return candidate_text  # not valid JSON; let the run fail naturally
    if not isinstance(candidate, dict):
        return candidate_text
    try:
        original = json.loads(original_text) if original_text else {}
    except (ValueError, TypeError):
        original = {}
    if not isinstance(original, dict):
        original = {}

    changed = False

    # 1) Top-level embedded runner configs: restore to original, or drop if absent.
    for key in _PKG_RUNNER_KEYS:
        if key in original:
            if candidate.get(key) != original[key]:
                candidate[key] = original[key]
                changed = True
        elif key in candidate:
            del candidate[key]
            changed = True

    # 2) Test scripts: restore each judging script to original, or drop if absent.
    orig_scripts = original.get("scripts")
    orig_scripts = orig_scripts if isinstance(orig_scripts, dict) else {}
    cand_scripts_raw = candidate.get("scripts")
    cand_scripts = dict(cand_scripts_raw) if isinstance(cand_scripts_raw, dict) else {}
    scripts_changed = False
    for name in {n for n in (set(cand_scripts) | set(orig_scripts)) if _is_judge_script(n)}:
        if name in orig_scripts:
            if cand_scripts.get(name) != orig_scripts[name]:
                cand_scripts[name] = orig_scripts[name]
                scripts_changed = True
        elif name in cand_scripts:
            del cand_scripts[name]
            scripts_changed = True
    if scripts_changed:
        changed = True
        candidate["scripts"] = cand_scripts

    if not changed:
        return candidate_text
    return json.dumps(candidate, indent=2, ensure_ascii=False) + "\n"


def reject_unsafe_or_protected(
    paths: list[str], extra: tuple[str, ...]
) -> VerdictResult | None:
    """Reject the first unsafe or judge path.

    Shared by both repo verifiers and covering ``FILE`` and ``PATCH`` paths
    alike: a candidate may never escape the repo root or touch the tests that
    judge it (else the loop would learn to delete its own judge). Returns the
    rejection verdict for the first offending path, or ``None`` if all are safe.
    """
    for path in paths:
        if not is_safe_relpath(path):
            return VerdictResult(
                passed=False,
                score=0.05,
                diagnostics=f"unsafe path rejected: {path}",
                artifact={"files_changed": []},
            )
        if is_protected(path, extra):
            return VerdictResult(
                passed=False,
                score=0.05,
                diagnostics=(
                    f"modifying the judging tests is "
                    f"forbidden: {path}"
                ),
                artifact={"files_changed": []},
            )
        if is_protected_config(path):
            return VerdictResult(
                passed=False,
                score=0.05,
                diagnostics=(
                    f"modifying the test/build configuration is forbidden: {path} — "
                    "fix the source under test, not the harness that judges it"
                ),
                artifact={"files_changed": []},
            )
        if is_judge_autoexec(path):
            return VerdictResult(
                passed=False,
                score=0.05,
                diagnostics=(
                    f"writing an auto-executed judge file is forbidden: {path} — it "
                    "would run code inside the judge process itself (not the program "
                    "under test); fix the source instead"
                ),
                artifact={"files_changed": []},
            )
    return None


def _read_text_or_none(path: str) -> str | None:
    """Read a UTF-8 file, returning ``None`` if it does not exist / cannot be read."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def apply_blocks_to_copy(
    copy: str, file_blocks: dict[str, str], patch_blocks: list[PatchBlock]
) -> str | None:
    """Materialize file blocks then patches into ``copy``.

    File blocks are written whole (create or replace); patch blocks then edit an
    existing file in place via :func:`apply_patch` (unique anchor). Returns a
    precise error diagnostic when a patch fails to apply — a missing or ambiguous
    anchor, an empty search, or a target that does not exist — so the loop can
    fix the anchor next generation; ``None`` on success.

    Reward-hack guard: ``package.json`` is dual-purpose, so it is not rejected like
    a dedicated config file (:func:`is_protected_config`); instead, for every
    ``package.json`` a candidate touches we snapshot the pristine copy first and,
    after applying, restore only its test-harness fields
    (:func:`restore_judge_package_json`) — neutralising a JS-judge reward-hack while
    keeping legitimate dependency/source edits.
    """
    pkg_paths = sorted(
        {p for p in file_blocks if p.split("/")[-1] == "package.json"}
        | {pb.path for pb in patch_blocks if pb.path.split("/")[-1] == "package.json"}
    )
    pkg_originals: dict[str, str | None] = {}
    for rel in pkg_paths:
        fp = os.path.join(copy, *rel.split("/"))
        pkg_originals[rel] = _read_text_or_none(fp)

    for path, content in file_blocks.items():
        target = os.path.join(copy, *path.split("/"))
        os.makedirs(os.path.dirname(target) or copy, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)

    for pb in patch_blocks:
        target = os.path.join(copy, *pb.path.split("/"))
        try:
            with open(target, encoding="utf-8") as f:
                source = f.read()
        except OSError:
            return (
                f"PATCH target not found: {pb.path} — "
                "use a <<<FILE>>> block "
                "to create new files"
            )
        try:
            patched = apply_patch(source, pb.search, pb.replace)
        except (PatchError, ValueError) as exc:
            return (
                f"PATCH did not apply to {pb.path}: "
                f"{type(exc).__name__}: {exc} — "
                ""
                "copy a unique anchor verbatim from the shown file"
            )
        with open(target, "w", encoding="utf-8") as f:
            f.write(patched)

    # Restore each touched package.json's test-harness fields from its pristine
    # snapshot (no-op for non-JS repos, where pkg_paths is empty).
    for rel in pkg_paths:
        fp = os.path.join(copy, *rel.split("/"))
        candidate_pkg = _read_text_or_none(fp)
        if candidate_pkg is None:
            continue
        restored = restore_judge_package_json(pkg_originals.get(rel), candidate_pkg)
        if restored != candidate_pkg:
            with open(fp, "w", encoding="utf-8") as f:
                f.write(restored)
    return None


def parse_pytest_counts(output: str) -> tuple[int, int]:
    """Read ``(passed, total)`` from a pytest/vitest run's *human* output.

    NOTE — this scrapes the runner's stdout/stderr and is therefore **forgeable**:
    a candidate whose code prints (or writes to fd 1/2) a fake ``"9999 passed"``
    summary inflates these counts. It is retained only to enrich the *diagnostic*
    text shown to the generator; it is **no longer used to compute the score or the
    pass/fail verdict** — that now comes from the judge-owned JUnit-XML report and
    the process exit code (see :func:`parse_junit_xml` / :func:`grade_repo_run`).

    The vitest ``Test Files`` line is excluded — folding it into the ``Tests``
    line counted files as tests and skewed the fitness gradient (a double-count
    confirmed by audit).
    """
    lines = [ln for ln in (output or "").splitlines() if "Test Files" not in ln]
    text = "\n".join(lines)
    passed = sum(int(n) for n in _PASSED_RE.findall(text))
    failed = sum(int(n) for n in _FAILED_RE.findall(text))
    errors = sum(int(n) for n in _ERROR_RE.findall(text))
    return passed, passed + failed + errors


class JUnitCounts(NamedTuple):
    """Authoritative test counts read from a pytest JUnit-XML report."""

    passed: int
    total: int
    failures: int
    errors: int


def parse_junit_xml(xml_text: str) -> JUnitCounts | None:
    """Read authoritative test counts from a pytest JUnit-XML report.

    The counts come from the structured ``<testsuite tests= failures= errors=
    skipped=>`` element pytest writes from the *actual collected test items* — not
    from human-readable stdout — so a candidate that prints a fake ``"9999 passed"``
    summary cannot move them. ``total`` excludes skipped tests; ``passed`` is what
    is left after subtracting failures and errors.

    Returns ``None`` when the document is missing, empty, or malformed (i.e. there
    is no trustworthy verdict to read), so the caller can fall back to the exit code.
    """
    if not xml_text or not xml_text.strip():
        return None
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    # pytest emits <testsuites><testsuite .../></testsuites> (or a bare <testsuite>);
    # ``iter`` finds the suite element(s) either way. Aggregate across suites.
    total = failures = errors = skipped = 0
    seen = False
    for suite in root.iter("testsuite"):
        seen = True
        try:
            total += int(suite.get("tests", 0))
            failures += int(suite.get("failures", 0))
            errors += int(suite.get("errors", 0))
            skipped += int(suite.get("skipped", 0))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None
    if not seen:
        return None
    effective_total = max(0, total - skipped)
    passed = max(0, effective_total - failures - errors)
    return JUnitCounts(passed=passed, total=effective_total, failures=failures, errors=errors)


def _is_pytest_command(cmd: list[str]) -> bool:
    """Does this judge command invoke pytest (so we can request a JUnit report)?"""
    return any("pytest" in str(tok) for tok in cmd)


def grade_repo_run(
    returncode: int, junit: JUnitCounts | None, *, is_pytest: bool
) -> tuple[bool, float, int, int]:
    """Turn a finished suite run into ``(passed, score, tests_passed, tests_total)``.

    The verdict is derived from two signals the candidate cannot forge by writing to
    stdout: the process **exit code**, and (for pytest) the judge-owned **JUnit-XML**
    report. The two must *agree* — an all-pass report paired with a non-zero exit, or
    a clean exit with a report full of failures, is treated as a tamper signature and
    scores the no-verdict floor rather than a pass. Tiers (unchanged from the
    historical gradient, only the *source* of the counts changed):

    * exit 0 **and** the report shows tests ran with no failures/errors → full pass (1.0)
    * exit 1 **and** the report corroborates real failures → climb ``fraction_score``
    * anything else (collection/usage error, missing report, exit/report mismatch)
      → 0.10, the "no clean verdict" floor.

    With no JUnit report and a *custom* (non-pytest) runner, the verdict falls back to
    the exit code alone (still not stdout-scraped); the gradient is coarse without a
    structured report. A missing report from a pytest run means no trustworthy
    verdict (0.10).
    """
    if junit is not None:
        if returncode == 0 and junit.total > 0 and junit.failures == 0 and junit.errors == 0:
            return True, 1.0, junit.passed, junit.total
        # exit 1 = the suite ran and some tests failed; reward the passing fraction,
        # but only when the report corroborates the failure (guards an all-pass XML
        # paired with a non-zero exit — a forgery signature).
        if returncode == 1 and junit.total > 0 and (junit.failures > 0 or junit.errors > 0):
            return False, fraction_score(junit.passed, junit.total), junit.passed, junit.total
        return False, 0.10, junit.passed, junit.total
    # No JUnit report.
    if is_pytest:
        return False, 0.10, 0, 0  # a pytest run with no report = no trustworthy verdict
    # Custom runner: grade by the exit code alone (never stdout). Coarse gradient.
    if returncode == 0:
        return True, 1.0, 0, 0
    if returncode == 1:
        return False, 0.25, 0, 0  # ran and judged, but not a full pass
    return False, 0.10, 0, 0


class RepoVerifier:
    """Apply the hypothesis to a copy of the repo and judge it with its tests.

    Applies the hypothesis to a throwaway copy of ``problem["repo_path"]`` and
    judges it with the repo's own suite (``problem["test_command"]`` or pytest).
    """

    domain = "repo"

    def __init__(
        self,
        timeout: int = 120,
        mem_limit_mb: int = 1024,
        *,
        test_command: list[str] | None = None,
        protected: tuple[str, ...] = (),
    ) -> None:
        self.timeout = timeout
        self.mem_limit_mb = mem_limit_mb
        self.test_command = test_command
        self.protected = protected

    # ------------------------------------------------------------------ #
    def _limits(self):  # pragma: no cover - exercised in the child process
        """preexec hook: cap CPU seconds and address space before exec."""
        if resource is None:
            return None

        def apply() -> None:
            cpu = max(1, int(self.timeout) + 1)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            if self.mem_limit_mb <= 0:
                # mem_limit_mb=0 disables the address-space cap: V8/node reserves
                # terabytes of *virtual* memory (Wasm cage, pointer sandbox), so
                # any practical RLIMIT_AS aborts it. CPU cap + timeout still bound us.
                return
            mem = self.mem_limit_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            except (ValueError, OSError):
                # Some platforms reject RLIMIT_AS; the timeout still bounds us.
                pass

        return apply

    # ------------------------------------------------------------------ #
    def _command(self, problem: RepoProblem | dict) -> list[str]:
        cmd = self.test_command or problem.get("test_command")
        if isinstance(cmd, str):
            return cmd.split()
        if cmd:
            return list(cmd)
        return [sys.executable, "-m", "pytest", "-q", "--color=no", "-p", "no:cacheprovider"]

    # ------------------------------------------------------------------ #
    def verify(self, hypothesis: str, problem: RepoProblem | dict) -> VerdictResult:
        repo_path = str(problem.get("repo_path", ""))
        if not repo_path or not os.path.isdir(repo_path):
            raise ValueError(f"problem['repo_path'] is not a directory: {repo_path!r}")

        file_blocks = parse_file_blocks(hypothesis)
        patch_blocks = parse_patch_blocks(hypothesis)
        if not file_blocks and not patch_blocks:
            # Tolerant fallback before discarding the candidate: recover near-miss
            # delimiters, inferring a missing PATCH path from a single target file.
            targets = [str(t) for t in problem.get("target_files", ()) if str(t).strip()]
            default_path = targets[0] if len(targets) == 1 else None
            file_blocks, patch_blocks = parse_blocks_lenient(hypothesis, default_path)
        if not file_blocks and not patch_blocks:
            return VerdictResult(
                passed=False,
                score=0.02,
                diagnostics=(
                    "no parseable blocks; expected "
                    "<<<FILE: path>>> … <<<END FILE>>> or "
                    "<<<PATCH: path>>> <<<SEARCH>>> … <<<REPLACE>>> … <<<END PATCH>>>"
                ),
                artifact={"files_changed": []},
            )

        extra = self.protected + tuple(problem.get("protected", ()))
        changed = sorted(set(file_blocks) | {pb.path for pb in patch_blocks})
        rejection = reject_unsafe_or_protected(changed, extra)
        if rejection is not None:
            return rejection

        workdir = tempfile.mkdtemp(prefix="evo_repo_")
        copy = os.path.join(workdir, "repo")
        try:
            shutil.copytree(
                repo_path, copy, ignore=shutil.ignore_patterns(*COPY_IGNORE)
            )
            apply_error = apply_blocks_to_copy(copy, file_blocks, patch_blocks)
            if apply_error is not None:
                return VerdictResult(
                    passed=False,
                    score=0.08,
                    diagnostics=apply_error,
                    artifact={"files_changed": changed},
                )

            env = {
                "PATH": os.environ.get("PATH", "/usr/bin"),
                "HOME": workdir,
                "LANG": "C.UTF-8",
                "PYTHONDONTWRITEBYTECODE": "1",
                # Don't let a user-site ``usercustomize`` load into the judge process
                # (defence-in-depth alongside the path-gate's auto-exec rejection).
                "PYTHONNOUSERSITE": "1",
            }
            cmd = self._command(problem)
            is_pytest = _is_pytest_command(cmd)
            # The machine-readable verdict is written to a JUnit report the JUDGE
            # owns — a path *outside* the repo copy, so the candidate (restricted to
            # relative paths inside the copy) cannot pre-plant or overwrite it via an
            # edit. The score is read from this report and the exit code, never from
            # the candidate-influenced stdout.
            xml_path = os.path.join(workdir, "judge-result.xml")
            if is_pytest:
                cmd = [*cmd, f"--junitxml={xml_path}", "-o", "junit_family=xunit2"]
            t0 = time.perf_counter()
            try:
                r = subprocess.run(
                    cmd,
                    cwd=copy,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=env,
                    preexec_fn=self._limits() if os.name == "posix" else None,
                )
            except subprocess.TimeoutExpired:
                return VerdictResult(
                    passed=False,
                    score=0.0,
                    diagnostics=f"test suite timed out after {self.timeout}s",
                    artifact={"elapsed": self.timeout, "files_changed": changed},
                )
            elapsed = time.perf_counter() - t0

            junit = parse_junit_xml(_read_text_or_none(xml_path) or "")
            passed, score, tests_passed, tests_total = grade_repo_run(
                r.returncode, junit, is_pytest=is_pytest
            )
            # stdout/stderr feed only the human-facing diagnostic the generator
            # learns from — never the score or the verdict (see grade_repo_run).
            output = r.stdout + "\n" + r.stderr

            return VerdictResult(
                passed=passed,
                score=score,
                diagnostics=distill_diagnostics(output),
                artifact={
                    "returncode": r.returncode,
                    "elapsed": elapsed,
                    "tests_passed": tests_passed,
                    "tests_total": tests_total,
                    "files_changed": changed,
                    "verdict_source": "junit+exit" if junit is not None else "exit",
                },
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
