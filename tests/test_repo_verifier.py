# ─────────────────────────────────────────────────────────────────────────────
# Copyright (c) 2026 Mana Alharbi (مانع الحربي). All rights reserved.
# Source-available — see LICENSE for permitted use.
# Sole owner & author: Mana Alharbi (مانع الحربي).
# ─────────────────────────────────────────────────────────────────────────────
"""Repo-level verifier tests (S19).

The block-parsing and path-safety layers are pure functions (no subprocess).
The end-to-end tests scaffold a tiny buggy repo in a temp dir and run a real
pytest suite against candidate patches; they are skipped when pytest is not
importable (the dev extra installs it).
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evoom_guard.scoring import fraction_score
from evoom_guard.repo_verifier import (
    RepoVerifier,
    apply_blocks_to_copy,
    distill_diagnostics,
    grade_repo_run,
    is_judge_autoexec,
    is_protected,
    is_protected_config,
    is_safe_relpath,
    parse_blocks_lenient,
    parse_file_blocks,
    parse_junit_xml,
    parse_patch_blocks,
    parse_pytest_counts,
    restore_judge_package_json,
)

HAS_PYTEST = importlib.util.find_spec("pytest") is not None

# The demo repo: mathx.average divides by len+1 (bug #1)
# and mathx.clamp ignores the lower bound (bug #2). Two tests fail, one passes.
BUGGY_MATHX = """\
def average(nums):
    return sum(nums) / (len(nums) + 1)


def clamp(x, lo, hi):
    return min(x, hi)


def double(x):
    return 2 * x
"""

FIXED_MATHX = """\
def average(nums):
    return sum(nums) / len(nums)


def clamp(x, lo, hi):
    return max(lo, min(x, hi))


def double(x):
    return 2 * x
"""

HALF_FIXED_MATHX = BUGGY_MATHX.replace(
    "return sum(nums) / (len(nums) + 1)", "return sum(nums) / len(nums)"
)

TESTS_MATHX = """\
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mathx import average, clamp, double


def test_average():
    assert average([2, 4]) == 3


def test_clamp_low():
    assert clamp(-5, 0, 10) == 0


def test_double():
    assert double(3) == 6
"""


def block(path: str, body: str) -> str:
    return f"<<<FILE: {path}>>>\n{body}\n<<<END FILE>>>"


def patch_block(path: str, search: str, replace: str) -> str:
    return (
        f"<<<PATCH: {path}>>>\n<<<SEARCH>>>\n{search}\n"
        f"<<<REPLACE>>>\n{replace}\n<<<END PATCH>>>"
    )


def make_repo(root: str) -> None:
    os.makedirs(os.path.join(root, "tests"))
    with open(os.path.join(root, "mathx.py"), "w", encoding="utf-8") as f:
        f.write(BUGGY_MATHX)
    with open(os.path.join(root, "tests", "test_mathx.py"), "w", encoding="utf-8") as f:
        f.write(TESTS_MATHX)


class ParsingTests(unittest.TestCase):
    def test_single_block(self) -> None:
        got = parse_file_blocks(block("a/b.py", "x = 1"))
        self.assertEqual(got, {"a/b.py": "x = 1"})

    def test_multiple_blocks_and_surrounding_prose(self) -> None:
        text = "Here you go:\n```\n" + block("a.py", "A") + "\n```\n" + block("b.py", "B")
        got = parse_file_blocks(text)
        self.assertEqual(got, {"a.py": "A", "b.py": "B"})

    def test_later_block_wins(self) -> None:
        text = block("a.py", "old") + "\n" + block("a.py", "new")
        self.assertEqual(parse_file_blocks(text)["a.py"], "new")

    def test_no_blocks(self) -> None:
        self.assertEqual(parse_file_blocks("def f(): pass"), {})

    def test_pytest_counts(self) -> None:
        self.assertEqual(parse_pytest_counts("2 failed, 3 passed in 0.1s"), (3, 5))
        self.assertEqual(parse_pytest_counts("4 passed in 0.1s"), (4, 4))
        self.assertEqual(parse_pytest_counts("1 error in 0.1s"), (0, 1))
        self.assertEqual(parse_pytest_counts("no tests ran"), (0, 0))

    def test_parse_single_patch_block(self) -> None:
        got = parse_patch_blocks(patch_block("a/b.py", "old line", "new line"))
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].path, "a/b.py")
        self.assertEqual(got[0].search, "old line")
        self.assertEqual(got[0].replace, "new line")

    def test_parse_multiple_patches_in_order(self) -> None:
        text = "do:\n" + patch_block("m.py", "A", "B") + "\n" + patch_block("m.py", "C", "D")
        got = parse_patch_blocks(text)
        self.assertEqual([(p.search, p.replace) for p in got], [("A", "B"), ("C", "D")])

    def test_file_and_patch_parsers_do_not_overlap(self) -> None:
        text = block("new.py", "x = 1") + "\n" + patch_block("old.py", "foo", "bar")
        self.assertEqual(parse_file_blocks(text), {"new.py": "x = 1"})
        self.assertNotIn("old.py", parse_file_blocks(text))
        self.assertEqual([p.path for p in parse_patch_blocks(text)], ["old.py"])

    def test_parse_multiline_search_and_replace(self) -> None:
        got = parse_patch_blocks(
            patch_block("m.py", "def f():\n    return 1", "def f():\n    return 2")
        )
        self.assertEqual(got[0].search, "def f():\n    return 1")
        self.assertEqual(got[0].replace, "def f():\n    return 2")

    def test_no_patch_blocks(self) -> None:
        self.assertEqual(parse_patch_blocks(block("a.py", "x = 1")), [])

    # lenient fallback — recover near-miss formats instead of discarding a fix.
    def test_lenient_recovers_single_bracket_patch_with_inferred_path(self) -> None:
        # The exact shape observed live: single-angle <PATCH>, XML closers, no path.
        text = (
            "Here is the fix:\n<PATCH>\n<SEARCH>old line</SEARCH>\n"
            "<REPLACE>new line</REPLACE>\n</PATCH>\n"
        )
        files, patches = parse_blocks_lenient(text, default_path="pkg/a.py")
        self.assertEqual(files, {})
        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0].path, "pkg/a.py")
        self.assertEqual(patches[0].search, "old line")
        self.assertEqual(patches[0].replace, "new line")

    def test_lenient_pathless_patch_dropped_without_a_default(self) -> None:
        text = "<PATCH>\n<SEARCH>a</SEARCH>\n<REPLACE>b</REPLACE>\n</PATCH>"
        _files, patches = parse_blocks_lenient(text, default_path=None)
        self.assertEqual(patches, [])

    def test_lenient_recovers_xml_style_file_block(self) -> None:
        files, patches = parse_blocks_lenient("<FILE: a.py>\nx = 1\n</FILE>")
        self.assertEqual(files, {"a.py": "x = 1"})
        self.assertEqual(patches, [])

    def test_lenient_keeps_multiline_anchor_and_explicit_path(self) -> None:
        text = (
            "<PATCH: m.py>\n<SEARCH>def f():\n    return 1</SEARCH>\n"
            "<REPLACE>def f():\n    return 2</REPLACE>\n</PATCH>"
        )
        _files, patches = parse_blocks_lenient(text)
        self.assertEqual(patches[0].path, "m.py")
        self.assertEqual(patches[0].search, "def f():\n    return 1")
        self.assertEqual(patches[0].replace, "def f():\n    return 2")

    def test_lenient_finds_nothing_in_prose(self) -> None:
        self.assertEqual(parse_blocks_lenient("no blocks here", "a.py"), ({}, []))


class DiagnosticsDistillationTests(unittest.TestCase):
    """The diagnostics are the loop's senses (S19 fix).

    Observed live: a raw 800-char tail was all fast-check stack trace, the
    generator never saw the failing assertion, and the loop stagnated.
    """

    def test_keeps_failure_essence_and_drops_stack_noise(self) -> None:
        out = (
            " FAIL  tests/quantity.test.ts > formatQuantity > trims zeros\n"
            "AssertionError: expected '1.500' to be '1.5'\n"
            "- Expected\n+ Received\n"
            "Counterexample: [1n]\n"
            " ❯ runIt node_modules/fast-check/lib/fast-check.js:2484:24\n"
            " ❯ check node_modules/fast-check/lib/fast-check.js:2516:204\n"
            " Tests  4 failed | 58 passed (62)\n"
        )
        d = distill_diagnostics(out)
        self.assertIn("expected '1.500' to be '1.5'", d)
        self.assertIn("Counterexample: [1n]", d)
        self.assertIn("4 failed | 58 passed", d)
        self.assertNotIn("fast-check.js:2484", d)

    def test_falls_back_to_tail_when_nothing_matches(self) -> None:
        self.assertEqual(distill_diagnostics("x" * 2000), "x" * 800)

    def test_strips_ansi_colors(self) -> None:
        d = distill_diagnostics("\x1b[31mAssertionError: expected 1 to be 2\x1b[39m")
        self.assertIn("AssertionError: expected 1 to be 2", d)
        self.assertNotIn("\x1b", d)


class SafetyTests(unittest.TestCase):
    def test_safe_paths(self) -> None:
        self.assertTrue(is_safe_relpath("src/app.py"))
        self.assertTrue(is_safe_relpath("mathx.py"))

    def test_unsafe_paths(self) -> None:
        for bad in ("/etc/passwd", "../escape.py", "a/../../b.py", "", "a//b.py", "a\\b.py"):
            self.assertFalse(is_safe_relpath(bad), bad)

    def test_protected_test_files(self) -> None:
        for path in ("tests/test_x.py", "pkg/tests/helper.py", "test_app.py",
                     "src/app_test.py", "conftest.py", "src/conftest.py"):
            self.assertTrue(is_protected(path), path)
        self.assertFalse(is_protected("src/app.py"))
        self.assertTrue(is_protected("src/secret.py", ("src/secret.py",)))

    def test_protected_is_case_insensitive_without_overmatching(self) -> None:
        # Judge files must be protected regardless of case (a candidate must not
        # bypass the golden rule by uppercasing a path) — but whole segments and
        # patterns are compared, so look-alikes are NOT over-matched.
        for path in ("TESTS/foo.py", "A/TEST/b.py", "Conftest.PY",
                     "TEST_x.py", "src/App_TEST.py"):
            self.assertTrue(is_protected(path), path)
        for path in ("latest/x.py", "testing/b.py", "contest.py",
                     "mytest.py", "src/attest.py", "greatest.py"):
            self.assertFalse(is_protected(path), path)
        # extra globs match case-insensitively in both directions.
        self.assertTrue(is_protected("Foo/Bar.PY", ("foo/bar.py",)))
        self.assertTrue(is_protected("foo/bar.py", ("FOO/BAR.PY",)))

    def test_protected_config_files(self) -> None:
        # Test-runner / build config is the harness the candidate may not edit:
        # changing it (not the source) games the judge. Matched on basename,
        # case-insensitively, anywhere in the tree.
        for path in (
            "pyproject.toml", "pytest.ini", ".pytest.ini", "tox.ini", "setup.cfg",
            "pkg/pyproject.toml", "PyProject.TOML",
            "vitest.config.ts", "vite.config.js", "jest.config.cjs", ".mocharc.json",
            "foundry.toml", "slither.config.json",
        ):
            self.assertTrue(is_protected_config(path), path)
        # Source files (even config-ish names) and the dual-purpose package.json
        # are NOT rejected — only dedicated runner/build config is.
        for path in (
            "src/app.py", "config.py", "configuration.py", "settings.py",
            "package.json", "requirements.txt", "README.md",
            "pyproject.toml.bak", "src/setup.py",
        ):
            self.assertFalse(is_protected_config(path), path)


class RewardHackRejectionTests(unittest.TestCase):
    """Regression: the golden rule covers test *configuration*, not only tests.

    Demonstrated exploit (pre-fix): a candidate that never fixes the bug but adds
    ``[tool.pytest.ini_options] addopts = "-k <passing test>"`` to ``pyproject.toml``
    deselects the failing tests, so the suite reports a full pass (1/1) and the
    loop hands back a "fix" that changed no source. Editing any test-runner / build
    config is now rejected at the path gate, before the suite ever runs — so these
    tests need no pytest installed (the rejection is pre-subprocess).
    """

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="evo_rh_repo_")
        self.problem = {"name": "rh", "repo_path": self.root}
        self.v = RepoVerifier(timeout=30)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_pyproject_addopts_deselect_is_rejected(self) -> None:
        hack = block(
            "pyproject.toml",
            '[tool.pytest.ini_options]\naddopts = "-k passing_test"\n',
        )
        r = self.v.verify(hack, self.problem)
        self.assertFalse(r.passed)
        self.assertEqual(r.score, 0.05)
        self.assertIn("configuration", r.diagnostics)

    def test_pytest_ini_ignore_is_rejected(self) -> None:
        r = self.v.verify(
            block("pytest.ini", "[pytest]\naddopts = --ignore=tests\n"), self.problem
        )
        self.assertFalse(r.passed)
        self.assertEqual(r.score, 0.05)
        self.assertIn("configuration", r.diagnostics)

    def test_config_edit_via_surgical_patch_is_rejected(self) -> None:
        # The PATCH path goes through the same gate as whole-file blocks.
        r = self.v.verify(patch_block("setup.cfg", "old", "new"), self.problem)
        self.assertFalse(r.passed)
        self.assertEqual(r.score, 0.05)
        self.assertIn("configuration", r.diagnostics)

    def test_config_edit_alongside_a_real_source_fix_is_still_rejected(self) -> None:
        # Smuggling a config edit in with a legitimate-looking source block must
        # not slip through: any protected path in the change set rejects the run.
        hyp = block("mathx.py", "x = 1") + "\n" + block("tox.ini", "[pytest]\n")
        r = self.v.verify(hyp, self.problem)
        self.assertFalse(r.passed)
        self.assertEqual(r.score, 0.05)


class PackageJsonJudgeFieldsTests(unittest.TestCase):
    """package.json is dual-purpose, so it isn't rejected like a dedicated config
    file — instead its test-harness fields are restored from the pristine original,
    neutralising a JS-judge reward-hack while keeping legitimate edits.
    """

    @staticmethod
    def _pkg(**kw) -> str:
        return json.dumps(kw)

    def test_restores_narrowed_test_script(self) -> None:
        original = self._pkg(name="x", scripts={"test": "vitest run", "build": "tsc"})
        # candidate keeps the bug; narrows the test script to only-passing specs.
        candidate = self._pkg(
            name="x", scripts={"test": "vitest run -t passing", "build": "tsc"}
        )
        out = json.loads(restore_judge_package_json(original, candidate))
        self.assertEqual(out["scripts"]["test"], "vitest run")  # restored

    def test_keeps_legitimate_non_test_edits(self) -> None:
        original = self._pkg(scripts={"test": "vitest run"}, dependencies={"a": "1.0.0"})
        # candidate adds a dependency and a build script — both must survive.
        candidate = self._pkg(
            scripts={"test": "vitest run", "build": "tsc"},
            dependencies={"a": "1.0.0", "b": "2.0.0"},
        )
        # nothing judging changed → byte-for-byte unchanged.
        self.assertEqual(restore_judge_package_json(original, candidate), candidate)

    def test_restores_embedded_runner_config(self) -> None:
        original = self._pkg(scripts={"test": "jest"}, jest={"testMatch": ["**/*.test.js"]})
        candidate = self._pkg(
            scripts={"test": "jest"}, jest={"testMatch": ["**/passing.test.js"]}
        )
        out = json.loads(restore_judge_package_json(original, candidate))
        self.assertEqual(out["jest"], {"testMatch": ["**/*.test.js"]})  # restored

    def test_strips_harness_when_original_had_none(self) -> None:
        # repo had no package.json; a created one must not introduce a test harness.
        candidate = self._pkg(scripts={"test": "echo fake && exit 0"}, jest={"x": 1})
        out = json.loads(restore_judge_package_json(None, candidate))
        self.assertNotIn("jest", out)
        self.assertNotIn("test", out.get("scripts", {}))

    def test_restores_removed_test_script(self) -> None:
        original = self._pkg(scripts={"test": "vitest run"})
        candidate = self._pkg(scripts={"build": "tsc"})  # dropped the test script
        out = json.loads(restore_judge_package_json(original, candidate))
        self.assertEqual(out["scripts"]["test"], "vitest run")

    def test_malformed_candidate_is_left_untouched(self) -> None:
        self.assertEqual(restore_judge_package_json("{}", "not json {"), "not json {")

    def test_wired_into_apply_blocks(self) -> None:
        # End-to-end through apply_blocks_to_copy (pure filesystem, no node needed).
        copy = tempfile.mkdtemp(prefix="evo_pkg_")
        try:
            with open(os.path.join(copy, "package.json"), "w", encoding="utf-8") as f:
                f.write(self._pkg(scripts={"test": "vitest run", "build": "tsc"}))
            hack = self._pkg(scripts={"test": "vitest run -t passing", "build": "tsc"})
            err = apply_blocks_to_copy(copy, {"package.json": hack}, [])
            self.assertIsNone(err)
            with open(os.path.join(copy, "package.json"), encoding="utf-8") as f:
                written = json.load(f)
            self.assertEqual(written["scripts"]["test"], "vitest run")  # neutralised
            self.assertEqual(written["scripts"]["build"], "tsc")        # legit kept
        finally:
            shutil.rmtree(copy, ignore_errors=True)


@unittest.skipUnless(HAS_PYTEST, "pytest not installed")
class RepoVerifierEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="evo_test_repo_")
        make_repo(self.root)
        self.problem = {"name": "fix_mathx", "repo_path": self.root}
        self.v = RepoVerifier(timeout=60)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_full_fix_passes(self) -> None:
        r = self.v.verify(block("mathx.py", FIXED_MATHX), self.problem)
        self.assertTrue(r.passed)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(r.artifact["tests_passed"], r.artifact["tests_total"])

    def test_partial_fix_scores_between(self) -> None:
        r = self.v.verify(block("mathx.py", HALF_FIXED_MATHX), self.problem)
        self.assertFalse(r.passed)
        self.assertGreater(r.score, 0.25)
        self.assertLess(r.score, 1.0)
        self.assertEqual(r.artifact["tests_passed"], 2)

    def test_gradient_orders_candidates(self) -> None:
        full = self.v.verify(block("mathx.py", FIXED_MATHX), self.problem).score
        half = self.v.verify(block("mathx.py", HALF_FIXED_MATHX), self.problem).score
        broken = self.v.verify(block("mathx.py", "syntax error ((("), self.problem).score
        none = self.v.verify("no blocks here", self.problem).score
        self.assertGreater(full, half)
        self.assertGreater(half, broken)
        self.assertGreater(broken, none)

    def test_source_repo_never_modified(self) -> None:
        before = open(os.path.join(self.root, "mathx.py"), encoding="utf-8").read()
        self.v.verify(block("mathx.py", FIXED_MATHX), self.problem)
        after = open(os.path.join(self.root, "mathx.py"), encoding="utf-8").read()
        self.assertEqual(before, after)

    def test_cannot_edit_the_judge(self) -> None:
        cheat = block("tests/test_mathx.py", "def test_nothing(): pass")
        r = self.v.verify(cheat, self.problem)
        self.assertFalse(r.passed)
        self.assertEqual(r.score, 0.05)
        self.assertIn("forbidden", r.diagnostics)

    def test_path_escape_rejected(self) -> None:
        r = self.v.verify(block("../outside.py", "x = 1"), self.problem)
        self.assertFalse(r.passed)
        self.assertEqual(r.score, 0.05)

    def test_new_file_can_be_created(self) -> None:
        hyp = block("mathx.py", FIXED_MATHX) + "\n" + block("helpers/extra.py", "Y = 2")
        r = self.v.verify(hyp, self.problem)
        self.assertTrue(r.passed)
        self.assertIn("helpers/extra.py", r.artifact["files_changed"])

    def test_collection_error_scores_low_but_above_rejection(self) -> None:
        # Removing the module the tests import breaks collection: no test ran.
        r = self.v.verify(block("mathx.py", "raise RuntimeError('boom')"), self.problem)
        self.assertFalse(r.passed)
        self.assertLessEqual(r.score, 0.10)
        self.assertGreater(r.score, 0.05)

    def test_missing_repo_path_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.v.verify(block("a.py", "x"), {"name": "x", "repo_path": "/nonexistent_xyz"})

    # surgical PATCH edits (issue #15) ──────────────────────
    def _full_fix_patches(self) -> str:
        return (
            patch_block(
                "mathx.py",
                "return sum(nums) / (len(nums) + 1)",
                "return sum(nums) / len(nums)",
            )
            + "\n"
            + patch_block("mathx.py", "return min(x, hi)", "return max(lo, min(x, hi))")
        )

    def test_patch_fix_passes(self) -> None:
        r = self.v.verify(self._full_fix_patches(), self.problem)
        self.assertTrue(r.passed)
        self.assertEqual(r.score, 1.0)
        self.assertIn("mathx.py", r.artifact["files_changed"])

    def test_mixed_file_and_patch_blocks(self) -> None:
        hyp = self._full_fix_patches() + "\n" + block("helpers/extra.py", "Y = 2")
        r = self.v.verify(hyp, self.problem)
        self.assertTrue(r.passed)
        self.assertIn("helpers/extra.py", r.artifact["files_changed"])
        self.assertIn("mathx.py", r.artifact["files_changed"])

    def test_patch_keeps_source_repo_unmodified(self) -> None:
        before = open(os.path.join(self.root, "mathx.py"), encoding="utf-8").read()
        self.v.verify(self._full_fix_patches(), self.problem)
        after = open(os.path.join(self.root, "mathx.py"), encoding="utf-8").read()
        self.assertEqual(before, after)

    def test_patch_cannot_edit_the_judge(self) -> None:
        cheat = patch_block("tests/test_mathx.py", "assert average([2, 4]) == 3", "assert True")
        r = self.v.verify(cheat, self.problem)
        self.assertFalse(r.passed)
        self.assertEqual(r.score, 0.05)
        self.assertIn("forbidden", r.diagnostics)

    def test_patch_no_match_yields_diagnostic(self) -> None:
        r = self.v.verify(patch_block("mathx.py", "anchor not present", "x"), self.problem)
        self.assertFalse(r.passed)
        self.assertEqual(r.score, 0.08)
        self.assertIn("NoMatchError", r.diagnostics)

    def test_patch_ambiguous_anchor_yields_diagnostic(self) -> None:
        # "    return" begins all three functions — not a unique anchor.
        r = self.v.verify(patch_block("mathx.py", "    return", "    return  # x"), self.problem)
        self.assertFalse(r.passed)
        self.assertEqual(r.score, 0.08)
        self.assertIn("AmbiguousMatchError", r.diagnostics)

    def test_patch_missing_target_yields_diagnostic(self) -> None:
        r = self.v.verify(patch_block("nope.py", "x", "y"), self.problem)
        self.assertFalse(r.passed)
        self.assertEqual(r.score, 0.08)
        self.assertIn("not found", r.diagnostics)

    def test_lenient_format_is_recovered_end_to_end(self) -> None:
        # Regression: a model emitted single-angle <PATCH> blocks with XML closers
        # and no file path — a correct, winning fix that the strict parser would
        # discard as "no parseable blocks" (score 0.02). With the target file
        # known, the verifier now recovers and applies it for a real verdict.
        hyp = (
            "Here is the fix:\n"
            "<PATCH>\n<SEARCH>def average(nums):\n"
            "    return sum(nums) / (len(nums) + 1)</SEARCH>\n"
            "<REPLACE>def average(nums):\n    return sum(nums) / len(nums)</REPLACE>\n"
            "</PATCH>\n"
            "<PATCH>\n<SEARCH>    return min(x, hi)</SEARCH>\n"
            "<REPLACE>    return max(lo, min(x, hi))</REPLACE>\n</PATCH>\n"
        )
        r = self.v.verify(hyp, {**self.problem, "target_files": ["mathx.py"]})
        self.assertTrue(r.passed)
        self.assertEqual(r.score, 1.0)
        self.assertIn("mathx.py", r.artifact["files_changed"])

    def test_unparseable_output_still_scores_floor(self) -> None:
        # No recoverable structure anywhere → still the 0.02 floor (unchanged).
        r = self.v.verify("just some prose, no edits", self.problem)
        self.assertFalse(r.passed)
        self.assertEqual(r.score, 0.02)


class JUnitOracleTests(unittest.TestCase):
    """The verdict is read from the structured JUnit report + exit code, never
    from stdout — so a forged ``"9999 passed"`` summary moves nothing. These are
    pure-function tests (no subprocess), so they run without pytest installed.
    """

    def test_counts_come_from_structured_xml(self) -> None:
        # total excludes skipped (5-1=4); passed = 4 - 1 failure - 1 error = 2.
        j = parse_junit_xml(
            '<testsuites><testsuite tests="5" failures="1" errors="1" skipped="1"/></testsuites>'
        )
        assert j is not None
        self.assertEqual((j.passed, j.total, j.failures, j.errors), (2, 4, 1, 1))

    def test_a_forged_stdout_summary_yields_no_counts(self) -> None:
        # A fake "9999 passed" is plain text, not <testsuite> structure.
        self.assertIsNone(parse_junit_xml("\n====== 9999 passed in 0.01s ======\n"))
        self.assertIsNone(parse_junit_xml(""))
        self.assertIsNone(parse_junit_xml("<not-valid-xml"))

    def test_full_pass_requires_exit0_and_a_clean_report(self) -> None:
        j = parse_junit_xml('<testsuite tests="3" failures="0" errors="0" skipped="0"/>')
        self.assertEqual(grade_repo_run(0, j, is_pytest=True), (True, 1.0, 3, 3))

    def test_partial_score_uses_the_report_fraction(self) -> None:
        j = parse_junit_xml('<testsuite tests="3" failures="1" errors="0" skipped="0"/>')
        passed, score, p, t = grade_repo_run(1, j, is_pytest=True)
        self.assertFalse(passed)
        self.assertEqual((p, t), (2, 3))
        self.assertAlmostEqual(score, fraction_score(2, 3))

    def test_exit0_with_a_failing_report_is_rejected_as_tamper(self) -> None:
        # Exit code claims success but the report shows failures → no clean verdict.
        j = parse_junit_xml('<testsuite tests="3" failures="2" errors="0" skipped="0"/>')
        self.assertEqual(grade_repo_run(0, j, is_pytest=True), (False, 0.10, 1, 3))

    def test_allpass_report_with_nonzero_exit_is_rejected_as_tamper(self) -> None:
        # Report claims all-pass but the process failed → not rewarded.
        j = parse_junit_xml('<testsuite tests="3" failures="0" errors="0" skipped="0"/>')
        passed, score, _, _ = grade_repo_run(1, j, is_pytest=True)
        self.assertFalse(passed)
        self.assertEqual(score, 0.10)

    def test_pytest_without_a_report_is_no_verdict(self) -> None:
        self.assertEqual(grade_repo_run(0, None, is_pytest=True), (False, 0.10, 0, 0))

    def test_custom_runner_falls_back_to_exit_code(self) -> None:
        self.assertEqual(grade_repo_run(0, None, is_pytest=False), (True, 1.0, 0, 0))
        self.assertFalse(grade_repo_run(1, None, is_pytest=False)[0])
        self.assertEqual(grade_repo_run(2, None, is_pytest=False), (False, 0.10, 0, 0))


# A reward-hack attempt: forge a passing summary on the judge's stdout AND stderr
# at import time (so it runs the moment pytest imports the module under test). The
# bytes are written straight to the file descriptors, bypassing any capture.
_STDOUT_FORGERY = (
    "\n\nimport os as _os\n"
    "for _fd in (1, 2):\n"
    "    try:\n"
    "        _os.write(_fd, b'\\n=========== 9999 passed in 0.01s ===========\\n')\n"
    "    except OSError:\n"
    "        pass\n"
)


class JudgeOracleRedTeamTests(unittest.TestCase):
    """Adversarial: prove a candidate cannot forge a pass or inflate the score."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="evo_redteam_repo_")
        make_repo(self.root)
        self.problem = {"name": "redteam", "repo_path": self.root}
        self.v = RepoVerifier(timeout=60)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_autoexec_and_runner_files_are_rejected_at_the_gate(self) -> None:
        # No subprocess runs: these are rejected before the suite, so this needs
        # no pytest. Each would otherwise run code inside the judge / redefine it.
        for path in (
            "sitecustomize.py", "usercustomize.py", "evil.pth", "pkg/sub/x.pth",
            "Makefile", "GNUmakefile", "noxfile.py",
        ):
            r = self.v.verify(block(path, "import os; os._exit(0)"), self.problem)
            self.assertFalse(r.passed, path)
            self.assertEqual(r.score, 0.05, path)
        # Direct predicate checks, case-insensitive, without over-matching.
        self.assertTrue(is_judge_autoexec("SiteCustomize.PY"))
        self.assertTrue(is_judge_autoexec("a/b/plugin.pth"))
        self.assertFalse(is_judge_autoexec("customize.py"))
        self.assertFalse(is_judge_autoexec("path.py"))

    @unittest.skipUnless(HAS_PYTEST, "needs pytest to run the suite")
    def test_stdout_injection_cannot_inflate_the_partial_score(self) -> None:
        # average is fixed (test passes) but clamp stays broken (test fails) → an
        # honest 2/3. The candidate forges "9999 passed" on stdout/stderr; the old
        # stdout-scraping judge would have summed it to ~10001/10002 ≈ 0.95 (the
        # ceiling). The JUnit-oracle judge reports the true 2/3.
        candidate = block("mathx.py", HALF_FIXED_MATHX + _STDOUT_FORGERY)
        r = self.v.verify(candidate, self.problem)
        self.assertFalse(r.passed)
        self.assertEqual((r.artifact["tests_passed"], r.artifact["tests_total"]), (2, 3))
        self.assertAlmostEqual(r.score, fraction_score(2, 3))
        self.assertLess(r.score, 0.9)  # NOT inflated toward the 0.95 ceiling
        self.assertEqual(r.artifact["verdict_source"], "junit+exit")

    @unittest.skipUnless(HAS_PYTEST, "needs pytest to run the suite")
    def test_stdout_injection_cannot_forge_a_full_pass(self) -> None:
        # Nothing fixed (2 tests fail) but a forged all-pass summary is printed.
        candidate = block("mathx.py", BUGGY_MATHX + _STDOUT_FORGERY)
        r = self.v.verify(candidate, self.problem)
        self.assertFalse(r.passed)
        self.assertLess(r.score, 1.0)
        self.assertEqual(r.artifact["tests_passed"], 1)  # only double() passes

    @unittest.skipUnless(HAS_PYTEST, "needs pytest to run the suite")
    def test_candidate_cannot_plant_a_fake_junit_report(self) -> None:
        # The candidate fixes nothing and plants a fake all-pass JUnit file at the
        # guessable basename. The judge reads its OWN report (written outside the
        # repo copy), so the plant — landing inside the copy — is inert.
        fake = (
            '<?xml version="1.0"?><testsuites><testsuite name="x" '
            'tests="3" failures="0" errors="0" skipped="0"/></testsuites>'
        )
        hyp = block("mathx.py", BUGGY_MATHX) + "\n" + block("judge-result.xml", fake)
        r = self.v.verify(hyp, self.problem)
        self.assertFalse(r.passed)
        self.assertLess(r.score, 1.0)
        self.assertEqual(r.artifact["tests_passed"], 1)

    @unittest.skipUnless(HAS_PYTEST, "needs pytest to run the suite")
    def test_an_honest_full_fix_still_passes(self) -> None:
        # Control: the oracle must not be so strict it rejects a genuine fix.
        r = self.v.verify(block("mathx.py", FIXED_MATHX), self.problem)
        self.assertTrue(r.passed)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(r.artifact["verdict_source"], "junit+exit")


if __name__ == "__main__":
    unittest.main()
