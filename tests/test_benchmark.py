from pathlib import Path

from benchmarks.evaluate import evaluate


def test_sample_benchmark_has_no_classification_errors() -> None:
    result = evaluate(Path(__file__).parents[1] / "benchmarks" / "sample.jsonl")
    assert result["tp"] == 3
    assert result["tn"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0


def test_published_live_results_match_their_labels() -> None:
    # benchmarks/results.jsonl is the COMMITTED observed run (real guard()
    # verdicts, produced by benchmarks/run_live.py). Its published metrics must
    # stay true: zero missed hacks, and exactly one false positive — the
    # documented-by-design legit-dependency-bump policy trip.
    path = Path(__file__).parents[1] / "benchmarks" / "results.jsonl"
    result = evaluate(path)
    assert result["fn"] == 0, "a published hack case is no longer blocked"
    assert result["fp"] == 1, "the FP count changed — update benchmarks/README.md"
    assert result["total"] == 16


def test_live_benchmark_reproduces_the_published_corpus(tmp_path) -> None:
    # Run the REAL harness end-to-end (16 live guard() runs, a few seconds) and
    # require every case to land on its expected verdict — so the published
    # numbers can never drift from what the engine actually does.
    from benchmarks.run_live import run_corpus

    out = tmp_path / "live.jsonl"
    assert run_corpus(str(out)) == 0
    fresh = evaluate(out)
    committed = evaluate(Path(__file__).parents[1] / "benchmarks" / "results.jsonl")
    assert fresh == committed
