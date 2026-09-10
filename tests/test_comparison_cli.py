from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.evaluation.batch_runner import (
    BatchEvaluationSummary,
)
from src.monitoring.comparison import compare_summaries
from src.monitoring.run_comparison import (
    build_argument_parser,
    format_run_comparison,
    main,
)


def create_summary(
    run_id: str,
    retrieval_recall: float = 1.0,
    fact_coverage: float = 1.0,
    overall_score: float = 1.0,
    judge_score: float | None = None,
    duration_ms: float = 1000.0,
) -> BatchEvaluationSummary:
    """Create a summary for comparison-CLI tests."""

    return BatchEvaluationSummary(
        run_id=run_id,
        started_at=datetime(
            2026,
            9,
            10,
            9,
            0,
            tzinfo=UTC,
        ),
        completed_at=datetime(
            2026,
            9,
            10,
            9,
            1,
            tzinfo=UTC,
        ),
        total_cases=8,
        successful_cases=8,
        failed_cases=0,
        success_rate=1.0,
        average_duration_ms=duration_ms,
        average_retrieval_recall=retrieval_recall,
        average_fact_coverage=fact_coverage,
        average_citation_validity=1.0,
        average_overall_score=overall_score,
        average_judge_score=judge_score,
        failed_case_ids=[],
        embedding_model="text-embedding-3-small",
        generation_model="gpt-5.6-luna",
        judge_model=(
            "gpt-5.6-luna"
            if judge_score is not None
            else None
        ),
        top_k=4,
    )


def write_summary(
    path: Path,
    summary: BatchEvaluationSummary,
) -> None:
    """Write a validated summary to JSON."""

    path.write_text(
        summary.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def test_format_run_comparison_shows_improvements() -> None:
    baseline = create_summary(
        run_id="baseline-run",
        retrieval_recall=0.875,
        fact_coverage=0.7708,
        overall_score=0.8802,
        duration_ms=3004.86,
    )
    current = create_summary(
        run_id="current-run",
        retrieval_recall=1.0,
        fact_coverage=0.8125,
        overall_score=0.9531,
        duration_ms=2274.52,
    )

    formatted = format_run_comparison(
        compare_summaries(
            baseline=baseline,
            current=current,
        )
    )

    assert "Baseline run: baseline-run" in formatted
    assert "Current run: current-run" in formatted
    assert "Retrieval recall: 0.8750 -> 1.0000" in formatted
    assert "Overall score: 0.8802 -> 0.9531" in formatted
    assert "LLM judge score" not in formatted
    assert "Regressions: None" in formatted
    assert "Quality gate: PASSED" in formatted


def test_format_run_comparison_shows_judge_score() -> None:
    baseline = create_summary(
        run_id="baseline-run",
        judge_score=0.85,
    )
    current = create_summary(
        run_id="current-run",
        judge_score=0.95,
    )

    formatted = format_run_comparison(
        compare_summaries(
            baseline=baseline,
            current=current,
        )
    )

    assert "LLM judge score: 0.8500 -> 0.9500" in formatted
    assert "judge_score" in formatted
    assert "Quality gate: PASSED" in formatted


def test_argument_parser_accepts_judge_threshold() -> None:
    arguments = build_argument_parser().parse_args(
        [
            "baseline.json",
            "current.json",
            "--judge-score-drop",
            "0.10",
        ]
    )

    assert arguments.judge_score_drop == pytest.approx(0.10)


def test_main_returns_zero_when_quality_gate_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"

    write_summary(
        baseline_path,
        create_summary(
            run_id="baseline-run",
            overall_score=0.8,
        ),
    )
    write_summary(
        current_path,
        create_summary(
            run_id="current-run",
            overall_score=0.9,
        ),
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_comparison",
            str(baseline_path),
            str(current_path),
        ],
    )

    exit_code = main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Quality gate: PASSED" in output


def test_main_returns_one_for_judge_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"

    write_summary(
        baseline_path,
        create_summary(
            run_id="baseline-run",
            judge_score=0.95,
        ),
    )
    write_summary(
        current_path,
        create_summary(
            run_id="current-run",
            judge_score=0.75,
        ),
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_comparison",
            str(baseline_path),
            str(current_path),
        ],
    )

    exit_code = main()
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "judge_score" in output
    assert "Quality gate: FAILED" in output


def test_main_returns_one_when_regression_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"

    write_summary(
        baseline_path,
        create_summary(
            run_id="baseline-run",
            retrieval_recall=1.0,
            overall_score=1.0,
            duration_ms=1000.0,
        ),
    )
    write_summary(
        current_path,
        create_summary(
            run_id="current-run",
            retrieval_recall=0.7,
            overall_score=0.7,
            duration_ms=1500.0,
        ),
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_comparison",
            str(baseline_path),
            str(current_path),
        ],
    )

    exit_code = main()
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "retrieval_recall" in output
    assert "overall_score" in output
    assert "average_duration_ms" in output
    assert "Quality gate: FAILED" in output