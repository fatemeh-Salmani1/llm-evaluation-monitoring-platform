from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.evaluation.batch_runner import (
    BatchEvaluationSummary,
)
from src.monitoring.comparison import (
    ComparisonThresholds,
    compare_summaries,
    compare_summary_files,
    load_batch_summary,
)


def create_summary(
    run_id: str,
    success_rate: float = 1.0,
    retrieval_recall: float = 1.0,
    fact_coverage: float = 1.0,
    citation_validity: float = 1.0,
    overall_score: float = 1.0,
    duration_ms: float = 1000.0,
) -> BatchEvaluationSummary:
    """Create a benchmark summary for comparison tests."""

    successful_cases = round(success_rate * 8)
    failed_cases = 8 - successful_cases

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
        successful_cases=successful_cases,
        failed_cases=failed_cases,
        success_rate=success_rate,
        average_duration_ms=duration_ms,
        average_retrieval_recall=retrieval_recall,
        average_fact_coverage=fact_coverage,
        average_citation_validity=citation_validity,
        average_overall_score=overall_score,
        failed_case_ids=[
            f"eval-{number:04d}"
            for number in range(
                successful_cases + 1,
                9,
            )
        ],
        embedding_model="text-embedding-3-small",
        generation_model="gpt-5.6-luna",
        top_k=4,
    )


def test_compare_summaries_identifies_improvements() -> None:
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

    comparison = compare_summaries(
        baseline=baseline,
        current=current,
    )

    assert comparison.has_regression is False
    assert comparison.regressions == []
    assert comparison.retrieval_recall.absolute_change == (
        pytest.approx(0.125)
    )
    assert comparison.overall_score.absolute_change == (
        pytest.approx(0.0729)
    )
    assert "retrieval_recall" in comparison.improvements
    assert "fact_coverage" in comparison.improvements
    assert "overall_score" in comparison.improvements
    assert "average_duration_ms" in comparison.improvements


def test_compare_summaries_identifies_regressions() -> None:
    baseline = create_summary(
        run_id="baseline-run",
        success_rate=1.0,
        retrieval_recall=1.0,
        overall_score=0.95,
        duration_ms=1000.0,
    )
    current = create_summary(
        run_id="current-run",
        success_rate=0.75,
        retrieval_recall=0.75,
        overall_score=0.80,
        duration_ms=1400.0,
    )

    comparison = compare_summaries(
        baseline=baseline,
        current=current,
    )

    assert comparison.has_regression is True
    assert comparison.regressions == [
        "success_rate",
        "retrieval_recall",
        "overall_score",
        "average_duration_ms",
    ]


def test_compare_summaries_allows_changes_within_thresholds() -> None:
    baseline = create_summary(
        run_id="baseline-run",
        retrieval_recall=1.0,
        overall_score=1.0,
        duration_ms=1000.0,
    )
    current = create_summary(
        run_id="current-run",
        retrieval_recall=0.99,
        overall_score=0.99,
        duration_ms=1200.0,
    )

    comparison = compare_summaries(
        baseline=baseline,
        current=current,
        thresholds=ComparisonThresholds(),
    )

    assert comparison.has_regression is False
    assert comparison.regressions == []


def test_compare_summary_files_loads_saved_summaries(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"

    baseline_path.write_text(
        create_summary(
            run_id="baseline-run",
            overall_score=0.8,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    current_path.write_text(
        create_summary(
            run_id="current-run",
            overall_score=0.9,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    comparison = compare_summary_files(
        baseline_path=baseline_path,
        current_path=current_path,
    )

    assert comparison.baseline_run_id == "baseline-run"
    assert comparison.current_run_id == "current-run"
    assert comparison.overall_score.absolute_change == (
        pytest.approx(0.1)
    )


def test_load_batch_summary_rejects_invalid_file(
    tmp_path: Path,
) -> None:
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(
        '{"run_id": "missing-fields"}',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid benchmark summary",
    ):
        load_batch_summary(invalid_path)


def test_load_batch_summary_rejects_missing_file(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.json"

    with pytest.raises(
        FileNotFoundError,
        match="Benchmark summary does not exist",
    ):
        load_batch_summary(missing_path)