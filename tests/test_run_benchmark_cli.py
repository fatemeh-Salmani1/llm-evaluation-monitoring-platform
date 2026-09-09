from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.evaluation.batch_runner import (
    BatchEvaluationResult,
    BatchEvaluationSummary,
)
from src.evaluation.models import (
    BenchmarkCase,
    Difficulty,
    EvaluationCategory,
)
from src.evaluation.run_benchmark import (
    format_batch_result,
    format_batch_summary,
    select_benchmark_cases,
)


def create_case(
    case_id: str,
) -> BenchmarkCase:
    """Create a benchmark case for CLI tests."""

    return BenchmarkCase(
        case_id=case_id,
        question="What two ingredients does an eval need?",
        reference_answer=(
            "An eval needs data_source_config "
            "and testing_criteria."
        ),
        category=EvaluationCategory.EVAL_CONCEPTS,
        difficulty=Difficulty.EASY,
        required_facts=[
            "data_source_config",
            "testing_criteria",
        ],
        expected_source_ids=["openai-evals-guide"],
        expected_chunk_ids=[
            "openai-evals-guide-chunk-0004"
        ],
        tags=["evals"],
    )


def create_summary() -> BatchEvaluationSummary:
    """Create a batch summary for formatting tests."""

    started_at = datetime(
        2026,
        9,
        9,
        10,
        0,
        tzinfo=UTC,
    )
    completed_at = datetime(
        2026,
        9,
        9,
        10,
        1,
        tzinfo=UTC,
    )

    return BatchEvaluationSummary(
        run_id="test-run-001",
        started_at=started_at,
        completed_at=completed_at,
        total_cases=2,
        successful_cases=1,
        failed_cases=1,
        success_rate=0.5,
        average_duration_ms=1250.5,
        average_retrieval_recall=1.0,
        average_fact_coverage=0.75,
        average_citation_validity=1.0,
        average_overall_score=0.9375,
        failed_case_ids=["eval-0002"],
        embedding_model="text-embedding-3-small",
        generation_model="gpt-5.6-luna",
        top_k=3,
    )


def test_select_benchmark_cases_returns_all_cases() -> None:
    cases = [
        create_case("eval-0001"),
        create_case("eval-0002"),
    ]

    selected = select_benchmark_cases(
        cases=cases,
        limit=None,
    )

    assert selected == cases


def test_select_benchmark_cases_applies_limit() -> None:
    cases = [
        create_case("eval-0001"),
        create_case("eval-0002"),
        create_case("eval-0003"),
    ]

    selected = select_benchmark_cases(
        cases=cases,
        limit=2,
    )

    assert [case.case_id for case in selected] == [
        "eval-0001",
        "eval-0002",
    ]


def test_select_benchmark_cases_rejects_invalid_limit() -> None:
    with pytest.raises(
        ValueError,
        match="Benchmark limit must be greater than zero",
    ):
        select_benchmark_cases(
            cases=[create_case("eval-0001")],
            limit=0,
        )


def test_format_batch_summary_includes_metrics() -> None:
    formatted = format_batch_summary(
        create_summary()
    )

    assert "Run ID: test-run-001" in formatted
    assert "Total cases: 2" in formatted
    assert "Success rate: 50.00%" in formatted
    assert "Average retrieval recall: 1.0000" in formatted
    assert "Average fact coverage: 0.7500" in formatted
    assert "Average overall score: 0.9375" in formatted
    assert "Failed case IDs: eval-0002" in formatted


def test_format_batch_result_includes_output_paths() -> None:
    result = BatchEvaluationResult(
        records=[],
        summary=create_summary(),
        records_path=Path(
            "data/processed/evaluation_runs/"
            "test-run-001.jsonl"
        ),
        summary_path=Path(
            "data/processed/evaluation_runs/"
            "test-run-001.summary.json"
        ),
    )

    formatted = format_batch_result(result)

    assert "Records: data/processed/evaluation_runs/" in formatted
    assert "test-run-001.jsonl" in formatted
    assert "Summary: data/processed/evaluation_runs/" in formatted
    assert "test-run-001.summary.json" in formatted