from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.evaluation.batch_runner import (
    BatchEvaluationSummary,
)
from src.evaluation.runner import EvaluationRunRecord
from src.monitoring.comparison import load_batch_summary

HISTORY_COLUMNS = [
    "run_id",
    "completed_at",
    "total_cases",
    "successful_cases",
    "failed_cases",
    "success_rate",
    "retrieval_recall",
    "fact_coverage",
    "citation_validity",
    "overall_score",
    "judge_score",
    "average_duration_ms",
    "embedding_model",
    "generation_model",
    "judge_model",
    "top_k",
]

CASE_COLUMNS = [
    "case_id",
    "status",
    "question",
    "category",
    "difficulty",
    "duration_ms",
    "retrieval_recall",
    "fact_coverage",
    "citation_validity",
    "deterministic_score",
    "judge_score",
    "judge_relevance",
    "judge_completeness",
    "judge_groundedness",
    "judge_clarity",
    "judge_reasoning",
    "error_message",
]


def discover_summary_paths(
    output_directory: Path,
) -> list[Path]:
    """Discover stored benchmark summary files."""

    if not output_directory.exists():
        return []

    if not output_directory.is_dir():
        raise NotADirectoryError(
            f"Evaluation output is not a directory: "
            f"{output_directory}"
        )

    return sorted(
        output_directory.glob("*.summary.json")
    )


def load_run_history(
    output_directory: Path,
) -> list[BatchEvaluationSummary]:
    """Load benchmark summaries ordered by completion time."""

    summaries = [
        load_batch_summary(path)
        for path in discover_summary_paths(
            output_directory
        )
    ]

    return sorted(
        summaries,
        key=lambda summary: summary.completed_at,
    )


def load_evaluation_records(
    records_path: Path,
) -> list[EvaluationRunRecord]:
    """Load and validate case records from a JSONL file."""

    if not records_path.exists():
        raise FileNotFoundError(
            f"Evaluation records do not exist: "
            f"{records_path}"
        )

    records: list[EvaluationRunRecord] = []

    for line_number, line in enumerate(
        records_path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            record = (
                EvaluationRunRecord.model_validate_json(
                    line
                )
            )
        except ValueError as error:
            raise ValueError(
                "Invalid evaluation record at "
                f"line {line_number}: {records_path}"
            ) from error

        records.append(record)

    if not records:
        raise ValueError(
            f"Evaluation records are empty: {records_path}"
        )

    return records


def build_history_frame(
    summaries: Sequence[BatchEvaluationSummary],
) -> pd.DataFrame:
    """Convert benchmark summaries into a dashboard table."""

    rows = [
        {
            "run_id": summary.run_id,
            "completed_at": summary.completed_at,
            "total_cases": summary.total_cases,
            "successful_cases": summary.successful_cases,
            "failed_cases": summary.failed_cases,
            "success_rate": summary.success_rate,
            "retrieval_recall": (
                summary.average_retrieval_recall
            ),
            "fact_coverage": (
                summary.average_fact_coverage
            ),
            "citation_validity": (
                summary.average_citation_validity
            ),
            "overall_score": (
                summary.average_overall_score
            ),
            "judge_score": (
                summary.average_judge_score
            ),
            "average_duration_ms": (
                summary.average_duration_ms
            ),
            "embedding_model": (
                summary.embedding_model
            ),
            "generation_model": (
                summary.generation_model
            ),
            "judge_model": summary.judge_model,
            "top_k": summary.top_k,
        }
        for summary in summaries
    ]

    return pd.DataFrame(
        rows,
        columns=HISTORY_COLUMNS,
    )


def build_case_frame(
    records: Sequence[EvaluationRunRecord],
) -> pd.DataFrame:
    """Convert evaluation case records into a dashboard table."""

    rows = []

    for record in records:
        metrics = record.metrics
        judge = record.judge_result

        rows.append(
            {
                "case_id": record.case_id,
                "status": record.status.value,
                "question": record.question,
                "category": record.category.value,
                "difficulty": record.difficulty.value,
                "duration_ms": record.duration_ms,
                "retrieval_recall": (
                    metrics.retrieval_recall
                    if metrics is not None
                    else None
                ),
                "fact_coverage": (
                    metrics.fact_coverage
                    if metrics is not None
                    else None
                ),
                "citation_validity": (
                    metrics.citation_validity
                    if metrics is not None
                    else None
                ),
                "deterministic_score": (
                    metrics.overall_score
                    if metrics is not None
                    else None
                ),
                "judge_score": (
                    judge.overall_score
                    if judge is not None
                    else None
                ),
                "judge_relevance": (
                    judge.relevance
                    if judge is not None
                    else None
                ),
                "judge_completeness": (
                    judge.completeness
                    if judge is not None
                    else None
                ),
                "judge_groundedness": (
                    judge.groundedness
                    if judge is not None
                    else None
                ),
                "judge_clarity": (
                    judge.clarity
                    if judge is not None
                    else None
                ),
                "judge_reasoning": (
                    judge.reasoning
                    if judge is not None
                    else None
                ),
                "error_message": record.error_message,
            }
        )

    return pd.DataFrame(
        rows,
        columns=CASE_COLUMNS,
    )


def records_path_for_run(
    output_directory: Path,
    run_id: str,
) -> Path:
    """Build the records path for a validated run identifier."""

    if not run_id:
        raise ValueError("Run ID cannot be empty")

    if Path(run_id).name != run_id:
        raise ValueError(
            "Run ID contains unsupported path characters"
        )

    return output_directory / f"{run_id}.jsonl"