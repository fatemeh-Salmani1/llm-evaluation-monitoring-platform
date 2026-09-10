import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from src.evaluation.judge import DEFAULT_JUDGE_MODEL
from src.evaluation.models import BenchmarkCase
from src.evaluation.runner import (
    EvaluationRunRecord,
    EvaluationStatus,
    run_evaluation_case,
)
from src.generation.answerer import (
    DEFAULT_GENERATION_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
)
from src.retrieval.embedder import DEFAULT_EMBEDDING_MODEL
from src.retrieval.run_retrieval import DEFAULT_TOP_K

RUN_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
)


class BatchEvaluationSummary(BaseModel):
    """Aggregate metrics for one benchmark execution."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    started_at: datetime
    completed_at: datetime
    total_cases: int = Field(ge=1)
    successful_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    success_rate: float = Field(ge=0.0, le=1.0)
    average_duration_ms: float = Field(ge=0.0)
    average_retrieval_recall: float = Field(
        ge=0.0,
        le=1.0,
    )
    average_fact_coverage: float = Field(
        ge=0.0,
        le=1.0,
    )
    average_citation_validity: float = Field(
        ge=0.0,
        le=1.0,
    )
    average_overall_score: float = Field(
        ge=0.0,
        le=1.0,
    )
    average_judge_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    failed_case_ids: list[str]
    embedding_model: str
    generation_model: str
    judge_model: str | None = None
    top_k: int = Field(ge=1)


class BatchEvaluationResult(BaseModel):
    """Records, summary, and output paths for a benchmark run."""

    model_config = ConfigDict(frozen=True)

    records: list[EvaluationRunRecord]
    summary: BatchEvaluationSummary
    records_path: Path
    summary_path: Path


def generate_run_id() -> str:
    """Generate a sortable unique identifier for a benchmark run."""

    timestamp = datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    return f"{timestamp}-{uuid4().hex[:8]}"


def calculate_average(
    values: Sequence[float],
) -> float:
    """Calculate an average, returning zero for no values."""

    if not values:
        return 0.0

    return sum(values) / len(values)


def run_benchmark(
    cases: Sequence[BenchmarkCase],
    chunks_path: Path,
    embeddings_path: Path,
    output_directory: Path,
    client: OpenAI,
    run_id: str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    generation_model: str = DEFAULT_GENERATION_MODEL,
    top_k: int = DEFAULT_TOP_K,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    enable_llm_judge: bool = False,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> BatchEvaluationResult:
    """Execute, summarize, and store a complete benchmark run."""

    if not cases:
        raise ValueError(
            "At least one benchmark case is required"
        )

    case_ids = [case.case_id for case in cases]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError(
            "Benchmark cases must have unique case IDs"
        )

    resolved_run_id = run_id or generate_run_id()

    if not RUN_ID_PATTERN.fullmatch(resolved_run_id):
        raise ValueError(
            "Run ID contains unsupported characters"
        )

    if top_k < 1:
        raise ValueError(
            "top_k must be greater than zero"
        )

    if enable_llm_judge and not judge_model.strip():
        raise ValueError(
            "Judge model cannot be empty when judging is enabled"
        )

    started_at = datetime.now(UTC)

    records = [
        run_evaluation_case(
            case=case,
            chunks_path=chunks_path,
            embeddings_path=embeddings_path,
            client=client,
            run_id=resolved_run_id,
            embedding_model=embedding_model,
            generation_model=generation_model,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            enable_llm_judge=enable_llm_judge,
            judge_model=judge_model,
        )
        for case in cases
    ]

    completed_at = datetime.now(UTC)

    successful_records = [
        record
        for record in records
        if record.status == EvaluationStatus.SUCCESS
    ]

    failed_records = [
        record
        for record in records
        if record.status == EvaluationStatus.FAILED
    ]

    successful_metrics = [
        record.metrics
        for record in successful_records
        if record.metrics is not None
    ]

    successful_judge_results = [
        record.judge_result
        for record in successful_records
        if record.judge_result is not None
    ]

    average_judge_score = None

    if enable_llm_judge:
        average_judge_score = calculate_average(
            [
                result.overall_score
                for result in successful_judge_results
            ]
        )

    summary = BatchEvaluationSummary(
        run_id=resolved_run_id,
        started_at=started_at,
        completed_at=completed_at,
        total_cases=len(records),
        successful_cases=len(successful_records),
        failed_cases=len(failed_records),
        success_rate=(
            len(successful_records) / len(records)
        ),
        average_duration_ms=calculate_average(
            [
                record.duration_ms
                for record in records
            ]
        ),
        average_retrieval_recall=calculate_average(
            [
                metrics.retrieval_recall
                for metrics in successful_metrics
            ]
        ),
        average_fact_coverage=calculate_average(
            [
                metrics.fact_coverage
                for metrics in successful_metrics
            ]
        ),
        average_citation_validity=calculate_average(
            [
                metrics.citation_validity
                for metrics in successful_metrics
            ]
        ),
        average_overall_score=calculate_average(
            [
                metrics.overall_score
                for metrics in successful_metrics
            ]
        ),
        average_judge_score=average_judge_score,
        failed_case_ids=[
            record.case_id
            for record in failed_records
        ],
        embedding_model=embedding_model,
        generation_model=generation_model,
        judge_model=(
            judge_model
            if enable_llm_judge
            else None
        ),
        top_k=top_k,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    records_path = (
        output_directory
        / f"{resolved_run_id}.jsonl"
    )
    summary_path = (
        output_directory
        / f"{resolved_run_id}.summary.json"
    )

    records_path.write_text(
        "\n".join(
            record.model_dump_json()
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )

    summary_path.write_text(
        summary.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    return BatchEvaluationResult(
        records=records,
        summary=summary,
        records_path=records_path,
        summary_path=summary_path,
    )