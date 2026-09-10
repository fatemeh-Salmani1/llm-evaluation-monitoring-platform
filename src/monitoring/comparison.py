from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from src.evaluation.batch_runner import (
    BatchEvaluationSummary,
)


class ComparisonThresholds(BaseModel):
    """Thresholds used to identify benchmark regressions."""

    model_config = ConfigDict(frozen=True)

    success_rate_drop: float = Field(
        default=0.01,
        ge=0.0,
    )
    retrieval_recall_drop: float = Field(
        default=0.02,
        ge=0.0,
    )
    overall_score_drop: float = Field(
        default=0.02,
        ge=0.0,
    )
    judge_score_drop: float = Field(
        default=0.05,
        ge=0.0,
    )
    latency_increase_ratio: float = Field(
        default=0.25,
        ge=0.0,
    )


class MetricDelta(BaseModel):
    """Change in one metric between two benchmark runs."""

    model_config = ConfigDict(frozen=True)

    baseline: float
    current: float
    absolute_change: float


class RunComparison(BaseModel):
    """Quality and performance comparison between two runs."""

    model_config = ConfigDict(frozen=True)

    baseline_run_id: str
    current_run_id: str
    success_rate: MetricDelta
    retrieval_recall: MetricDelta
    fact_coverage: MetricDelta
    citation_validity: MetricDelta
    overall_score: MetricDelta
    judge_score: MetricDelta | None = None
    average_duration_ms: MetricDelta
    has_regression: bool
    regressions: list[str]
    improvements: list[str]


def calculate_metric_delta(
    baseline: float,
    current: float,
) -> MetricDelta:
    """Calculate the absolute change in a metric."""

    return MetricDelta(
        baseline=baseline,
        current=current,
        absolute_change=current - baseline,
    )


def calculate_optional_metric_delta(
    baseline: float | None,
    current: float | None,
) -> MetricDelta | None:
    """Calculate a change only when both values exist."""

    if baseline is None or current is None:
        return None

    return calculate_metric_delta(
        baseline=baseline,
        current=current,
    )


def load_batch_summary(
    path: Path,
) -> BatchEvaluationSummary:
    """Load and validate a benchmark summary file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark summary does not exist: {path}"
        )

    try:
        return BatchEvaluationSummary.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except ValidationError as error:
        raise ValueError(
            f"Invalid benchmark summary: {path}"
        ) from error


def compare_summaries(
    baseline: BatchEvaluationSummary,
    current: BatchEvaluationSummary,
    thresholds: ComparisonThresholds | None = None,
) -> RunComparison:
    """Compare benchmark summaries and identify regressions."""

    resolved_thresholds = (
        thresholds or ComparisonThresholds()
    )

    success_rate = calculate_metric_delta(
        baseline.success_rate,
        current.success_rate,
    )
    retrieval_recall = calculate_metric_delta(
        baseline.average_retrieval_recall,
        current.average_retrieval_recall,
    )
    fact_coverage = calculate_metric_delta(
        baseline.average_fact_coverage,
        current.average_fact_coverage,
    )
    citation_validity = calculate_metric_delta(
        baseline.average_citation_validity,
        current.average_citation_validity,
    )
    overall_score = calculate_metric_delta(
        baseline.average_overall_score,
        current.average_overall_score,
    )
    judge_score = calculate_optional_metric_delta(
        baseline.average_judge_score,
        current.average_judge_score,
    )
    average_duration_ms = calculate_metric_delta(
        baseline.average_duration_ms,
        current.average_duration_ms,
    )

    regressions: list[str] = []
    improvements: list[str] = []

    if (
        success_rate.absolute_change
        < -resolved_thresholds.success_rate_drop
    ):
        regressions.append("success_rate")
    elif success_rate.absolute_change > 0:
        improvements.append("success_rate")

    if (
        retrieval_recall.absolute_change
        < -resolved_thresholds.retrieval_recall_drop
    ):
        regressions.append("retrieval_recall")
    elif retrieval_recall.absolute_change > 0:
        improvements.append("retrieval_recall")

    if fact_coverage.absolute_change > 0:
        improvements.append("fact_coverage")

    if citation_validity.absolute_change > 0:
        improvements.append("citation_validity")

    if (
        overall_score.absolute_change
        < -resolved_thresholds.overall_score_drop
    ):
        regressions.append("overall_score")
    elif overall_score.absolute_change > 0:
        improvements.append("overall_score")

    if judge_score is not None:
        if (
            judge_score.absolute_change
            < -resolved_thresholds.judge_score_drop
        ):
            regressions.append("judge_score")
        elif judge_score.absolute_change > 0:
            improvements.append("judge_score")

    latency_limit = baseline.average_duration_ms * (
        1 + resolved_thresholds.latency_increase_ratio
    )

    if current.average_duration_ms > latency_limit:
        regressions.append("average_duration_ms")
    elif average_duration_ms.absolute_change < 0:
        improvements.append("average_duration_ms")

    return RunComparison(
        baseline_run_id=baseline.run_id,
        current_run_id=current.run_id,
        success_rate=success_rate,
        retrieval_recall=retrieval_recall,
        fact_coverage=fact_coverage,
        citation_validity=citation_validity,
        overall_score=overall_score,
        judge_score=judge_score,
        average_duration_ms=average_duration_ms,
        has_regression=bool(regressions),
        regressions=regressions,
        improvements=improvements,
    )


def compare_summary_files(
    baseline_path: Path,
    current_path: Path,
    thresholds: ComparisonThresholds | None = None,
) -> RunComparison:
    """Load and compare two benchmark summary files."""

    baseline = load_batch_summary(baseline_path)
    current = load_batch_summary(current_path)

    return compare_summaries(
        baseline=baseline,
        current=current,
        thresholds=thresholds,
    )