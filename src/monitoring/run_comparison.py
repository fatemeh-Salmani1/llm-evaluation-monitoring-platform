import argparse
from pathlib import Path

from src.monitoring.comparison import (
    ComparisonThresholds,
    MetricDelta,
    RunComparison,
    compare_summary_files,
)

DEFAULT_THRESHOLDS = ComparisonThresholds()


def format_metric_delta(
    name: str,
    delta: MetricDelta,
    precision: int = 4,
) -> str:
    """Format one metric comparison."""

    return (
        f"{name}: "
        f"{delta.baseline:.{precision}f} -> "
        f"{delta.current:.{precision}f} "
        f"({delta.absolute_change:+.{precision}f})"
    )


def format_run_comparison(
    comparison: RunComparison,
) -> str:
    """Format a run comparison for terminal output."""

    regressions = (
        ", ".join(comparison.regressions)
        if comparison.regressions
        else "None"
    )
    improvements = (
        ", ".join(comparison.improvements)
        if comparison.improvements
        else "None"
    )
    quality_gate = (
        "FAILED"
        if comparison.has_regression
        else "PASSED"
    )

    lines = [
        f"Baseline run: {comparison.baseline_run_id}",
        f"Current run: {comparison.current_run_id}",
        "",
        format_metric_delta(
            "Success rate",
            comparison.success_rate,
        ),
        format_metric_delta(
            "Retrieval recall",
            comparison.retrieval_recall,
        ),
        format_metric_delta(
            "Fact coverage",
            comparison.fact_coverage,
        ),
        format_metric_delta(
            "Citation validity",
            comparison.citation_validity,
        ),
        format_metric_delta(
            "Overall score",
            comparison.overall_score,
        ),
        format_metric_delta(
            "Average duration (ms)",
            comparison.average_duration_ms,
            precision=2,
        ),
        "",
        f"Improvements: {improvements}",
        f"Regressions: {regressions}",
        f"Quality gate: {quality_gate}",
    ]

    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for run comparison."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare two LLM benchmark summaries "
            "and detect regressions."
        )
    )
    parser.add_argument(
        "baseline_path",
        type=Path,
        help="Path to the baseline summary JSON file.",
    )
    parser.add_argument(
        "current_path",
        type=Path,
        help="Path to the current summary JSON file.",
    )
    parser.add_argument(
        "--success-rate-drop",
        type=float,
        default=DEFAULT_THRESHOLDS.success_rate_drop,
        help="Maximum permitted success-rate decrease.",
    )
    parser.add_argument(
        "--retrieval-recall-drop",
        type=float,
        default=(
            DEFAULT_THRESHOLDS.retrieval_recall_drop
        ),
        help="Maximum permitted retrieval-recall decrease.",
    )
    parser.add_argument(
        "--overall-score-drop",
        type=float,
        default=DEFAULT_THRESHOLDS.overall_score_drop,
        help="Maximum permitted overall-score decrease.",
    )
    parser.add_argument(
        "--latency-increase-ratio",
        type=float,
        default=(
            DEFAULT_THRESHOLDS.latency_increase_ratio
        ),
        help="Maximum permitted proportional latency increase.",
    )

    return parser


def main() -> int:
    """Compare benchmark runs and return a quality-gate code."""

    arguments = build_argument_parser().parse_args()
    thresholds = ComparisonThresholds(
        success_rate_drop=arguments.success_rate_drop,
        retrieval_recall_drop=(
            arguments.retrieval_recall_drop
        ),
        overall_score_drop=arguments.overall_score_drop,
        latency_increase_ratio=(
            arguments.latency_increase_ratio
        ),
    )

    comparison = compare_summary_files(
        baseline_path=arguments.baseline_path,
        current_path=arguments.current_path,
        thresholds=thresholds,
    )

    print(format_run_comparison(comparison))

    return 1 if comparison.has_regression else 0


if __name__ == "__main__":
    raise SystemExit(main())