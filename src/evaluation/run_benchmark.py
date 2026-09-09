import argparse
from collections.abc import Sequence
from pathlib import Path

from openai import OpenAI

from src.config.settings import get_settings
from src.evaluation.batch_runner import (
    BatchEvaluationResult,
    BatchEvaluationSummary,
)
from src.evaluation.batch_runner import (
    run_benchmark as run_batch_benchmark,
)
from src.evaluation.loader import load_benchmark
from src.evaluation.models import BenchmarkCase
from src.generation.answerer import (
    DEFAULT_GENERATION_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
)
from src.retrieval.embedder import DEFAULT_EMBEDDING_MODEL
from src.retrieval.run_retrieval import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_TOP_K,
)

DEFAULT_BENCHMARK_PATH = Path(
    "data/benchmarks/openai_evals.jsonl"
)
DEFAULT_OUTPUT_DIRECTORY = Path(
    "data/processed/evaluation_runs"
)


def select_benchmark_cases(
    cases: Sequence[BenchmarkCase],
    limit: int | None,
) -> list[BenchmarkCase]:
    """Select a safe subset of benchmark cases."""

    if limit is None:
        return list(cases)

    if limit < 1:
        raise ValueError(
            "Benchmark limit must be greater than zero"
        )

    return list(cases[:limit])


def format_batch_summary(
    summary: BatchEvaluationSummary,
) -> str:
    """Format aggregate evaluation results for the terminal."""

    failed_cases = (
        ", ".join(summary.failed_case_ids)
        if summary.failed_case_ids
        else "None"
    )

    return "\n".join(
        [
            f"Run ID: {summary.run_id}",
            f"Total cases: {summary.total_cases}",
            (
                "Successful cases: "
                f"{summary.successful_cases}"
            ),
            f"Failed cases: {summary.failed_cases}",
            f"Success rate: {summary.success_rate:.2%}",
            (
                "Average retrieval recall: "
                f"{summary.average_retrieval_recall:.4f}"
            ),
            (
                "Average fact coverage: "
                f"{summary.average_fact_coverage:.4f}"
            ),
            (
                "Average citation validity: "
                f"{summary.average_citation_validity:.4f}"
            ),
            (
                "Average overall score: "
                f"{summary.average_overall_score:.4f}"
            ),
            (
                "Average duration: "
                f"{summary.average_duration_ms:.2f} ms"
            ),
            f"Failed case IDs: {failed_cases}",
        ]
    )


def format_batch_result(
    result: BatchEvaluationResult,
) -> str:
    """Format a completed benchmark run and output paths."""

    return (
        f"{format_batch_summary(result.summary)}\n"
        f"Records: {result.records_path}\n"
        f"Summary: {result.summary_path}"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for benchmark execution."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the LLM evaluation benchmark and save metrics."
        )
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=DEFAULT_BENCHMARK_PATH,
        help="Path to the benchmark JSONL file.",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to the document chunk JSONL file.",
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=DEFAULT_EMBEDDINGS_PATH,
        help="Path to the embedding JSONL file.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for evaluation records and summaries.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional identifier for this benchmark run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N benchmark cases.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Model used for question embeddings.",
    )
    parser.add_argument(
        "--generation-model",
        default=DEFAULT_GENERATION_MODEL,
        help="Model used for answer generation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of retrieved chunks per question.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum generated tokens per answer.",
    )

    return parser


def main() -> None:
    """Execute the benchmark from the command line."""

    arguments = build_argument_parser().parse_args()
    settings = get_settings()

    if settings.openai_api_key is None:
        raise RuntimeError(
            "OPENAI_API_KEY is required to run the benchmark"
        )

    all_cases = load_benchmark(
        arguments.benchmark_path
    )
    selected_cases = select_benchmark_cases(
        cases=all_cases,
        limit=arguments.limit,
    )

    client = OpenAI(
        api_key=settings.openai_api_key.get_secret_value()
    )

    result = run_batch_benchmark(
        cases=selected_cases,
        chunks_path=arguments.chunks_path,
        embeddings_path=arguments.embeddings_path,
        output_directory=arguments.output_directory,
        client=client,
        run_id=arguments.run_id,
        embedding_model=arguments.embedding_model,
        generation_model=arguments.generation_model,
        top_k=arguments.top_k,
        max_output_tokens=arguments.max_output_tokens,
    )

    print(format_batch_result(result))


if __name__ == "__main__":
    main()