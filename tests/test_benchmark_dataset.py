from pathlib import Path

from src.evaluation.loader import load_benchmark
from src.evaluation.models import EvaluationCategory

PROJECT_ROOT = Path(__file__).parents[1]
BENCHMARK_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmarks"
    / "openai_evals.jsonl"
)


def test_project_benchmark_dataset_is_valid() -> None:
    cases = load_benchmark(BENCHMARK_PATH)

    assert len(cases) == 8
    assert [case.case_id for case in cases] == [
        f"eval-{number:04d}"
        for number in range(1, 9)
    ]
    assert all(case.required_facts for case in cases)
    assert all(case.expected_source_ids for case in cases)
    assert all(case.expected_chunk_ids for case in cases)


def test_project_benchmark_covers_core_categories() -> None:
    cases = load_benchmark(BENCHMARK_PATH)
    categories = {
        case.category
        for case in cases
    }

    assert categories == {
        EvaluationCategory.EVAL_CONCEPTS,
        EvaluationCategory.DATA_SOURCES,
        EvaluationCategory.GRADERS,
        EvaluationCategory.EVAL_RUNS,
        EvaluationCategory.API_USAGE,
    }