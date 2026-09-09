import pytest
from pydantic import ValidationError

from src.evaluation.models import (
    BenchmarkCase,
    Difficulty,
    EvaluationCategory,
)


def create_valid_case() -> BenchmarkCase:
    """Create a valid benchmark case for model tests."""

    return BenchmarkCase(
        case_id="eval-0001",
        question="What are the two key ingredients of an eval?",
        reference_answer=(
            "An eval needs a data source configuration "
            "and testing criteria."
        ),
        category=EvaluationCategory.EVAL_CONCEPTS,
        difficulty=Difficulty.EASY,
        required_facts=[
            "An eval needs data_source_config.",
            "An eval needs testing_criteria.",
        ],
        expected_source_ids=["openai-evals-guide"],
        expected_chunk_ids=[
            "openai-evals-guide-chunk-0004"
        ],
        tags=[
            "evals",
            "configuration",
            "graders",
        ],
    )


def test_benchmark_case_accepts_valid_data() -> None:
    case = create_valid_case()

    assert case.case_id == "eval-0001"
    assert case.category == EvaluationCategory.EVAL_CONCEPTS
    assert len(case.required_facts) == 2
    assert case.expected_source_ids == [
        "openai-evals-guide"
    ]
    assert case.expected_chunk_ids == [
        "openai-evals-guide-chunk-0004"
    ]


def test_benchmark_case_rejects_invalid_case_id() -> None:
    valid_data = create_valid_case().model_dump()
    valid_data["case_id"] = "de-0001"

    with pytest.raises(ValidationError):
        BenchmarkCase.model_validate(valid_data)


def test_benchmark_case_requires_at_least_one_fact() -> None:
    with pytest.raises(ValidationError):
        BenchmarkCase(
            case_id="eval-0002",
            question="Why are graders required when running evaluations?",
            reference_answer=(
                "Graders determine whether a model output "
                "satisfies the evaluation criteria."
            ),
            category=EvaluationCategory.GRADERS,
            difficulty=Difficulty.EASY,
            required_facts=[],
        )


def test_benchmark_case_serializes_to_json() -> None:
    serialized = create_valid_case().model_dump_json()

    assert '"case_id":"eval-0001"' in serialized
    assert '"category":"eval_concepts"' in serialized
    assert '"difficulty":"easy"' in serialized
    assert '"expected_chunk_ids":' in serialized