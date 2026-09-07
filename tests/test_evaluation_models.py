import pytest
from pydantic import ValidationError

from src.evaluation.models import (
    BenchmarkCase,
    Difficulty,
    EvaluationCategory,
)


def create_valid_case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="de-0001",
        question="What is the purpose of a primary key in a database?",
        reference_answer=(
            "A primary key uniquely identifies each row in a database table."
        ),
        category=EvaluationCategory.SQL,
        difficulty=Difficulty.EASY,
        required_facts=[
            "A primary key uniquely identifies each row.",
            "Primary-key values cannot be null.",
        ],
        expected_source_ids=["database-keys"],
        tags=["sql", "database", "primary-key"],
    )


def test_benchmark_case_accepts_valid_data() -> None:
    case = create_valid_case()

    assert case.case_id == "de-0001"
    assert case.category == EvaluationCategory.SQL
    assert len(case.required_facts) == 2


def test_benchmark_case_rejects_invalid_case_id() -> None:
    valid_data = create_valid_case().model_dump()
    valid_data["case_id"] = "invalid"

    with pytest.raises(ValidationError):
        BenchmarkCase.model_validate(valid_data)


def test_benchmark_case_requires_at_least_one_fact() -> None:
    with pytest.raises(ValidationError):
        BenchmarkCase(
            case_id="de-0002",
            question="How can duplicate records affect analytical results?",
            reference_answer=(
                "Duplicate records can inflate aggregations and distort metrics."
            ),
            category=EvaluationCategory.DATA_QUALITY,
            difficulty=Difficulty.EASY,
            required_facts=[],
        )


def test_benchmark_case_serializes_to_json() -> None:
    serialized = create_valid_case().model_dump_json()

    assert '"case_id":"de-0001"' in serialized
    assert '"category":"sql"' in serialized
    assert '"difficulty":"easy"' in serialized