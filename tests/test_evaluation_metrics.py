import pytest

from src.evaluation.metrics import (
    calculate_citation_validity,
    calculate_fact_coverage,
    calculate_retrieval_metrics,
    evaluate_deterministically,
    extract_chunk_citations,
    normalize_text,
)
from src.evaluation.models import (
    BenchmarkCase,
    Difficulty,
    EvaluationCategory,
)


def create_benchmark_case() -> BenchmarkCase:
    """Create a benchmark case for metric tests."""

    return BenchmarkCase(
        case_id="eval-0001",
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


def test_normalize_text_removes_formatting() -> None:
    normalized = normalize_text(
        "An eval needs `data_source_config`!"
    )

    assert normalized == (
        "an eval needs data_source_config"
    )


def test_extract_chunk_citations_preserves_unique_order() -> None:
    answer = (
        "First fact [openai-evals-guide-chunk-0004]. "
        "Second fact [openai-evals-guide-chunk-0001]. "
        "Again [openai-evals-guide-chunk-0004]."
    )

    citations = extract_chunk_citations(answer)

    assert citations == [
        "openai-evals-guide-chunk-0004",
        "openai-evals-guide-chunk-0001",
    ]


def test_calculate_retrieval_metrics_returns_recall() -> None:
    retrieval_hit, retrieval_recall = (
        calculate_retrieval_metrics(
            expected_chunk_ids=[
                "chunk-0001",
                "chunk-0002",
            ],
            retrieved_chunk_ids=[
                "chunk-0002",
                "chunk-0003",
            ],
        )
    )

    assert retrieval_hit is True
    assert retrieval_recall == pytest.approx(0.5)


def test_calculate_fact_coverage_identifies_missing_facts() -> None:
    coverage, matched_facts, missing_facts = (
        calculate_fact_coverage(
            required_facts=[
                "data_source_config",
                "testing_criteria",
            ],
            answer=(
                "The eval defines a `data_source_config`."
            ),
        )
    )

    assert coverage == pytest.approx(0.5)
    assert matched_facts == ["data_source_config"]
    assert missing_facts == ["testing_criteria"]


def test_calculate_citation_validity_finds_invalid_ids() -> None:
    answer = (
        "Supported fact "
        "[openai-evals-guide-chunk-0004]. "
        "Unsupported fact "
        "[openai-evals-guide-chunk-9999]."
    )

    validity, citations, invalid_citations = (
        calculate_citation_validity(
            answer=answer,
            retrieved_chunk_ids=[
                "openai-evals-guide-chunk-0004"
            ],
        )
    )

    assert validity == pytest.approx(0.5)
    assert citations == [
        "openai-evals-guide-chunk-0004",
        "openai-evals-guide-chunk-9999",
    ]
    assert invalid_citations == [
        "openai-evals-guide-chunk-9999"
    ]


def test_evaluate_deterministically_returns_perfect_score() -> None:
    case = create_benchmark_case()
    answer = (
        "An eval needs `data_source_config` and "
        "`testing_criteria` "
        "[openai-evals-guide-chunk-0004]."
    )

    result = evaluate_deterministically(
        case=case,
        answer=answer,
        retrieved_chunk_ids=[
            "openai-evals-guide-chunk-0004",
            "openai-evals-guide-chunk-0000",
        ],
    )

    assert result.retrieval_hit is True
    assert result.retrieval_recall == pytest.approx(1.0)
    assert result.fact_coverage == pytest.approx(1.0)
    assert result.citation_validity == pytest.approx(1.0)
    assert result.overall_score == pytest.approx(1.0)
    assert result.missing_facts == []
    assert result.invalid_citation_ids == []


def test_evaluate_deterministically_rejects_empty_answer() -> None:
    with pytest.raises(
        ValueError,
        match="Answer cannot be empty",
    ):
        evaluate_deterministically(
            case=create_benchmark_case(),
            answer="   ",
            retrieved_chunk_ids=[
                "openai-evals-guide-chunk-0004"
            ],
        )


def test_fact_coverage_accepts_paraphrased_wording() -> None:
    required_facts = [
        "test model outputs",
        "style and content criteria",
        "application performance against expectations",
    ]
    answer = (
        "Evaluations test an LLM application's outputs "
        "against specified style and content criteria. "
        "They help determine whether the application "
        "meets expectations."
    )

    coverage, matched_facts, missing_facts = (
        calculate_fact_coverage(
            required_facts=required_facts,
            answer=answer,
        )
    )

    assert coverage == pytest.approx(1.0)
    assert matched_facts == required_facts
    assert missing_facts == []