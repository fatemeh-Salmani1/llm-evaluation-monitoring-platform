from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.evaluation.judge import (
    JudgeScores,
    build_judge_prompt,
    calculate_overall_judge_score,
    judge_answer,
)
from src.evaluation.models import (
    BenchmarkCase,
    Difficulty,
    EvaluationCategory,
)
from src.retrieval.models import DocumentChunk


class FakeResponses:
    """Fake Responses endpoint for judge tests."""

    def __init__(
        self,
        scores: JudgeScores | None,
    ) -> None:
        self.scores = scores
        self.calls: list[dict[str, object]] = []

    def parse(
        self,
        *,
        model: str,
        input: list[dict[str, str]],
        text_format: type[JudgeScores],
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "model": model,
                "input": input,
                "text_format": text_format,
            }
        )

        return SimpleNamespace(
            output_parsed=self.scores,
        )


class FakeOpenAI:
    """Fake OpenAI client for judge tests."""

    def __init__(
        self,
        scores: JudgeScores | None,
    ) -> None:
        self.responses = FakeResponses(scores)


def create_case() -> BenchmarkCase:
    """Create an evaluation case for judge tests."""

    return BenchmarkCase(
        case_id="eval-0001",
        question="What is the purpose of an LLM evaluation?",
        reference_answer=(
            "An evaluation tests model outputs against defined criteria."
        ),
        category=EvaluationCategory.EVAL_CONCEPTS,
        difficulty=Difficulty.EASY,
        required_facts=[
            "test model outputs",
            "defined criteria",
        ],
        expected_source_ids=["openai-evals-guide"],
        expected_chunk_ids=[
            "openai-evals-guide-chunk-0000",
        ],
        tags=["evals"],
    )


def create_chunk() -> DocumentChunk:
    """Create a retrieved context chunk for judge tests."""

    return DocumentChunk(
        chunk_id="openai-evals-guide-chunk-0000",
        source_id="openai-evals-guide",
        position=0,
        heading_path=["Working with evals"],
        content=(
            "Evaluations test model outputs to ensure they meet "
            "specified style and content criteria."
        ),
        token_count=15,
    )


def create_scores() -> JudgeScores:
    """Create valid structured judge scores."""

    return JudgeScores(
        relevance=5,
        completeness=4,
        groundedness=5,
        clarity=4,
        reasoning=(
            "The answer is relevant, grounded, and mostly complete."
        ),
    )


def test_judge_scores_reject_out_of_range_score() -> None:
    with pytest.raises(ValidationError):
        JudgeScores(
            relevance=6,
            completeness=4,
            groundedness=5,
            clarity=4,
            reasoning="Invalid relevance score.",
        )


def test_build_judge_prompt_contains_evaluation_inputs() -> None:
    case = create_case()
    chunk = create_chunk()

    prompt = build_judge_prompt(
        case=case,
        answer="Evaluations test model outputs.",
        retrieved_chunks=[chunk],
    )

    assert case.question in prompt
    assert case.reference_answer in prompt
    assert case.required_facts[0] in prompt
    assert chunk.chunk_id in prompt
    assert chunk.content in prompt
    assert "Evaluations test model outputs." in prompt


def test_build_judge_prompt_rejects_empty_answer() -> None:
    with pytest.raises(ValueError, match="Answer cannot be empty"):
        build_judge_prompt(
            case=create_case(),
            answer=" ",
            retrieved_chunks=[create_chunk()],
        )


def test_build_judge_prompt_requires_context() -> None:
    with pytest.raises(
        ValueError,
        match="At least one retrieved chunk",
    ):
        build_judge_prompt(
            case=create_case(),
            answer="Evaluations test model outputs.",
            retrieved_chunks=[],
        )


def test_calculate_overall_judge_score_normalizes_scores() -> None:
    assert calculate_overall_judge_score(create_scores()) == 0.875


def test_judge_answer_returns_structured_result() -> None:
    client = FakeOpenAI(create_scores())

    result = judge_answer(
        case=create_case(),
        answer=(
            "An evaluation tests model outputs against defined criteria."
        ),
        retrieved_chunks=[create_chunk()],
        client=client,
        model="judge-model",
    )

    assert result.case_id == "eval-0001"
    assert result.judge_model == "judge-model"
    assert result.relevance == 5
    assert result.completeness == 4
    assert result.groundedness == 5
    assert result.clarity == 4
    assert result.overall_score == 0.875
    assert len(client.responses.calls) == 1
    assert client.responses.calls[0]["model"] == "judge-model"
    assert (
        client.responses.calls[0]["text_format"]
        is JudgeScores
    )


def test_judge_answer_rejects_missing_structured_output() -> None:
    client = FakeOpenAI(None)

    with pytest.raises(
        ValueError,
        match="did not contain structured scores",
    ):
        judge_answer(
            case=create_case(),
            answer="Evaluations test model outputs.",
            retrieved_chunks=[create_chunk()],
            client=client,
        )