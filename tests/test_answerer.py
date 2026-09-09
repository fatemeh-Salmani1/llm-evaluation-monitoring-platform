from types import SimpleNamespace

import pytest

from src.generation.answerer import (
    DEFAULT_GENERATION_MODEL,
    SYSTEM_INSTRUCTIONS,
    build_grounded_prompt,
    generate_grounded_answer,
)
from src.retrieval.models import DocumentChunk
from src.retrieval.retriever import RetrievedChunk


class FakeResponses:
    """Fake Responses API endpoint for answer-generation tests."""

    def __init__(
        self,
        output_text: str,
    ) -> None:
        self.output_text = output_text
        self.calls: list[dict[str, object]] = []

    def create(
        self,
        *,
        model: str,
        instructions: str,
        input: str,
        max_output_tokens: int,
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "model": model,
                "instructions": instructions,
                "input": input,
                "max_output_tokens": max_output_tokens,
            }
        )

        return SimpleNamespace(
            output_text=self.output_text
        )


class FakeOpenAI:
    """Fake OpenAI client for answer-generation tests."""

    def __init__(
        self,
        output_text: str,
    ) -> None:
        self.responses = FakeResponses(
            output_text=output_text
        )


def create_retrieved_chunk(
    position: int = 0,
) -> RetrievedChunk:
    """Create a retrieved chunk for answer-generation tests."""

    chunk = DocumentChunk(
        chunk_id=f"openai-evals-guide-chunk-{position:04d}",
        source_id="openai-evals-guide",
        position=position,
        heading_path=[
            "Working with evals",
            "Create an eval for a task",
        ],
        content=(
            "An eval needs data_source_config and "
            "testing_criteria."
        ),
        token_count=10,
    )

    return RetrievedChunk(
        rank=position + 1,
        similarity_score=0.9,
        chunk=chunk,
    )


def test_build_grounded_prompt_contains_question_and_context() -> None:
    retrieved_chunk = create_retrieved_chunk()

    prompt = build_grounded_prompt(
        question="What does an eval need?",
        retrieved_chunks=[retrieved_chunk],
    )

    assert "Question:\nWhat does an eval need?" in prompt
    assert retrieved_chunk.chunk.chunk_id in prompt
    assert "Create an eval for a task" in prompt
    assert "data_source_config" in prompt
    assert "testing_criteria" in prompt


def test_generate_grounded_answer_calls_responses_api() -> None:
    client = FakeOpenAI(
        output_text=(
            "An eval needs `data_source_config` and "
            "`testing_criteria` "
            "[openai-evals-guide-chunk-0000]."
        )
    )
    retrieved_chunk = create_retrieved_chunk()

    result = generate_grounded_answer(
        question="What does an eval need?",
        retrieved_chunks=[retrieved_chunk],
        client=client,
    )

    assert result.question == "What does an eval need?"
    assert result.model == DEFAULT_GENERATION_MODEL
    assert result.source_chunk_ids == [
        retrieved_chunk.chunk.chunk_id
    ]
    assert "data_source_config" in result.answer

    api_call = client.responses.calls[0]

    assert api_call["model"] == DEFAULT_GENERATION_MODEL
    assert api_call["instructions"] == SYSTEM_INSTRUCTIONS
    assert retrieved_chunk.chunk.chunk_id in api_call["input"]
    assert api_call["max_output_tokens"] == 500


def test_build_grounded_prompt_rejects_empty_question() -> None:
    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        build_grounded_prompt(
            question="   ",
            retrieved_chunks=[create_retrieved_chunk()],
        )


def test_build_grounded_prompt_requires_context() -> None:
    with pytest.raises(
        ValueError,
        match="At least one retrieved chunk",
    ):
        build_grounded_prompt(
            question="What does an eval need?",
            retrieved_chunks=[],
        )


def test_generate_grounded_answer_rejects_invalid_token_limit() -> None:
    client = FakeOpenAI(output_text="An answer.")

    with pytest.raises(
        ValueError,
        match="max_output_tokens must be greater than zero",
    ):
        generate_grounded_answer(
            question="What does an eval need?",
            retrieved_chunks=[create_retrieved_chunk()],
            client=client,
            max_output_tokens=0,
        )


def test_generate_grounded_answer_rejects_empty_response() -> None:
    client = FakeOpenAI(output_text="   ")

    with pytest.raises(
        RuntimeError,
        match="returned an empty answer",
    ):
        generate_grounded_answer(
            question="What does an eval need?",
            retrieved_chunks=[create_retrieved_chunk()],
            client=client,
        )