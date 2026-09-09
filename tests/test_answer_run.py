from pathlib import Path
from types import SimpleNamespace

from src.generation.run_answer import (
    answer_question,
    format_grounded_answer,
)
from src.retrieval.embedder import ChunkEmbedding
from src.retrieval.embedding_storage import write_embeddings
from src.retrieval.models import DocumentChunk
from src.retrieval.storage import write_chunks_jsonl


class FakeEmbeddings:
    """Fake embeddings endpoint for answer-runner tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(
        self,
        *,
        model: str,
        input: list[str],
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "model": model,
                "input": input,
            }
        )

        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    index=0,
                    embedding=[1.0, 0.0],
                )
            ]
        )


class FakeResponses:
    """Fake Responses API endpoint for answer-runner tests."""

    def __init__(self) -> None:
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
            output_text=(
                "An eval needs `data_source_config` and "
                "`testing_criteria` "
                "[openai-evals-guide-chunk-0001]."
            )
        )


class FakeOpenAI:
    """Fake client containing embeddings and responses endpoints."""

    def __init__(self) -> None:
        self.embeddings = FakeEmbeddings()
        self.responses = FakeResponses()


def create_chunk(position: int) -> DocumentChunk:
    """Create a document chunk for answer-runner tests."""

    return DocumentChunk(
        chunk_id=f"openai-evals-guide-chunk-{position:04d}",
        source_id="openai-evals-guide",
        position=position,
        heading_path=[
            "Working with evals",
            f"Section {position}",
        ],
        content=f"Evaluation guidance number {position}.",
        token_count=5,
    )


def create_embedding(
    position: int,
    vector: list[float],
) -> ChunkEmbedding:
    """Create a stored embedding for answer-runner tests."""

    return ChunkEmbedding(
        chunk_id=f"openai-evals-guide-chunk-{position:04d}",
        source_id="openai-evals-guide",
        embedding_model="text-embedding-3-small",
        embedding=vector,
    )


def test_answer_question_runs_retrieval_and_generation(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    embeddings_path = tmp_path / "embeddings.jsonl"
    chunks = [
        create_chunk(0),
        create_chunk(1),
    ]
    embeddings = [
        create_embedding(0, [0.0, 1.0]),
        create_embedding(1, [1.0, 0.0]),
    ]

    write_chunks_jsonl(
        chunks=chunks,
        output_path=chunks_path,
    )
    write_embeddings(
        embeddings=embeddings,
        output_path=embeddings_path,
    )
    client = FakeOpenAI()

    result = answer_question(
        question="What two ingredients does an eval need?",
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        client=client,
        top_k=1,
    )

    assert "data_source_config" in result.answer
    assert result.source_chunk_ids == [
        "openai-evals-guide-chunk-0001"
    ]
    assert len(client.embeddings.calls) == 1
    assert len(client.responses.calls) == 1

    response_prompt = client.responses.calls[0]["input"]

    assert "openai-evals-guide-chunk-0001" in response_prompt
    assert "openai-evals-guide-chunk-0000" not in response_prompt


def test_answer_question_passes_custom_limits(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    embeddings_path = tmp_path / "embeddings.jsonl"
    chunks = [create_chunk(0)]
    embeddings = [
        create_embedding(0, [1.0, 0.0])
    ]

    write_chunks_jsonl(
        chunks=chunks,
        output_path=chunks_path,
    )
    write_embeddings(
        embeddings=embeddings,
        output_path=embeddings_path,
    )
    client = FakeOpenAI()

    answer_question(
        question="What is an eval?",
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        client=client,
        top_k=1,
        generation_model="test-generation-model",
        max_output_tokens=200,
    )

    response_call = client.responses.calls[0]

    assert response_call["model"] == "test-generation-model"
    assert response_call["max_output_tokens"] == 200


def test_format_grounded_answer_includes_sources(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    embeddings_path = tmp_path / "embeddings.jsonl"

    write_chunks_jsonl(
        chunks=[create_chunk(0)],
        output_path=chunks_path,
    )
    write_embeddings(
        embeddings=[
            create_embedding(0, [1.0, 0.0])
        ],
        output_path=embeddings_path,
    )
    client = FakeOpenAI()

    result = answer_question(
        question="What is an eval?",
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        client=client,
        top_k=1,
    )
    formatted = format_grounded_answer(result)

    assert "Question:\nWhat is an eval?" in formatted
    assert "Answer:" in formatted
    assert "Generation model:" in formatted
    assert "Retrieved source chunks:" in formatted
    assert "openai-evals-guide-chunk-0000" in formatted