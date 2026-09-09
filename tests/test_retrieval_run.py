from pathlib import Path
from types import SimpleNamespace

import pytest

from src.retrieval.embedder import ChunkEmbedding
from src.retrieval.embedding_storage import write_embeddings
from src.retrieval.models import DocumentChunk
from src.retrieval.retriever import RetrievedChunk
from src.retrieval.run_retrieval import (
    format_retrieval_results,
    retrieve_from_files,
)
from src.retrieval.storage import write_chunks_jsonl


class FakeEmbeddings:
    """Fake embeddings endpoint for retrieval-runner tests."""

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


class FakeOpenAI:
    """Fake OpenAI client for retrieval-runner tests."""

    def __init__(self) -> None:
        self.embeddings = FakeEmbeddings()


def create_chunk(position: int) -> DocumentChunk:
    """Create a document chunk for runner tests."""

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
    """Create a stored embedding for runner tests."""

    return ChunkEmbedding(
        chunk_id=f"openai-evals-guide-chunk-{position:04d}",
        source_id="openai-evals-guide",
        embedding_model="text-embedding-3-small",
        embedding=vector,
    )


def test_retrieve_from_files_returns_best_chunk(
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

    results = retrieve_from_files(
        question="How do evaluations work?",
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        client=client,
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk == chunks[1]
    assert results[0].rank == 1
    assert client.embeddings.calls[0]["input"] == [
        "How do evaluations work?"
    ]


def test_format_retrieval_results_includes_metadata() -> None:
    result = RetrievedChunk(
        rank=1,
        similarity_score=0.98765,
        chunk=create_chunk(0),
    )

    formatted = format_retrieval_results(
        results=[result],
        preview_characters=15,
    )

    assert "Rank: 1" in formatted
    assert "Similarity: 0.9877" in formatted
    assert "openai-evals-guide-chunk-0000" in formatted
    assert "Working with evals > Section 0" in formatted
    assert "Preview:" in formatted
    assert formatted.endswith("...")


def test_format_retrieval_results_handles_empty_results() -> None:
    formatted = format_retrieval_results(results=[])

    assert formatted == "No matching chunks found."


def test_format_retrieval_results_rejects_invalid_preview_size() -> None:
    with pytest.raises(
        ValueError,
        match="preview_characters must be greater than zero",
    ):
        format_retrieval_results(
            results=[],
            preview_characters=0,
        )