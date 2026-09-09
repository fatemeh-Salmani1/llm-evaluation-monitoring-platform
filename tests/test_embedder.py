from types import SimpleNamespace

import pytest

from src.retrieval.embedder import (
    DEFAULT_EMBEDDING_MODEL,
    embed_chunks,
)
from src.retrieval.models import DocumentChunk


class FakeEmbeddings:
    """Fake embeddings endpoint used without calling OpenAI."""

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

        response_items = [
            SimpleNamespace(
                index=index,
                embedding=[
                    float(index),
                    float(len(content)),
                ],
            )
            for index, content in enumerate(input)
        ]

        return SimpleNamespace(data=list(reversed(response_items)))


class FakeOpenAI:
    """Fake OpenAI client exposing an embeddings resource."""

    def __init__(self) -> None:
        self.embeddings = FakeEmbeddings()


def create_chunk(position: int) -> DocumentChunk:
    """Create a document chunk for embedding tests."""

    return DocumentChunk(
        chunk_id=f"openai-evals-guide-chunk-{position:04d}",
        source_id="openai-evals-guide",
        position=position,
        heading_path=["Working with evals"],
        content=f"Evaluation guidance number {position}.",
        token_count=5,
    )


def test_embed_chunks_preserves_chunk_lineage() -> None:
    client = FakeOpenAI()
    chunks = [create_chunk(0), create_chunk(1)]

    results = embed_chunks(
        chunks=chunks,
        client=client,
    )

    assert len(results) == 2
    assert results[0].chunk_id == chunks[0].chunk_id
    assert results[0].source_id == chunks[0].source_id
    assert results[0].embedding_model == DEFAULT_EMBEDDING_MODEL
    assert results[0].embedding == [
        0.0,
        float(len(chunks[0].content)),
    ]
    assert results[1].chunk_id == chunks[1].chunk_id


def test_embed_chunks_sends_content_in_batches() -> None:
    client = FakeOpenAI()
    chunks = [
        create_chunk(0),
        create_chunk(1),
        create_chunk(2),
    ]

    results = embed_chunks(
        chunks=chunks,
        client=client,
        batch_size=2,
    )

    assert len(results) == 3
    assert len(client.embeddings.calls) == 2
    assert client.embeddings.calls[0]["input"] == [
        chunks[0].content,
        chunks[1].content,
    ]
    assert client.embeddings.calls[1]["input"] == [
        chunks[2].content,
    ]


def test_embed_chunks_rejects_empty_collection() -> None:
    client = FakeOpenAI()

    with pytest.raises(
        ValueError,
        match="At least one document chunk",
    ):
        embed_chunks(
            chunks=[],
            client=client,
        )


def test_embed_chunks_rejects_invalid_batch_size() -> None:
    client = FakeOpenAI()

    with pytest.raises(
        ValueError,
        match="Batch size must be greater than zero",
    ):
        embed_chunks(
            chunks=[create_chunk(0)],
            client=client,
            batch_size=0,
        )


def test_embed_chunks_rejects_duplicate_chunk_ids() -> None:
    client = FakeOpenAI()
    chunks = [
        create_chunk(0),
        create_chunk(0),
    ]

    with pytest.raises(
        ValueError,
        match="unique chunk IDs",
    ):
        embed_chunks(
            chunks=chunks,
            client=client,
        )