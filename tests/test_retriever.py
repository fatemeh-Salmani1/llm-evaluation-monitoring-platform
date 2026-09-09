from types import SimpleNamespace

import pytest

from src.retrieval.embedder import ChunkEmbedding
from src.retrieval.models import DocumentChunk
from src.retrieval.retriever import retrieve_chunks


class FakeEmbeddings:
    """Fake embeddings endpoint for retrieval tests."""

    def __init__(
        self,
        query_vector: list[float],
        response_index: int = 0,
    ) -> None:
        self.query_vector = query_vector
        self.response_index = response_index
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
                    index=self.response_index,
                    embedding=self.query_vector,
                )
            ]
        )


class FakeOpenAI:
    """Fake OpenAI client for retrieval tests."""

    def __init__(
        self,
        query_vector: list[float],
        response_index: int = 0,
    ) -> None:
        self.embeddings = FakeEmbeddings(
            query_vector=query_vector,
            response_index=response_index,
        )


def create_chunk(position: int) -> DocumentChunk:
    """Create a document chunk for retrieval tests."""

    return DocumentChunk(
        chunk_id=f"openai-evals-guide-chunk-{position:04d}",
        source_id="openai-evals-guide",
        position=position,
        heading_path=["Working with evals"],
        content=f"Evaluation guidance number {position}.",
        token_count=5,
    )


def create_embedding(
    position: int,
    vector: list[float],
    model: str = "text-embedding-3-small",
) -> ChunkEmbedding:
    """Create a chunk embedding for retrieval tests."""

    return ChunkEmbedding(
        chunk_id=f"openai-evals-guide-chunk-{position:04d}",
        source_id="openai-evals-guide",
        embedding_model=model,
        embedding=vector,
    )


def test_retrieve_chunks_returns_ranked_document_content() -> None:
    client = FakeOpenAI(query_vector=[1.0, 0.0])
    chunks = [
        create_chunk(0),
        create_chunk(1),
        create_chunk(2),
    ]
    embeddings = [
        create_embedding(0, [0.0, 1.0]),
        create_embedding(1, [1.0, 0.0]),
        create_embedding(2, [0.8, 0.2]),
    ]

    results = retrieve_chunks(
        question="How do evals test model outputs?",
        chunks=chunks,
        chunk_embeddings=embeddings,
        client=client,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].rank == 1
    assert results[0].chunk == chunks[1]
    assert results[0].similarity_score == pytest.approx(1.0)
    assert results[1].rank == 2
    assert results[1].chunk == chunks[2]


def test_retrieve_chunks_normalizes_question() -> None:
    client = FakeOpenAI(query_vector=[1.0, 0.0])
    chunks = [create_chunk(0)]
    embeddings = [
        create_embedding(0, [1.0, 0.0])
    ]

    retrieve_chunks(
        question="  What is an evaluation?  ",
        chunks=chunks,
        chunk_embeddings=embeddings,
        client=client,
    )

    assert client.embeddings.calls[0]["input"] == [
        "What is an evaluation?"
    ]


def test_retrieve_chunks_rejects_empty_question() -> None:
    client = FakeOpenAI(query_vector=[1.0, 0.0])

    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        retrieve_chunks(
            question="   ",
            chunks=[create_chunk(0)],
            chunk_embeddings=[
                create_embedding(0, [1.0, 0.0])
            ],
            client=client,
        )


def test_retrieve_chunks_requires_matching_chunk_ids() -> None:
    client = FakeOpenAI(query_vector=[1.0, 0.0])
    chunks = [
        create_chunk(0),
        create_chunk(1),
    ]
    embeddings = [
        create_embedding(0, [1.0, 0.0])
    ]

    with pytest.raises(
        ValueError,
        match="matching chunk IDs",
    ):
        retrieve_chunks(
            question="What is an eval?",
            chunks=chunks,
            chunk_embeddings=embeddings,
            client=client,
        )


def test_retrieve_chunks_requires_matching_models() -> None:
    client = FakeOpenAI(query_vector=[1.0, 0.0])
    chunks = [create_chunk(0)]
    embeddings = [
        create_embedding(
            position=0,
            vector=[1.0, 0.0],
            model="different-model",
        )
    ]

    with pytest.raises(
        ValueError,
        match="must use the same model",
    ):
        retrieve_chunks(
            question="What is an eval?",
            chunks=chunks,
            chunk_embeddings=embeddings,
            client=client,
        )


def test_retrieve_chunks_rejects_unexpected_api_response() -> None:
    client = FakeOpenAI(
        query_vector=[1.0, 0.0],
        response_index=1,
    )
    chunks = [create_chunk(0)]
    embeddings = [
        create_embedding(0, [1.0, 0.0])
    ]

    with pytest.raises(
        RuntimeError,
        match="does not match the question",
    ):
        retrieve_chunks(
            question="What is an eval?",
            chunks=chunks,
            chunk_embeddings=embeddings,
            client=client,
        )