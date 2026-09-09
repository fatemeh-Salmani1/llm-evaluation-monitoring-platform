import pytest

from src.retrieval.embedder import ChunkEmbedding
from src.retrieval.search import (
    cosine_similarity,
    search_embeddings,
)


def create_embedding(
    chunk_id: str,
    vector: list[float],
) -> ChunkEmbedding:
    """Create a chunk embedding for semantic-search tests."""

    return ChunkEmbedding(
        chunk_id=chunk_id,
        source_id="openai-evals-guide",
        embedding_model="text-embedding-3-small",
        embedding=vector,
    )


def test_cosine_similarity_identifies_equal_vectors() -> None:
    similarity = cosine_similarity(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    )

    assert similarity == pytest.approx(1.0)


def test_cosine_similarity_identifies_orthogonal_vectors() -> None:
    similarity = cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    assert similarity == pytest.approx(0.0)


def test_search_embeddings_returns_most_similar_chunks() -> None:
    embeddings = [
        create_embedding(
            "chunk-0000",
            [0.0, 1.0],
        ),
        create_embedding(
            "chunk-0001",
            [0.8, 0.2],
        ),
        create_embedding(
            "chunk-0002",
            [1.0, 0.0],
        ),
    ]

    results = search_embeddings(
        query_embedding=[1.0, 0.0],
        chunk_embeddings=embeddings,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk_id == "chunk-0002"
    assert results[0].rank == 1
    assert results[0].similarity_score == pytest.approx(1.0)
    assert results[1].chunk_id == "chunk-0001"
    assert results[1].rank == 2


def test_search_embeddings_uses_deterministic_tie_breaking() -> None:
    embeddings = [
        create_embedding(
            "chunk-0002",
            [1.0, 0.0],
        ),
        create_embedding(
            "chunk-0001",
            [1.0, 0.0],
        ),
    ]

    results = search_embeddings(
        query_embedding=[1.0, 0.0],
        chunk_embeddings=embeddings,
    )

    assert [result.chunk_id for result in results] == [
        "chunk-0001",
        "chunk-0002",
    ]


def test_search_embeddings_rejects_invalid_top_k() -> None:
    embeddings = [
        create_embedding(
            "chunk-0000",
            [1.0, 0.0],
        )
    ]

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        search_embeddings(
            query_embedding=[1.0, 0.0],
            chunk_embeddings=embeddings,
            top_k=0,
        )


def test_search_embeddings_rejects_dimension_mismatch() -> None:
    embeddings = [
        create_embedding(
            "chunk-0000",
            [1.0, 0.0, 0.0],
        )
    ]

    with pytest.raises(
        ValueError,
        match="same dimensions",
    ):
        search_embeddings(
            query_embedding=[1.0, 0.0],
            chunk_embeddings=embeddings,
        )


def test_cosine_similarity_rejects_zero_vectors() -> None:
    with pytest.raises(
        ValueError,
        match="zero vectors",
    ):
        cosine_similarity(
            [0.0, 0.0],
            [1.0, 0.0],
        )