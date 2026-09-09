import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.retrieval.embedder import ChunkEmbedding
from src.retrieval.embedding_storage import validate_embeddings


class SearchResult(BaseModel):
    """A document chunk returned by semantic search."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    similarity_score: float = Field(ge=-1.0, le=1.0)


def cosine_similarity(
    left_vector: Sequence[float],
    right_vector: Sequence[float],
) -> float:
    """Calculate cosine similarity between two vectors."""

    if not left_vector or not right_vector:
        raise ValueError("Similarity vectors cannot be empty")

    if len(left_vector) != len(right_vector):
        raise ValueError(
            "Similarity vectors must have the same dimensions"
        )

    left_norm = math.sqrt(
        sum(value * value for value in left_vector)
    )
    right_norm = math.sqrt(
        sum(value * value for value in right_vector)
    )

    if left_norm == 0 or right_norm == 0:
        raise ValueError("Similarity vectors cannot be zero vectors")

    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(
            left_vector,
            right_vector,
            strict=True,
        )
    )

    similarity = dot_product / (left_norm * right_norm)

    return max(-1.0, min(1.0, similarity))


def search_embeddings(
    query_embedding: Sequence[float],
    chunk_embeddings: Sequence[ChunkEmbedding],
    top_k: int = 5,
) -> list[SearchResult]:
    """Return the document chunks most similar to a query vector."""

    if top_k < 1:
        raise ValueError("top_k must be greater than zero")

    validate_embeddings(chunk_embeddings)

    expected_dimensions = len(
        chunk_embeddings[0].embedding
    )

    if len(query_embedding) != expected_dimensions:
        raise ValueError(
            "Query and chunk embeddings must have the same dimensions"
        )

    scored_embeddings = [
        (
            cosine_similarity(
                query_embedding,
                item.embedding,
            ),
            item,
        )
        for item in chunk_embeddings
    ]

    ranked_embeddings = sorted(
        scored_embeddings,
        key=lambda result: (
            -result[0],
            result[1].chunk_id,
        ),
    )

    return [
        SearchResult(
            rank=rank,
            chunk_id=item.chunk_id,
            source_id=item.source_id,
            similarity_score=score,
        )
        for rank, (score, item) in enumerate(
            ranked_embeddings[:top_k],
            start=1,
        )
    ]