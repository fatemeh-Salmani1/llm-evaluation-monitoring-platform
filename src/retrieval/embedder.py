from collections.abc import Sequence

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from src.retrieval.models import DocumentChunk

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_BATCH_SIZE = 100


class ChunkEmbedding(BaseModel):
    """Embedding vector and lineage information for a document chunk."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)


def embed_chunks(
    chunks: Sequence[DocumentChunk],
    client: OpenAI,
    model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[ChunkEmbedding]:
    """Create embeddings for document chunks in batches."""

    if not chunks:
        raise ValueError("At least one document chunk is required")

    if batch_size < 1:
        raise ValueError("Batch size must be greater than zero")

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Document chunks must have unique chunk IDs")

    embedded_chunks: list[ChunkEmbedding] = []

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]

        response = client.embeddings.create(
            model=model,
            input=[chunk.content for chunk in batch],
        )

        response_items = sorted(
            response.data,
            key=lambda item: item.index,
        )

        expected_indices = list(range(len(batch)))
        actual_indices = [item.index for item in response_items]

        if actual_indices != expected_indices:
            raise RuntimeError(
                "Embedding response does not match the requested batch"
            )

        for chunk, response_item in zip(
            batch,
            response_items,
            strict=True,
        ):
            embedded_chunks.append(
                ChunkEmbedding(
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    embedding_model=model,
                    embedding=response_item.embedding,
                )
            )

    return embedded_chunks