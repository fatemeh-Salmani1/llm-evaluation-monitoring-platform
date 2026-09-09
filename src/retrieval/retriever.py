from collections.abc import Sequence

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from src.retrieval.embedder import (
    DEFAULT_EMBEDDING_MODEL,
    ChunkEmbedding,
)
from src.retrieval.models import DocumentChunk
from src.retrieval.search import search_embeddings


class RetrievedChunk(BaseModel):
    """A document chunk selected for a user question."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=1)
    similarity_score: float = Field(ge=-1.0, le=1.0)
    chunk: DocumentChunk


def retrieve_chunks(
    question: str,
    chunks: Sequence[DocumentChunk],
    chunk_embeddings: Sequence[ChunkEmbedding],
    client: OpenAI,
    model: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Embed a question and return its most relevant document chunks."""

    normalized_question = question.strip()

    if not normalized_question:
        raise ValueError("Question cannot be empty")

    if not chunks:
        raise ValueError("At least one document chunk is required")

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Document chunks must have unique chunk IDs")

    chunk_id_set = set(chunk_ids)
    embedding_id_set = {
        embedding.chunk_id
        for embedding in chunk_embeddings
    }

    if chunk_id_set != embedding_id_set:
        raise ValueError(
            "Document chunks and chunk embeddings must have "
            "matching chunk IDs"
        )

    embedding_models = {
        embedding.embedding_model
        for embedding in chunk_embeddings
    }

    if embedding_models != {model}:
        raise ValueError(
            "Question and document embeddings must use the same model"
        )

    response = client.embeddings.create(
        model=model,
        input=[normalized_question],
    )

    if len(response.data) != 1 or response.data[0].index != 0:
        raise RuntimeError(
            "Embedding response does not match the question"
        )

    query_embedding = response.data[0].embedding

    search_results = search_embeddings(
        query_embedding=query_embedding,
        chunk_embeddings=chunk_embeddings,
        top_k=top_k,
    )

    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunk in chunks
    }

    return [
        RetrievedChunk(
            rank=result.rank,
            similarity_score=result.similarity_score,
            chunk=chunks_by_id[result.chunk_id],
        )
        for result in search_results
    ]