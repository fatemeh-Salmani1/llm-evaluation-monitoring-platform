from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from src.retrieval.embedder import ChunkEmbedding


def validate_embeddings(
    embeddings: Sequence[ChunkEmbedding],
) -> None:
    """Validate an embedding collection before using or storing it."""

    if not embeddings:
        raise ValueError("At least one chunk embedding is required")

    chunk_ids = [item.chunk_id for item in embeddings]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Chunk embeddings must have unique chunk IDs")

    models = {item.embedding_model for item in embeddings}

    if len(models) != 1:
        raise ValueError(
            "Chunk embeddings must use the same embedding model"
        )

    dimensions = {len(item.embedding) for item in embeddings}

    if len(dimensions) != 1:
        raise ValueError(
            "Chunk embeddings must have the same vector dimensions"
        )


def write_embeddings(
    embeddings: Sequence[ChunkEmbedding],
    output_path: Path,
) -> Path:
    """Write chunk embeddings to a JSONL file."""

    validate_embeddings(embeddings)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serialized_records = [
        item.model_dump_json()
        for item in embeddings
    ]

    output_path.write_text(
        "\n".join(serialized_records) + "\n",
        encoding="utf-8",
    )

    return output_path


def load_embeddings(
    input_path: Path,
) -> list[ChunkEmbedding]:
    """Load and validate chunk embeddings from a JSONL file."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Embedding file does not exist: {input_path}"
        )

    embeddings: list[ChunkEmbedding] = []

    with input_path.open(
        mode="r",
        encoding="utf-8",
    ) as embedding_file:
        for line_number, line in enumerate(
            embedding_file,
            start=1,
        ):
            record = line.strip()

            if not record:
                continue

            try:
                embedding = ChunkEmbedding.model_validate_json(
                    record
                )
            except ValidationError as error:
                raise ValueError(
                    "Invalid embedding record "
                    f"at line {line_number}"
                ) from error

            embeddings.append(embedding)

    validate_embeddings(embeddings)

    return embeddings