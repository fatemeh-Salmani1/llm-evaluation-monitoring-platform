from pathlib import Path

import pytest

from src.retrieval.embedder import ChunkEmbedding
from src.retrieval.embedding_storage import (
    load_embeddings,
    write_embeddings,
)


def create_embedding(
    position: int,
    vector: list[float] | None = None,
    model: str = "text-embedding-3-small",
) -> ChunkEmbedding:
    """Create an embedding record for storage tests."""

    return ChunkEmbedding(
        chunk_id=f"openai-evals-guide-chunk-{position:04d}",
        source_id="openai-evals-guide",
        embedding_model=model,
        embedding=vector or [0.1, 0.2, 0.3],
    )


def test_embedding_storage_round_trip(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "embeddings.jsonl"
    embeddings = [
        create_embedding(0),
        create_embedding(1),
    ]

    written_path = write_embeddings(
        embeddings=embeddings,
        output_path=output_path,
    )
    loaded_embeddings = load_embeddings(written_path)

    assert written_path == output_path
    assert loaded_embeddings == embeddings
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 2


def test_write_embeddings_rejects_empty_collection(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="At least one chunk embedding",
    ):
        write_embeddings(
            embeddings=[],
            output_path=tmp_path / "embeddings.jsonl",
        )


def test_write_embeddings_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    embeddings = [
        create_embedding(0),
        create_embedding(0),
    ]

    with pytest.raises(
        ValueError,
        match="unique chunk IDs",
    ):
        write_embeddings(
            embeddings=embeddings,
            output_path=tmp_path / "embeddings.jsonl",
        )


def test_write_embeddings_rejects_mixed_models(
    tmp_path: Path,
) -> None:
    embeddings = [
        create_embedding(
            position=0,
            model="text-embedding-3-small",
        ),
        create_embedding(
            position=1,
            model="different-model",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="same embedding model",
    ):
        write_embeddings(
            embeddings=embeddings,
            output_path=tmp_path / "embeddings.jsonl",
        )


def test_write_embeddings_rejects_mixed_dimensions(
    tmp_path: Path,
) -> None:
    embeddings = [
        create_embedding(
            position=0,
            vector=[0.1, 0.2, 0.3],
        ),
        create_embedding(
            position=1,
            vector=[0.1, 0.2],
        ),
    ]

    with pytest.raises(
        ValueError,
        match="same vector dimensions",
    ):
        write_embeddings(
            embeddings=embeddings,
            output_path=tmp_path / "embeddings.jsonl",
        )


def test_load_embeddings_rejects_invalid_record(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "invalid.jsonl"
    input_path.write_text(
        '{"chunk_id": "missing-fields"}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid embedding record at line 1",
    ):
        load_embeddings(input_path)


def test_load_embeddings_rejects_missing_file(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "missing.jsonl"

    with pytest.raises(
        FileNotFoundError,
        match="Embedding file does not exist",
    ):
        load_embeddings(input_path)