from pathlib import Path

import pytest

from src.retrieval.models import DocumentChunk
from src.retrieval.storage import (
    load_chunks_jsonl,
    write_chunks_jsonl,
)


def create_chunk(position: int = 0) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=(
            f"openai-evals-guide-chunk-{position:04d}"
        ),
        source_id="openai-evals-guide",
        position=position,
        heading_path=["Working with evals"],
        content="Evaluation guidance.",
        token_count=4,
    )


def test_chunk_storage_round_trip(tmp_path: Path) -> None:
    output_path = tmp_path / "chunks.jsonl"
    original_chunks = [
        create_chunk(0),
        create_chunk(1),
    ]

    returned_path = write_chunks_jsonl(
        chunks=original_chunks,
        output_path=output_path,
    )
    loaded_chunks = load_chunks_jsonl(output_path)

    assert returned_path == output_path
    assert loaded_chunks == original_chunks


def test_write_chunks_rejects_empty_collection(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="empty chunk"):
        write_chunks_jsonl(
            chunks=[],
            output_path=tmp_path / "chunks.jsonl",
        )


def test_write_chunks_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    duplicate_chunk = create_chunk(0)

    with pytest.raises(ValueError, match="duplicate IDs"):
        write_chunks_jsonl(
            chunks=[duplicate_chunk, duplicate_chunk],
            output_path=tmp_path / "chunks.jsonl",
        )


def test_load_chunks_rejects_invalid_record(
    tmp_path: Path,
) -> None:
    chunk_path = tmp_path / "chunks.jsonl"
    chunk_path.write_text(
        '{"chunk_id": "invalid"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 1"):
        load_chunks_jsonl(chunk_path)