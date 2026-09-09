import json
from pathlib import Path

from pydantic import ValidationError

from src.retrieval.models import DocumentChunk


def write_chunks_jsonl(
    chunks: list[DocumentChunk],
    output_path: Path,
) -> Path:
    """Write validated document chunks to a JSONL file."""

    if not chunks:
        raise ValueError("Cannot write an empty chunk collection")

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Chunk collection contains duplicate IDs")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        chunk.model_dump_json() for chunk in chunks
    )
    output_path.write_text(f"{content}\n", encoding="utf-8")

    return output_path


def load_chunks_jsonl(path: Path) -> list[DocumentChunk]:
    """Load and validate document chunks from a JSONL file."""

    if not path.exists():
        raise FileNotFoundError(f"Chunk file does not exist: {path}")

    chunks: list[DocumentChunk] = []
    chunk_ids: set[str] = set()

    with path.open(encoding="utf-8") as chunk_file:
        for line_number, line in enumerate(chunk_file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                chunk = DocumentChunk.model_validate(record)
            except (json.JSONDecodeError, ValidationError) as error:
                message = (
                    f"Invalid chunk record in {path} "
                    f"at line {line_number}"
                )
                raise ValueError(message) from error

            if chunk.chunk_id in chunk_ids:
                raise ValueError(
                    f"Duplicate chunk_id '{chunk.chunk_id}' "
                    f"in {path} at line {line_number}"
                )

            chunk_ids.add(chunk.chunk_id)
            chunks.append(chunk)

    if not chunks:
        raise ValueError(f"Chunk file contains no records: {path}")

    return chunks