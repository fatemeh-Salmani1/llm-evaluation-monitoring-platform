import json
from pathlib import Path

from pydantic import ValidationError

from src.ingestion.models import DocumentSource


def load_document_sources(path: Path) -> list[DocumentSource]:
    """Load and validate document sources from a JSON manifest."""

    if not path.exists():
        raise FileNotFoundError(f"Source manifest does not exist: {path}")

    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in source manifest: {path}") from error

    if not isinstance(records, list):
        raise TypeError("Source manifest must contain a JSON array")

    try:
        sources = [
            DocumentSource.model_validate(record)
            for record in records
        ]
    except ValidationError as error:
        raise ValueError(
            f"Invalid document source in manifest: {path}"
        ) from error

    source_ids = [source.source_id for source in sources]

    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Source manifest contains duplicate source IDs")

    return sources