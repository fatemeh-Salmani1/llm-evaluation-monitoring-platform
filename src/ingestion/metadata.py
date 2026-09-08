import hashlib
from datetime import UTC, datetime
from pathlib import Path

from src.ingestion.models import (
    DocumentSource,
    IngestedDocumentMetadata,
)


def calculate_sha256(path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""

    if not path.exists():
        raise FileNotFoundError(f"Document does not exist: {path}")

    checksum = hashlib.sha256()

    with path.open("rb") as document_file:
        while chunk := document_file.read(8192):
            checksum.update(chunk)

    return checksum.hexdigest()


def build_document_metadata(
    source: DocumentSource,
    document_path: Path,
    retrieved_at: datetime | None = None,
) -> IngestedDocumentMetadata:
    """Build lineage metadata for a downloaded document."""

    if not document_path.exists():
        raise FileNotFoundError(
            f"Downloaded document does not exist: {document_path}"
        )

    content_bytes = document_path.stat().st_size

    if content_bytes == 0:
        raise ValueError("Downloaded document cannot be empty")

    return IngestedDocumentMetadata(
        source_id=source.source_id,
        canonical_url=source.canonical_url,
        local_path=document_path.as_posix(),
        content_sha256=calculate_sha256(document_path),
        content_bytes=content_bytes,
        retrieved_at=retrieved_at or datetime.now(UTC),
    )