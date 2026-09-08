from pathlib import Path

import httpx

from src.ingestion.models import DocumentSource


def download_document(
    source: DocumentSource,
    output_directory: Path,
    client: httpx.Client,
) -> Path:
    """Download one Markdown document and return its local path."""

    response = client.get(str(source.markdown_url))
    response.raise_for_status()

    content = response.text.strip()

    if not content:
        raise ValueError(
            f"Downloaded document is empty: {source.source_id}"
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{source.source_id}.md"
    output_path.write_text(f"{content}\n", encoding="utf-8")

    return output_path