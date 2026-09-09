import argparse
import logging
from pathlib import Path

import httpx

from src.ingestion.cleaner import clean_markdown
from src.ingestion.downloader import download_document
from src.ingestion.metadata import (
    build_document_metadata,
    write_document_metadata,
)
from src.ingestion.source_loader import load_document_sources
from src.retrieval.chunker import chunk_markdown
from src.retrieval.storage import write_chunks_jsonl

LOGGER = logging.getLogger(__name__)

DEFAULT_MANIFEST = Path("data/sources/openai_docs.json")
DEFAULT_RAW_DIRECTORY = Path("data/raw/openai_docs")
DEFAULT_PROCESSED_DIRECTORY = Path("data/processed/openai_docs")
DEFAULT_CHUNKS_DIRECTORY = Path("data/processed/chunks")


def ingest_sources(
    manifest_path: Path,
    raw_output_directory: Path,
    processed_output_directory: Path,
    chunks_output_directory: Path,
    client: httpx.Client,
) -> list[Path]:
    """Download, document, clean and chunk every enabled source."""

    sources = load_document_sources(manifest_path)
    processed_paths: list[Path] = []

    for source in sources:
        if not source.enabled:
            continue

        raw_path = download_document(
            source=source,
            output_directory=raw_output_directory,
            client=client,
        )

        metadata = build_document_metadata(
            source=source,
            document_path=raw_path,
        )
        metadata_path = (
            raw_output_directory
            / f"{source.source_id}.metadata.json"
        )
        write_document_metadata(
            metadata=metadata,
            output_path=metadata_path,
        )

        raw_content = raw_path.read_text(encoding="utf-8")
        cleaned_content = clean_markdown(raw_content)

        processed_output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        processed_path = (
            processed_output_directory / f"{source.source_id}.md"
        )
        processed_path.write_text(
            cleaned_content,
            encoding="utf-8",
        )

        chunks = chunk_markdown(
            source_id=source.source_id,
            content=cleaned_content,
        )
        chunks_path = (
            chunks_output_directory / f"{source.source_id}.jsonl"
        )
        write_chunks_jsonl(
            chunks=chunks,
            output_path=chunks_path,
        )

        processed_paths.append(processed_path)

    return processed_paths


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest documents for the RAG knowledge base."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the document-source manifest.",
    )
    parser.add_argument(
        "--raw-output-directory",
        type=Path,
        default=DEFAULT_RAW_DIRECTORY,
        help="Directory for original downloaded documents.",
    )
    parser.add_argument(
        "--processed-output-directory",
        type=Path,
        default=DEFAULT_PROCESSED_DIRECTORY,
        help="Directory for cleaned documents.",
    )
    parser.add_argument(
        "--chunks-output-directory",
        type=Path,
        default=DEFAULT_CHUNKS_DIRECTORY,
        help="Directory for prepared document chunks.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    arguments = parse_arguments()

    with httpx.Client(
        timeout=30.0,
        follow_redirects=True,
        headers={
            "User-Agent": "llm-evaluation-monitoring-platform/0.1"
        },
    ) as client:
        processed_paths = ingest_sources(
            manifest_path=arguments.manifest,
            raw_output_directory=arguments.raw_output_directory,
            processed_output_directory=(
                arguments.processed_output_directory
            ),
            chunks_output_directory=(
                arguments.chunks_output_directory
            ),
            client=client,
        )

    for path in processed_paths:
        LOGGER.info("Prepared %s", path)

    LOGGER.info("Ingested %d document(s)", len(processed_paths))


if __name__ == "__main__":
    main()