import argparse
import logging
from pathlib import Path

import httpx

from src.ingestion.downloader import download_document
from src.ingestion.source_loader import load_document_sources

LOGGER = logging.getLogger(__name__)

DEFAULT_MANIFEST = Path("data/sources/openai_docs.json")
DEFAULT_OUTPUT_DIRECTORY = Path("data/raw/openai_docs")


def ingest_sources(
    manifest_path: Path,
    output_directory: Path,
    client: httpx.Client,
) -> list[Path]:
    """Download every enabled source from a validated manifest."""

    sources = load_document_sources(manifest_path)
    downloaded_paths: list[Path] = []

    for source in sources:
        if not source.enabled:
            continue

        output_path = download_document(
            source=source,
            output_directory=output_directory,
            client=client,
        )
        downloaded_paths.append(output_path)

    return downloaded_paths


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download documents for the RAG knowledge base."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the document-source manifest.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory where downloaded documents will be stored.",
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
        downloaded_paths = ingest_sources(
            manifest_path=arguments.manifest,
            output_directory=arguments.output_directory,
            client=client,
        )

    for path in downloaded_paths:
        LOGGER.info("Downloaded %s", path)

    LOGGER.info("Downloaded %d document(s)", len(downloaded_paths))


if __name__ == "__main__":
    main()