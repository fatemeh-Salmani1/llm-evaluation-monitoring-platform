import argparse
import logging
from pathlib import Path

from openai import OpenAI

from src.config.settings import get_settings
from src.retrieval.embedder import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    embed_chunks,
)
from src.retrieval.embedding_storage import write_embeddings
from src.retrieval.storage import load_chunks_jsonl

LOGGER = logging.getLogger(__name__)

DEFAULT_CHUNKS_PATH = Path(
    "data/processed/chunks/openai-evals-guide.jsonl"
)
DEFAULT_EMBEDDINGS_PATH = Path(
    "data/processed/embeddings/openai-evals-guide.jsonl"
)


def generate_and_store_embeddings(
    chunks_path: Path,
    output_path: Path,
    client: OpenAI,
    model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    overwrite: bool = False,
) -> Path:
    """Generate embeddings for stored chunks and save them as JSONL."""

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Embedding file already exists: {output_path}. "
            "Use overwrite=True to replace it."
        )

    chunks = load_chunks_jsonl(chunks_path)

    embeddings = embed_chunks(
        chunks=chunks,
        client=client,
        model=model,
        batch_size=batch_size,
    )

    written_path = write_embeddings(
        embeddings=embeddings,
        output_path=output_path,
    )

    LOGGER.info(
        "Stored %s embedding(s) in %s",
        len(embeddings),
        written_path,
    )

    return written_path


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate OpenAI embeddings for stored document chunks."
        )
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to the input chunk JSONL file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_EMBEDDINGS_PATH,
        help="Path for the generated embedding JSONL file.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="OpenAI embedding model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of chunks sent in each API request.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing embedding file.",
    )

    return parser


def main() -> None:
    """Run the document embedding pipeline."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    arguments = build_argument_parser().parse_args()
    settings = get_settings()

    if settings.openai_api_key is None:
        raise RuntimeError(
            "OPENAI_API_KEY is required to generate embeddings"
        )

    client = OpenAI(
        api_key=settings.openai_api_key.get_secret_value()
    )

    output_path = generate_and_store_embeddings(
        chunks_path=arguments.chunks_path,
        output_path=arguments.output_path,
        client=client,
        model=arguments.model,
        batch_size=arguments.batch_size,
        overwrite=arguments.overwrite,
    )

    LOGGER.info("Embedding pipeline completed: %s", output_path)


if __name__ == "__main__":
    main()