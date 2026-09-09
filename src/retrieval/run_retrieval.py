import argparse
from collections.abc import Sequence
from pathlib import Path

from openai import OpenAI

from src.config.settings import get_settings
from src.retrieval.embedder import DEFAULT_EMBEDDING_MODEL
from src.retrieval.embedding_storage import load_embeddings
from src.retrieval.retriever import (
    RetrievedChunk,
    retrieve_chunks,
)
from src.retrieval.storage import load_chunks_jsonl

DEFAULT_CHUNKS_PATH = Path(
    "data/processed/chunks/openai-evals-guide.jsonl"
)
DEFAULT_EMBEDDINGS_PATH = Path(
    "data/processed/embeddings/openai-evals-guide.jsonl"
)
DEFAULT_TOP_K = 3
DEFAULT_PREVIEW_CHARACTERS = 500


def retrieve_from_files(
    question: str,
    chunks_path: Path,
    embeddings_path: Path,
    client: OpenAI,
    model: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievedChunk]:
    """Retrieve relevant chunks using stored files."""

    chunks = load_chunks_jsonl(chunks_path)
    embeddings = load_embeddings(embeddings_path)

    return retrieve_chunks(
        question=question,
        chunks=chunks,
        chunk_embeddings=embeddings,
        client=client,
        model=model,
        top_k=top_k,
    )


def format_retrieval_results(
    results: Sequence[RetrievedChunk],
    preview_characters: int = DEFAULT_PREVIEW_CHARACTERS,
) -> str:
    """Format retrieved chunks for terminal output."""

    if preview_characters < 1:
        raise ValueError(
            "preview_characters must be greater than zero"
        )

    if not results:
        return "No matching chunks found."

    formatted_results: list[str] = []

    for result in results:
        heading = " > ".join(result.chunk.heading_path)
        content_preview = " ".join(
            result.chunk.content.split()
        )

        if len(content_preview) > preview_characters:
            content_preview = (
                content_preview[:preview_characters].rstrip()
                + "..."
            )

        formatted_results.append(
            "\n".join(
                [
                    f"Rank: {result.rank}",
                    (
                        "Similarity: "
                        f"{result.similarity_score:.4f}"
                    ),
                    f"Chunk ID: {result.chunk.chunk_id}",
                    f"Heading: {heading}",
                    f"Preview: {content_preview}",
                ]
            )
        )

    return "\n\n".join(formatted_results)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for semantic retrieval."""

    parser = argparse.ArgumentParser(
        description=(
            "Search the stored OpenAI documentation chunks."
        )
    )
    parser.add_argument(
        "question",
        help="Question used for semantic document retrieval.",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to the chunk JSONL file.",
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=DEFAULT_EMBEDDINGS_PATH,
        help="Path to the embedding JSONL file.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model used by the stored vectors.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of relevant chunks to return.",
    )
    parser.add_argument(
        "--preview-characters",
        type=int,
        default=DEFAULT_PREVIEW_CHARACTERS,
        help="Maximum preview length for each result.",
    )

    return parser


def main() -> None:
    """Run semantic retrieval from the command line."""

    arguments = build_argument_parser().parse_args()
    settings = get_settings()

    if settings.openai_api_key is None:
        raise RuntimeError(
            "OPENAI_API_KEY is required for semantic retrieval"
        )

    client = OpenAI(
        api_key=settings.openai_api_key.get_secret_value()
    )

    results = retrieve_from_files(
        question=arguments.question,
        chunks_path=arguments.chunks_path,
        embeddings_path=arguments.embeddings_path,
        client=client,
        model=arguments.model,
        top_k=arguments.top_k,
    )

    print(
        format_retrieval_results(
            results=results,
            preview_characters=arguments.preview_characters,
        )
    )


if __name__ == "__main__":
    main()