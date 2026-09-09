import argparse
from pathlib import Path

from openai import OpenAI

from src.config.settings import get_settings
from src.generation.answerer import (
    DEFAULT_GENERATION_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    GroundedAnswer,
    generate_grounded_answer,
)
from src.retrieval.embedder import DEFAULT_EMBEDDING_MODEL
from src.retrieval.run_retrieval import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_TOP_K,
    retrieve_from_files,
)


def answer_question(
    question: str,
    chunks_path: Path,
    embeddings_path: Path,
    client: OpenAI,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    generation_model: str = DEFAULT_GENERATION_MODEL,
    top_k: int = DEFAULT_TOP_K,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> GroundedAnswer:
    """Retrieve relevant chunks and generate a grounded answer."""

    retrieved_chunks = retrieve_from_files(
        question=question,
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        client=client,
        model=embedding_model,
        top_k=top_k,
    )

    return generate_grounded_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        client=client,
        model=generation_model,
        max_output_tokens=max_output_tokens,
    )


def format_grounded_answer(
    result: GroundedAnswer,
) -> str:
    """Format a grounded answer for terminal output."""

    source_lines = "\n".join(
        f"- {chunk_id}"
        for chunk_id in result.source_chunk_ids
    )

    return (
        f"Question:\n{result.question}\n\n"
        f"Answer:\n{result.answer}\n\n"
        f"Generation model:\n{result.model}\n\n"
        f"Retrieved source chunks:\n{source_lines}"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Answer a question using retrieved OpenAI documentation."
        )
    )
    parser.add_argument(
        "question",
        help="Question to answer from the document collection.",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to the stored document chunks.",
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=DEFAULT_EMBEDDINGS_PATH,
        help="Path to the stored document embeddings.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Model used for the question embedding.",
    )
    parser.add_argument(
        "--generation-model",
        default=DEFAULT_GENERATION_MODEL,
        help="Model used to generate the grounded answer.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of document chunks supplied as context.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum number of generated output tokens.",
    )

    return parser


def main() -> None:
    """Run the complete retrieval and generation workflow."""

    arguments = build_argument_parser().parse_args()
    settings = get_settings()

    if settings.openai_api_key is None:
        raise RuntimeError(
            "OPENAI_API_KEY is required to answer questions"
        )

    client = OpenAI(
        api_key=settings.openai_api_key.get_secret_value()
    )

    result = answer_question(
        question=arguments.question,
        chunks_path=arguments.chunks_path,
        embeddings_path=arguments.embeddings_path,
        client=client,
        embedding_model=arguments.embedding_model,
        generation_model=arguments.generation_model,
        top_k=arguments.top_k,
        max_output_tokens=arguments.max_output_tokens,
    )

    print(format_grounded_answer(result))


if __name__ == "__main__":
    main()