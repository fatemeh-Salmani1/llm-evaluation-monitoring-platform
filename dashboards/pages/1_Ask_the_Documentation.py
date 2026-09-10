from pathlib import Path

import streamlit as st
from openai import OpenAI, OpenAIError

from src.config.settings import get_settings
from src.generation.run_answer import answer_question
from src.retrieval.models import DocumentChunk
from src.retrieval.run_retrieval import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_TOP_K,
)
from src.retrieval.storage import load_chunks_jsonl

EXAMPLE_QUESTIONS = [
    "What is an eval?",
    "Why are evals useful?",
    "What does a grader do?",
    "What is test data used for?",
    "Where can I view eval results?",
]


def find_source_chunks(
    chunk_ids: list[str],
    chunks_path: Path,
) -> list[DocumentChunk]:
    """Find retrieved document chunks by their identifiers."""

    chunks = load_chunks_jsonl(chunks_path)
    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunk in chunks
    }

    return [
        chunks_by_id[chunk_id]
        for chunk_id in chunk_ids
        if chunk_id in chunks_by_id
    ]


def render_sources(
    chunk_ids: list[str],
    chunks_path: Path,
) -> None:
    """Display the document chunks supporting an answer."""

    source_chunks = find_source_chunks(
        chunk_ids=chunk_ids,
        chunks_path=chunks_path,
    )

    st.subheader("Sources")

    for chunk in source_chunks:
        heading = " > ".join(chunk.heading_path)

        with st.expander(
            f"{chunk.chunk_id} — {heading}"
        ):
            st.caption(
                f"Source: {chunk.source_id} · "
                f"Tokens: {chunk.token_count}"
            )
            st.markdown(chunk.content)


def main() -> None:
    """Render the grounded documentation assistant."""

    st.set_page_config(
        page_title="Ask About OpenAI Evals",
        page_icon="💬",
        layout="wide",
    )

    st.title("Ask About OpenAI Evals")
    st.caption(
        "Ask a question and receive an answer grounded "
        "in the OpenAI evals documentation."
    )

    settings = get_settings()

    with st.sidebar:
        st.header("Settings")

        with st.expander("Advanced settings"):
            top_k = st.slider(
                "Number of sources",
                min_value=1,
                max_value=8,
                value=DEFAULT_TOP_K,
                help=(
                    "Choose how many relevant document "
                    "sections are used to generate the answer."
                ),
            )

    chunks_path = DEFAULT_CHUNKS_PATH
    embeddings_path = DEFAULT_EMBEDDINGS_PATH

    if settings.openai_api_key is None:
        st.error(
            "The application is not configured to generate "
            "answers. Add OPENAI_API_KEY to the local .env file."
        )
        return

    example = st.selectbox(
        "Example questions",
        options=[""] + EXAMPLE_QUESTIONS,
        format_func=lambda value: (
            "Choose an example"
            if not value
            else value
        ),
    )

    with st.form("question-form"):
        question = st.text_area(
            "Your question",
            value=example,
            height=120,
            placeholder=(
                "Ask a question about OpenAI evals..."
            ),
        )

        submitted = st.form_submit_button(
            "Generate answer",
            type="primary",
        )

    if not submitted:
        st.info(
            "Enter a question or select an example to begin."
        )
        return

    if not question.strip():
        st.warning("Please enter a question.")
        return

    if not chunks_path.exists():
        st.error(
            "The document chunks are not available. "
            f"Expected file: {chunks_path}"
        )
        return

    if not embeddings_path.exists():
        st.error(
            "The document embeddings are not available. "
            f"Expected file: {embeddings_path}"
        )
        return

    client = OpenAI(
        api_key=(
            settings.openai_api_key.get_secret_value()
        )
    )

    try:
        with st.spinner(
            "Finding relevant information "
            "and generating an answer..."
        ):
            result = answer_question(
                question=question,
                chunks_path=chunks_path,
                embeddings_path=embeddings_path,
                client=client,
                top_k=top_k,
            )
    except (
        OpenAIError,
        OSError,
        ValueError,
        RuntimeError,
    ) as error:
        st.error(
            f"Unable to generate an answer: {error}"
        )
        return

    st.subheader("Answer")

    with st.chat_message("user"):
        st.write(result.question)

    with st.chat_message("assistant"):
        st.markdown(result.answer)
        st.caption(
            f"Generated with {result.model}"
        )

    render_sources(
        chunk_ids=result.source_chunk_ids,
        chunks_path=chunks_path,
    )


if __name__ == "__main__":
    main()