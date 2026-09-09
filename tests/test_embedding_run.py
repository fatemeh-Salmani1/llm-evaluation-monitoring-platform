from pathlib import Path
from types import SimpleNamespace

import pytest

from src.retrieval.embedding_storage import load_embeddings
from src.retrieval.models import DocumentChunk
from src.retrieval.run_embeddings import (
    generate_and_store_embeddings,
)
from src.retrieval.storage import write_chunks_jsonl


class FakeEmbeddings:
    """Fake embeddings endpoint for pipeline tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(
        self,
        *,
        model: str,
        input: list[str],
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "model": model,
                "input": input,
            }
        )

        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    index=index,
                    embedding=[
                        float(index + 1),
                        float(len(content)),
                        1.0,
                    ],
                )
                for index, content in enumerate(input)
            ]
        )


class FakeOpenAI:
    """Fake OpenAI client for embedding-pipeline tests."""

    def __init__(self) -> None:
        self.embeddings = FakeEmbeddings()


def create_chunk(position: int) -> DocumentChunk:
    """Create a document chunk for pipeline tests."""

    return DocumentChunk(
        chunk_id=f"openai-evals-guide-chunk-{position:04d}",
        source_id="openai-evals-guide",
        position=position,
        heading_path=["Working with evals"],
        content=f"Evaluation guidance number {position}.",
        token_count=5,
    )


def test_generate_and_store_embeddings_writes_jsonl(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    output_path = tmp_path / "embeddings.jsonl"
    chunks = [
        create_chunk(0),
        create_chunk(1),
    ]
    write_chunks_jsonl(
        chunks=chunks,
        output_path=chunks_path,
    )
    client = FakeOpenAI()

    written_path = generate_and_store_embeddings(
        chunks_path=chunks_path,
        output_path=output_path,
        client=client,
    )

    stored_embeddings = load_embeddings(written_path)

    assert written_path == output_path
    assert len(stored_embeddings) == 2
    assert stored_embeddings[0].chunk_id == chunks[0].chunk_id
    assert stored_embeddings[1].chunk_id == chunks[1].chunk_id
    assert len(client.embeddings.calls) == 1
    assert client.embeddings.calls[0]["input"] == [
        chunks[0].content,
        chunks[1].content,
    ]


def test_generate_and_store_embeddings_prevents_overwrite(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "embeddings.jsonl"
    output_path.write_text(
        "existing embedding data\n",
        encoding="utf-8",
    )
    client = FakeOpenAI()

    with pytest.raises(
        FileExistsError,
        match="Embedding file already exists",
    ):
        generate_and_store_embeddings(
            chunks_path=tmp_path / "chunks.jsonl",
            output_path=output_path,
            client=client,
        )

    assert client.embeddings.calls == []
    assert output_path.read_text(
        encoding="utf-8"
    ) == "existing embedding data\n"


def test_generate_and_store_embeddings_allows_overwrite(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    output_path = tmp_path / "embeddings.jsonl"
    chunks = [create_chunk(0)]

    write_chunks_jsonl(
        chunks=chunks,
        output_path=chunks_path,
    )
    output_path.write_text(
        "old embedding data\n",
        encoding="utf-8",
    )
    client = FakeOpenAI()

    generate_and_store_embeddings(
        chunks_path=chunks_path,
        output_path=output_path,
        client=client,
        overwrite=True,
    )

    stored_embeddings = load_embeddings(output_path)

    assert len(stored_embeddings) == 1
    assert stored_embeddings[0].chunk_id == chunks[0].chunk_id
    assert len(client.embeddings.calls) == 1