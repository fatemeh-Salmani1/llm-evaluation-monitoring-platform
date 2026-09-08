from pydantic import BaseModel, ConfigDict, Field


class DocumentChunk(BaseModel):
    """A searchable section extracted from a source document."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*-chunk-\d{4}$"
    )
    source_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    position: int = Field(ge=0)
    heading_path: list[str] = Field(min_length=1)
    content: str = Field(min_length=1)