from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class DocumentSource(BaseModel):
    """Metadata describing a document used by the RAG system."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    title: str = Field(min_length=3)
    canonical_url: HttpUrl
    markdown_url: HttpUrl
    enabled: bool = True

    @field_validator("canonical_url", "markdown_url")
    @classmethod
    def require_https(cls, value: HttpUrl) -> HttpUrl:
        """Only permit secure document sources."""

        if value.scheme != "https":
            raise ValueError("Document source URLs must use HTTPS")

        return value


class IngestedDocumentMetadata(BaseModel):
    """Lineage metadata for a downloaded document."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    canonical_url: HttpUrl
    local_path: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    content_bytes: int = Field(gt=0)
    retrieved_at: datetime