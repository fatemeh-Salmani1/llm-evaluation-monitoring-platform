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