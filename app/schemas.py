"""Pydantic request and response models.

Requests use ``extra="forbid"`` so a client cannot smuggle in fields such as
``system_prompt`` or ``namespace`` and override server-side behavior.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CONVERSATION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{7,63}$"


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2_000)
    conversation_id: str | None = Field(
        default=None,
        pattern=CONVERSATION_ID_PATTERN,
        description="Optional conversation identifier; a new one is generated when omitted.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Optional number of chunks to retrieve (bounded by server configuration).",
    )

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value

    @field_validator("conversation_id")
    @classmethod
    def _conversation_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        return value


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2_000)
    top_k: int | None = Field(default=None, ge=1, le=100)

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class SourceChunk(BaseModel):
    """A retrieved Pinecone record exposed to the client."""

    id: str
    citation_id: str = Field(description="Stable identifier used in the answer, e.g. S1.")
    rank: int = Field(ge=1)
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    sources: list[SourceChunk] = Field(default_factory=list)
    cited_source_ids: list[str] = Field(default_factory=list)
    evidence_found: bool
    model: str
    warnings: list[str] = Field(default_factory=list)


class RetrieveResponse(BaseModel):
    question: str
    namespace: str
    count: int
    matches: list[SourceChunk] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ClearConversationResponse(BaseModel):
    conversation_id: str
    cleared: bool = Field(
        description="True when stored history existed and was deleted; false when there was nothing to clear."
    )


class AppInfoResponse(BaseModel):
    name: str
    version: str
    environment: str
    status: str
    docs_url: str
    ui_url: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    name: str
    version: str
    environment: str


class ReadinessCheck(BaseModel):
    name: str
    ok: bool
    required: bool = True
    detail: str


class ReadyResponse(BaseModel):
    ready: bool
    version: str
    checks: list[ReadinessCheck] = Field(default_factory=list)


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ValidationErrorDetail(BaseModel):
    location: str
    message: str
    type: str


class ValidationErrorResponse(BaseModel):
    error: ErrorDetail
    details: list[ValidationErrorDetail] = Field(default_factory=list)
