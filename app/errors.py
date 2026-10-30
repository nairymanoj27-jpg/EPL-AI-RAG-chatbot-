"""Application error types with safe, client-facing messages."""

from __future__ import annotations


class AppError(Exception):
    """Base class for errors that can be translated into an HTTP response."""

    status_code: int = 500
    code: str = "internal_error"
    default_message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        retry_after: int | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.retry_after = retry_after
        super().__init__(self.message)


class RequestValidationAppError(AppError):
    status_code = 400
    code = "invalid_request"
    default_message = "The request is invalid."


class ConversationNotFoundError(AppError):
    status_code = 404
    code = "conversation_not_found"
    default_message = "Conversation not found."


class ConversationMemoryDisabledError(AppError):
    status_code = 409
    code = "conversation_memory_disabled"
    default_message = (
        "Conversation memory is disabled on this server (CONVERSATION_STORAGE=none)."
    )


class FeatureDisabledError(AppError):
    status_code = 404
    code = "feature_disabled"
    default_message = "This endpoint is disabled on this server."


class DependencyError(AppError):
    """An upstream dependency (Pinecone or Gemini) failed."""

    status_code = 502
    code = "upstream_error"
    default_message = "An upstream service failed."


class DependencyAuthError(DependencyError):
    status_code = 503
    code = "upstream_auth_failed"
    default_message = "An upstream service rejected the configured credentials."


class DependencyRateLimitError(DependencyError):
    status_code = 429
    code = "upstream_rate_limited"
    default_message = "An upstream service rate limit was reached. Retry shortly."

    def __init__(self, message: str | None = None, *, retry_after: int = 5) -> None:
        super().__init__(message, retry_after=retry_after)


class DependencyTimeoutError(DependencyError):
    status_code = 504
    code = "upstream_timeout"
    default_message = "An upstream service timed out."


class DependencyUnavailableError(DependencyError):
    status_code = 502
    code = "upstream_unavailable"
    default_message = "An upstream service is unavailable."


class EmptyGenerationError(DependencyError):
    status_code = 502
    code = "empty_generation"
    default_message = "Gemini returned an empty response."


class EmbeddingError(DependencyError):
    default_message = "Query embedding failed."


class RetrievalError(DependencyError):
    default_message = "Pinecone retrieval failed."


class GenerationError(DependencyError):
    default_message = "Gemini answer generation failed."
