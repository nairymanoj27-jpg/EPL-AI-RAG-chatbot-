"""Translate Pinecone SDK exceptions into safe application errors."""

from __future__ import annotations

from pinecone import errors as pc_errors

from app.errors import (
    DependencyAuthError,
    DependencyError,
    DependencyRateLimitError,
    DependencyTimeoutError,
    DependencyUnavailableError,
)


def map_pinecone_exception(
    exc: Exception,
    *,
    base_error: type[DependencyError],
    action: str,
) -> DependencyError:
    """Return an application error without exposing upstream internals.

    The optional *base_error* keeps the caller's domain (embedding vs.
    retrieval) while still preserving the more precise HTTP semantics of
    authentication, rate-limit and timeout failures.
    """
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()

    if isinstance(exc, (DependencyError,)):
        return exc
    if isinstance(exc, (pc_errors.UnauthorizedError, pc_errors.ForbiddenError)) or status in (401, 403):
        return DependencyAuthError(
            "Pinecone authentication failed. Check PINECONE_API_KEY and that it can access this project."
        )
    if isinstance(exc, pc_errors.RateLimitError) or status == 429:
        if "embedding token limit" in text or "resource_exhausted" in text:
            return DependencyRateLimitError(
                "Pinecone's embedding token quota for the configured model is exhausted. "
                "Check Pinecone usage and billing; retrieval resumes after the quota resets "
                "or the plan is upgraded.",
                retry_after=3600,
            )
        return DependencyRateLimitError("Pinecone rate limit reached. Retry shortly.")
    if isinstance(exc, (pc_errors.PineconeTimeoutError, TimeoutError)) or status in (408, 504):
        return DependencyTimeoutError("Pinecone timed out while handling the request.")
    if isinstance(exc, pc_errors.NotFoundError) or status == 404:
        return DependencyUnavailableError(
            "Pinecone could not find the configured index or namespace. "
            "Check PINECONE_INDEX_NAME, PINECONE_HOST and PINECONE_NAMESPACE."
        )
    if status == 400 and "dimension" in text:
        return DependencyUnavailableError(
            "Pinecone rejected the query vector dimension. Verify EMBEDDING_DIMENSION and that "
            "PINECONE_EMBED_MODEL matches the model used during ingestion."
        )
    if isinstance(exc, pc_errors.PineconeConnectionError):
        return DependencyUnavailableError("Could not connect to Pinecone. Check network access and PINECONE_HOST.")
    return base_error(f"{action} failed. Check the server logs for the upstream error type.")
