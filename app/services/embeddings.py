"""Query embedding generation through Pinecone Inference.

The stored vectors are assumed to have been produced by Pinecone Inference
with ``multilingual-e5-large`` and ``input_type="passage"`` (Pinecone adds the
E5 ``passage:`` prefix server-side for that parameter).  Queries therefore use
``input_type="query"`` and no manual prefix; adding one manually would
double-prefix and change the embedding.

No fallback embedding model is attempted.  A dimension mismatch is a hard
error so a mismatched model cannot silently produce meaningless results.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import Settings
from app.errors import EmbeddingError
from app.services.upstream_errors import map_pinecone_exception

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, settings: Settings, inference: Any) -> None:
        self._settings = settings
        self._inference = inference

    @property
    def model(self) -> str:
        return self._settings.pinecone_embed_model

    @property
    def expected_dimension(self) -> int | None:
        return self._settings.embedding_dimension

    def embed_query(self, text: str) -> list[float]:
        """Embed one query using the query-side E5 convention.

        Raises:
            EmbeddingError / Dependency*: on an upstream failure or when the
                returned dimension differs from ``EMBEDDING_DIMENSION``.
        """
        text = text.strip()
        if not text:
            raise EmbeddingError("Cannot embed an empty question.")

        try:
            result = self._inference.embed(
                model=self._settings.pinecone_embed_model,
                inputs=text,
                parameters={"input_type": "query", "truncate": "END"},
            )
        except Exception as exc:
            logger.warning(
                "Pinecone embedding request failed (%s, status=%s)",
                type(exc).__name__,
                getattr(exc, "status_code", None),
            )
            raise map_pinecone_exception(
                exc, base_error=EmbeddingError, action="Query embedding"
            ) from exc

        values = self._extract_values(result)
        if not values:
            raise EmbeddingError("Pinecone returned an embedding without values.")

        try:
            vector = [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise EmbeddingError("Pinecone returned a malformed embedding.") from exc

        expected = self._settings.embedding_dimension
        if expected is not None and len(vector) != expected:
            raise EmbeddingError(
                "Embedding dimension mismatch: "
                f"PINECONE_EMBED_MODEL={self._settings.pinecone_embed_model!r} returned "
                f"{len(vector)} values, but EMBEDDING_DIMENSION={expected}. "
                "Use the same model and dimension as the ingestion pipeline."
            )
        return vector

    @staticmethod
    def _extract_values(result: Any) -> list[float] | None:
        data = getattr(result, "data", None)
        if data is None and isinstance(result, dict):
            data = result.get("data", result)
        if data is None:
            data = result
        if isinstance(data, dict):
            data = data.get("data")
        if not data:
            return None
        first = data[0]
        values = (
            first.get("values")
            if isinstance(first, dict)
            else getattr(first, "values", None)
        )
        if values is None:
            return None
        return list(values)
