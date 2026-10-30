"""Read-only Pinecone data-plane access.

This wrapper exposes only ``query`` and ``describe_index_stats``.  It contains
no upsert, update, delete, or index-management calls, so application code
cannot modify the existing index.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from pinecone import Pinecone

from app.config import Settings
from app.errors import DependencyError, RetrievalError
from app.services.upstream_errors import map_pinecone_exception

logger = logging.getLogger(__name__)


@dataclass
class VectorMatch:
    id: str
    score: float
    metadata: dict[str, Any]


class PineconeService:
    """Lazy connection to one existing Pinecone index (read-only)."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client or Pinecone(
            api_key=settings.pinecone_api_key.get_secret_value(),
            timeout=settings.request_timeout_seconds,
        )
        self._index: Any | None = None
        self._index_lock = threading.Lock()

    @property
    def inference(self) -> Any:
        return self._client.inference

    @property
    def index_name(self) -> str:
        return self._settings.pinecone_index_name

    @property
    def namespace(self) -> str:
        return self._settings.namespace

    def _get_index(self) -> Any:
        # Resolving an index by name requires a control-plane call; resolve it
        # lazily so application startup does not depend on network access.
        if self._index is None:
            with self._index_lock:
                if self._index is None:
                    if self._settings.pinecone_host:
                        self._index = self._client.Index(host=self._settings.pinecone_host)
                    else:
                        self._index = self._client.Index(name=self._settings.pinecone_index_name)
        return self._index

    def query(
        self,
        vector: list[float],
        *,
        top_k: int,
        namespace: str | None = None,
    ) -> list[VectorMatch]:
        target_namespace = self._settings.namespace if namespace is None else namespace
        try:
            response = self._get_index().query(
                vector=vector,
                top_k=top_k,
                namespace=target_namespace,
                include_metadata=True,
                include_values=False,
            )
        except Exception as exc:
            logger.warning(
                "Pinecone query failed (%s, status=%s)",
                type(exc).__name__,
                getattr(exc, "status_code", None),
            )
            raise map_pinecone_exception(
                exc, base_error=RetrievalError, action="Pinecone query"
            ) from exc

        matches = getattr(response, "matches", None)
        if matches is None and isinstance(response, dict):
            matches = response.get("matches", [])
        results: list[VectorMatch] = []
        for match in matches or []:
            if isinstance(match, dict):
                match_id = str(match.get("id", ""))
                score = match.get("score", 0.0)
                metadata = match.get("metadata") or {}
            else:
                match_id = str(getattr(match, "id", ""))
                score = getattr(match, "score", 0.0)
                metadata = getattr(match, "metadata", None) or {}
            results.append(
                VectorMatch(
                    id=match_id,
                    score=float(score),
                    metadata=dict(metadata),
                )
            )
        return results

    def describe_stats(self) -> Any:
        try:
            return self._get_index().describe_index_stats()
        except Exception as exc:
            logger.warning(
                "Pinecone describe_index_stats failed (%s, status=%s)",
                type(exc).__name__,
                getattr(exc, "status_code", None),
            )
            raise map_pinecone_exception(
                exc, base_error=RetrievalError, action="Pinecone index inspection"
            ) from exc

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.debug("Error while closing Pinecone client", exc_info=True)


__all__ = ["DependencyError", "PineconeService", "VectorMatch"]
