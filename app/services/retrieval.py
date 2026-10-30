"""Retrieval: embed the question, query Pinecone, and normalize chunks."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.errors import DependencyError, DependencyTimeoutError, RequestValidationAppError
from app.services.embeddings import EmbeddingService
from app.services.pinecone_service import PineconeService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    citation_id: str
    rank: int
    score: float
    text: str
    metadata: dict[str, Any]


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    matches_received: int = 0
    skipped_missing_text: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.chunks


class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        pinecone_service: PineconeService,
    ) -> None:
        self._settings = settings
        self._embeddings = embedding_service
        self._pinecone = pinecone_service

    async def retrieve(self, question: str, *, top_k: int | None = None) -> RetrievalResult:
        requested_k = self._settings.top_k if top_k is None else top_k
        if not 1 <= requested_k <= self._settings.max_top_k:
            raise RequestValidationAppError(
                f"top_k must be between 1 and {self._settings.max_top_k}."
            )

        vector = await self._call_with_timeout(
            self._embeddings.embed_query, question, action="Query embedding"
        )
        matches = await self._call_with_timeout(
            self._pinecone.query,
            vector,
            top_k=requested_k,
            namespace=self._settings.namespace,
            action="Pinecone retrieval",
        )

        result = RetrievalResult(matches_received=len(matches))
        if not matches:
            result.warnings.append("No vectors matched this question in the configured namespace.")
            return result

        warnings: list[str] = []
        threshold = self._settings.min_relevance_score
        if threshold is not None and (self._settings.expected_metric or "").lower() == "euclidean":
            warnings.append(
                "MIN_RELEVANCE_SCORE is ignored because the configured metric is euclidean "
                "(lower scores are closer)."
            )
            threshold = None

        candidates: list[tuple[Any, str, str]] = []
        fallbacks_used: set[str] = set()
        for match in matches:
            if threshold is not None and match.score < threshold:
                continue
            text, field_used = self._extract_text(match.metadata)
            if text is None:
                result.skipped_missing_text += 1
                continue
            if field_used != self._settings.pinecone_text_field:
                fallbacks_used.add(field_used)
            candidates.append((match, text, field_used))

        if fallbacks_used:
            names = ", ".join(sorted(fallbacks_used))
            warnings.append(
                "The configured PINECONE_TEXT_FIELD was missing; used fallback metadata "
                f"field(s): {names}. Set PINECONE_TEXT_FIELD to the field that stores chunk text."
            )

        if result.skipped_missing_text:
            warnings.append(
                f"Skipped {result.skipped_missing_text} match(es) without usable text in "
                f"PINECONE_TEXT_FIELD='{self._settings.pinecone_text_field}' or its fallbacks."
            )

        if not candidates:
            if result.skipped_missing_text:
                warnings.append(
                    "Retrieved vectors were discarded because none contained chunk text. "
                    "Inspect the metadata schema and set PINECONE_TEXT_FIELD."
                )
            elif threshold is not None:
                warnings.append(
                    f"No matches met MIN_RELEVANCE_SCORE={threshold}. Lower the threshold only "
                    "after calibrating it against representative questions."
                )
            result.warnings.extend(dict.fromkeys(warnings))
            return result

        for rank, (match, text, _field_used) in enumerate(candidates, start=1):
            result.chunks.append(
                RetrievedChunk(
                    id=match.id,
                    citation_id=f"S{rank}",
                    rank=rank,
                    score=match.score,
                    text=text,
                    metadata=dict(match.metadata or {}),
                )
            )

        result.warnings.extend(dict.fromkeys(warnings))
        return result

    def _extract_text(self, metadata: dict[str, Any] | None) -> tuple[str | None, str]:
        if not metadata:
            return None, self._settings.pinecone_text_field
        fields = [self._settings.pinecone_text_field, *self._settings.text_field_fallbacks]
        for field_name in fields:
            value = metadata.get(field_name)
            if isinstance(value, str) and value.strip():
                return value.strip(), field_name
        return None, self._settings.pinecone_text_field

    async def _call_with_timeout(self, func, *args, action: str, **kwargs):
        try:
            async with asyncio.timeout(self._settings.request_timeout_seconds):
                return await asyncio.to_thread(func, *args, **kwargs)
        except TimeoutError as exc:
            raise DependencyTimeoutError(f"{action} timed out.") from exc
        except DependencyError:
            raise
        except Exception as exc:
            logger.exception("%s raised an unexpected error", action)
            raise DependencyError(f"{action} failed.") from exc
