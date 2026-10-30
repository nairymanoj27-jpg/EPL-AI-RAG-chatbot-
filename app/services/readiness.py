"""Cached readiness checks for configuration, Pinecone, and Gemini."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.config import Settings
from app.errors import AppError, DependencyError
from app.schemas import ReadinessCheck
from app.services.gemini_service import GeminiService
from app.services.pinecone_service import PineconeService

logger = logging.getLogger(__name__)


class ReadinessService:
    def __init__(
        self,
        settings: Settings,
        pinecone_service: PineconeService,
        gemini_service: GeminiService,
    ) -> None:
        self._settings = settings
        self._pinecone = pinecone_service
        self._gemini = gemini_service
        self._cached_checks: list[ReadinessCheck] | None = None
        self._cached_at = 0.0
        self._lock = asyncio.Lock()

    async def check(self, *, force: bool = False) -> tuple[bool, list[ReadinessCheck]]:
        now = time.monotonic()
        if (
            not force
            and self._cached_checks is not None
            and now - self._cached_at < self._settings.readiness_cache_seconds
        ):
            return self._is_ready(self._cached_checks), list(self._cached_checks)

        async with self._lock:
            now = time.monotonic()
            if (
                not force
                and self._cached_checks is not None
                and now - self._cached_at < self._settings.readiness_cache_seconds
            ):
                return self._is_ready(self._cached_checks), list(self._cached_checks)

            checks: list[ReadinessCheck] = [
                ReadinessCheck(
                    name="configuration",
                    ok=True,
                    detail="Required settings loaded; secrets are present.",
                )
            ]
            pinecone_checks, gemini_check = await asyncio.gather(
                self._check_pinecone(), self._check_gemini()
            )
            checks.extend(pinecone_checks)
            checks.append(gemini_check)
            self._cached_checks = checks
            self._cached_at = time.monotonic()
            return self._is_ready(checks), checks

    @staticmethod
    def _is_ready(checks: list[ReadinessCheck]) -> bool:
        return all(check.ok for check in checks if check.required)

    async def _check_pinecone(self) -> list[ReadinessCheck]:
        try:
            async with asyncio.timeout(self._settings.request_timeout_seconds):
                stats = await asyncio.to_thread(self._pinecone.describe_stats)
        except TimeoutError:
            return [
                ReadinessCheck(
                    name="pinecone",
                    ok=False,
                    detail="Pinecone index stats timed out.",
                )
            ]
        except AppError as exc:
            return [
                ReadinessCheck(name="pinecone", ok=False, detail=exc.message)
            ]
        except Exception as exc:
            logger.warning("Readiness Pinecone check raised %s", type(exc).__name__)
            return [
                ReadinessCheck(
                    name="pinecone",
                    ok=False,
                    detail="Pinecone readiness check failed.",
                )
            ]

        dimension = getattr(stats, "dimension", None)
        metric = getattr(stats, "metric", None)
        total_vectors = getattr(stats, "total_vector_count", None)
        namespaces = getattr(stats, "namespaces", None) or {}

        checks: list[ReadinessCheck] = []
        if dimension is None:
            checks.append(
                ReadinessCheck(
                    name="pinecone_dimension",
                    ok=False,
                    detail="Pinecone did not report a dense-vector dimension for this index.",
                )
            )
        elif self._settings.embedding_dimension is not None and dimension != self._settings.embedding_dimension:
            checks.append(
                ReadinessCheck(
                    name="pinecone_dimension",
                    ok=False,
                    detail=(
                        f"Index dimension {dimension} does not match EMBEDDING_DIMENSION="
                        f"{self._settings.embedding_dimension}. Retrieval would fail."
                    ),
                )
            )
        else:
            checks.append(
                ReadinessCheck(
                    name="pinecone_dimension",
                    ok=True,
                    detail=f"Index dimension is {dimension}.",
                )
            )

        expected_metric = (self._settings.expected_metric or "").lower()
        actual_metric = str(metric or "").lower()
        if expected_metric and actual_metric and expected_metric != actual_metric:
            checks.append(
                ReadinessCheck(
                    name="pinecone_metric",
                    ok=False,
                    required=False,
                    detail=(
                        f"Index metric is {actual_metric!r}, configured EXPECTED_METRIC is "
                        f"{expected_metric!r}. Retrieval can work, but relevance thresholds and "
                        "scores differ; verify this is intentional."
                    ),
                )
            )
        else:
            checks.append(
                ReadinessCheck(
                    name="pinecone_metric",
                    ok=True,
                    detail=f"Index metric is {actual_metric or 'unknown'}.",
                )
            )

        namespace = self._settings.namespace
        count = self._namespace_count(namespaces, namespace)
        if total_vectors == 0:
            checks.append(
                ReadinessCheck(
                    name="pinecone_namespace",
                    ok=False,
                    detail="The Pinecone index contains no vectors.",
                )
            )
        elif count is None:
            checks.append(
                ReadinessCheck(
                    name="pinecone_namespace",
                    ok=False,
                    detail=(
                        f"Namespace {namespace!r} was not found. Available namespaces: "
                        f"{self._namespace_names(namespaces)}. Set PINECONE_NAMESPACE."
                    ),
                )
            )
        elif count == 0:
            checks.append(
                ReadinessCheck(
                    name="pinecone_namespace",
                    ok=False,
                    detail=f"Namespace {namespace!r} contains no vectors.",
                )
            )
        else:
            checks.append(
                ReadinessCheck(
                    name="pinecone_namespace",
                    ok=True,
                    detail=f"Namespace {namespace!r} contains {count} vector(s).",
                )
            )
        return checks

    async def _check_gemini(self) -> ReadinessCheck:
        try:
            await self._gemini.probe()
            return ReadinessCheck(
                name="gemini",
                ok=True,
                detail=f"Gemini model {self._settings.gemini_model!r} is accessible.",
            )
        except (AppError, DependencyError) as exc:
            return ReadinessCheck(name="gemini", ok=False, detail=exc.message)
        except Exception as exc:
            logger.warning("Readiness Gemini check raised %s", type(exc).__name__)
            return ReadinessCheck(
                name="gemini", ok=False, detail="Gemini readiness check failed."
            )

    @staticmethod
    def _namespace_count(namespaces: Any, target: str) -> int | None:
        if target == "":
            for key in ("", "__default__"):
                if key in namespaces:
                    return ReadinessService._summary_count(namespaces[key])
        if target in namespaces:
            return ReadinessService._summary_count(namespaces[target])
        return None

    @staticmethod
    def _summary_count(summary: Any) -> int:
        if isinstance(summary, dict):
            return int(summary.get("vector_count", 0))
        return int(getattr(summary, "vector_count", 0))

    @staticmethod
    def _namespace_names(namespaces: Any) -> str:
        names = [repr(name) for name in namespaces]
        return ", ".join(names) if names else "(none)"
