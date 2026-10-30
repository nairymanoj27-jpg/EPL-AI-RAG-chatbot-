"""Readiness checks with fake upstream services."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.errors import DependencyUnavailableError
from app.services.readiness import ReadinessService
from tests.conftest import make_settings


class FakePinecone:
    def __init__(self, stats=None, error: Exception | None = None) -> None:
        self.stats = stats
        self.error = error
        self.calls = 0

    def describe_stats(self):
        self.calls += 1
        if self.error:
            raise self.error
        return self.stats


class FakeGemini:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    async def probe(self) -> None:
        self.calls += 1
        if self.error:
            raise self.error


def stats(
    dimension: int = 1024,
    metric: str = "cosine",
    total: int = 10,
    namespaces: dict | None = None,
):
    return SimpleNamespace(
        dimension=dimension,
        metric=metric,
        total_vector_count=total,
        namespaces=namespaces if namespaces is not None else {"epl": {"vector_count": 10}},
    )


@pytest.mark.asyncio
async def test_ready_when_dimension_metric_and_namespace_match(tmp_path) -> None:
    settings = make_settings(tmp_path, pinecone_namespace="epl")
    service = ReadinessService(settings, FakePinecone(stats()), FakeGemini())

    ready, checks = await service.check()

    assert ready is True
    by_name = {check.name: check for check in checks}
    assert by_name["pinecone_dimension"].ok is True
    assert by_name["pinecone_namespace"].ok is True
    assert by_name["pinecone_metric"].ok is True
    assert by_name["gemini"].ok is True


@pytest.mark.asyncio
async def test_dimension_mismatch_is_not_ready(tmp_path) -> None:
    settings = make_settings(tmp_path, embedding_dimension=1536, pinecone_namespace="epl")
    service = ReadinessService(settings, FakePinecone(stats()), FakeGemini())

    ready, checks = await service.check()

    assert ready is False
    assert any("does not match" in check.detail for check in checks if "dimension" in check.name)


@pytest.mark.asyncio
async def test_missing_namespace_lists_available_namespaces(tmp_path) -> None:
    settings = make_settings(tmp_path, pinecone_namespace="missing")
    service = ReadinessService(
        settings,
        FakePinecone(stats(namespaces={"epl-2024": {"vector_count": 5}})),
        FakeGemini(),
    )

    ready, checks = await service.check()

    assert ready is False
    detail = next(check.detail for check in checks if check.name == "pinecone_namespace")
    assert "epl-2024" in detail


@pytest.mark.asyncio
async def test_default_namespace_accepts_sdk_empty_key(tmp_path) -> None:
    settings = make_settings(tmp_path, pinecone_namespace="")
    service = ReadinessService(
        settings,
        FakePinecone(stats(namespaces={"": {"vector_count": 3}})),
        FakeGemini(),
    )

    ready, checks = await service.check()

    assert ready is True
    assert next(c for c in checks if c.name == "pinecone_namespace").ok is True


@pytest.mark.asyncio
async def test_metric_mismatch_is_warning_only(tmp_path) -> None:
    settings = make_settings(tmp_path, expected_metric="cosine", pinecone_namespace="epl")
    service = ReadinessService(settings, FakePinecone(stats(metric="dotproduct")), FakeGemini())

    ready, checks = await service.check()

    metric_check = next(check for check in checks if check.name == "pinecone_metric")
    assert metric_check.ok is False
    assert metric_check.required is False
    assert ready is True


@pytest.mark.asyncio
async def test_pinecone_failure_is_reported_without_details(tmp_path) -> None:
    settings = make_settings(tmp_path, pinecone_namespace="epl")
    service = ReadinessService(
        settings,
        FakePinecone(error=DependencyUnavailableError("Cannot reach Pinecone.")),
        FakeGemini(),
    )

    ready, checks = await service.check()

    assert ready is False
    assert next(check for check in checks if check.name == "pinecone").detail == "Cannot reach Pinecone."


@pytest.mark.asyncio
async def test_gemini_failure_is_reported(tmp_path) -> None:
    settings = make_settings(tmp_path, pinecone_namespace="epl")
    service = ReadinessService(
        settings,
        FakePinecone(stats()),
        FakeGemini(error=DependencyUnavailableError("Gemini unavailable.")),
    )

    ready, checks = await service.check()

    assert ready is False
    assert next(check for check in checks if check.name == "gemini").ok is False


@pytest.mark.asyncio
async def test_results_are_cached(tmp_path) -> None:
    settings = make_settings(tmp_path, pinecone_namespace="epl", readiness_cache_seconds=60)
    pinecone = FakePinecone(stats())
    gemini = FakeGemini()
    service = ReadinessService(settings, pinecone, gemini)

    await service.check()
    await service.check()

    assert pinecone.calls == 1
    assert gemini.calls == 1

    await service.check(force=True)
    assert pinecone.calls == 2


@pytest.mark.asyncio
async def test_concurrent_checks_are_serialised(tmp_path) -> None:
    settings = make_settings(tmp_path, pinecone_namespace="epl", readiness_cache_seconds=60)
    pinecone = FakePinecone(stats())
    service = ReadinessService(settings, pinecone, FakeGemini())

    await asyncio.gather(*(service.check() for _ in range(5)))

    assert pinecone.calls == 1
