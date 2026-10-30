"""Retrieval pipeline tests with fake embedding and Pinecone clients."""

from __future__ import annotations

from typing import Any

import pytest

from app.config import Settings
from app.errors import DependencyError, EmbeddingError, RequestValidationAppError
from app.services.pinecone_service import VectorMatch
from app.services.retrieval import RetrievalService
from tests.conftest import make_settings


class FakeEmbeddings:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [0.1, 0.2, 0.3]
        self.calls: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return self.vector


class FakePinecone:
    def __init__(self, matches: list[VectorMatch] | None = None) -> None:
        self.matches = matches or []
        self.calls: list[dict[str, Any]] = []

    def query(self, vector: list[float], *, top_k: int, namespace: str | None = None):
        self.calls.append({"vector": vector, "top_k": top_k, "namespace": namespace})
        return self.matches


def make_match(
    vector_id: str,
    score: float,
    metadata: dict[str, Any] | None = None,
) -> VectorMatch:
    return VectorMatch(
        id=vector_id,
        score=score,
        metadata=metadata if metadata is not None else {"text": f"text for {vector_id}"},
    )


def make_service(
    settings: Settings,
    matches: list[VectorMatch] | None = None,
) -> tuple[RetrievalService, FakeEmbeddings, FakePinecone]:
    embeddings = FakeEmbeddings()
    pinecone = FakePinecone(matches)
    return RetrievalService(settings, embeddings, pinecone), embeddings, pinecone


@pytest.mark.asyncio
async def test_retrieve_embeds_question_and_queries_configured_namespace(
    tmp_path,
) -> None:
    settings = make_settings(tmp_path, pinecone_namespace="epl-2024")
    service, embeddings, pinecone = make_service(
        settings, [make_match("a", 0.8), make_match("b", 0.7)]
    )

    result = await service.retrieve("Who won?", top_k=2)

    assert embeddings.calls == ["Who won?"]
    assert pinecone.calls[0]["top_k"] == 2
    assert pinecone.calls[0]["namespace"] == "epl-2024"
    assert [chunk.citation_id for chunk in result.chunks] == ["S1", "S2"]
    assert [chunk.rank for chunk in result.chunks] == [1, 2]
    assert result.matches_received == 2


@pytest.mark.asyncio
async def test_score_threshold_filters_and_renumbers_citations(tmp_path) -> None:
    settings = make_settings(tmp_path, min_relevance_score=0.75)
    service, _, _ = make_service(
        settings,
        [make_match("a", 0.6), make_match("b", 0.9), make_match("c", 0.8)],
    )

    result = await service.retrieve("question")

    assert [chunk.id for chunk in result.chunks] == ["b", "c"]
    assert [chunk.citation_id for chunk in result.chunks] == ["S1", "S2"]


@pytest.mark.asyncio
async def test_threshold_respects_euclidean_metric_warning(tmp_path) -> None:
    settings = make_settings(
        tmp_path, min_relevance_score=0.75, expected_metric="euclidean"
    )
    service, _, _ = make_service(settings, [make_match("a", 0.2)])

    result = await service.retrieve("question")

    assert len(result.chunks) == 1
    assert any("euclidean" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_missing_configured_text_field_uses_fallback(tmp_path) -> None:
    settings = make_settings(tmp_path, pinecone_text_field="text")
    service, _, _ = make_service(
        settings, [make_match("a", 0.9, {"chunk_text": "fallback text"})]
    )

    result = await service.retrieve("question")

    assert result.chunks[0].text == "fallback text"
    assert any("fallback" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_matches_without_text_are_skipped_with_warning(tmp_path) -> None:
    settings = make_settings(tmp_path)
    service, _, _ = make_service(
        settings,
        [make_match("a", 0.9, {"team": "Arsenal"}), make_match("b", 0.8)],
    )

    result = await service.retrieve("question")

    assert [chunk.id for chunk in result.chunks] == ["b"]
    assert result.skipped_missing_text == 1
    assert any("without usable text" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_no_matches_returns_warning_and_empty_result(tmp_path) -> None:
    settings = make_settings(tmp_path)
    service, _, _ = make_service(settings, [])

    result = await service.retrieve("question")

    assert result.empty
    assert any("No vectors matched" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_all_below_threshold_returns_helpful_warning(tmp_path) -> None:
    settings = make_settings(tmp_path, min_relevance_score=0.9)
    service, _, _ = make_service(settings, [make_match("a", 0.1)])

    result = await service.retrieve("question")

    assert result.empty
    assert any("MIN_RELEVANCE_SCORE" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_invalid_top_k_is_rejected_before_network_calls(tmp_path) -> None:
    settings = make_settings(tmp_path, max_top_k=5)
    service, embeddings, pinecone = make_service(settings)

    with pytest.raises(RequestValidationAppError):
        await service.retrieve("question", top_k=6)

    assert embeddings.calls == []
    assert pinecone.calls == []


@pytest.mark.asyncio
async def test_embedding_errors_are_not_swallowed(tmp_path) -> None:
    settings = make_settings(tmp_path)
    embeddings = FakeEmbeddings()

    def fail(_text: str) -> list[float]:
        raise EmbeddingError("bad embedding")

    embeddings.embed_query = fail  # type: ignore[method-assign]
    pinecone = FakePinecone([make_match("a", 0.5)])
    service = RetrievalService(settings, embeddings, pinecone)

    with pytest.raises(DependencyError):
        await service.retrieve("question")

    assert pinecone.calls == []


@pytest.mark.asyncio
async def test_metadata_is_preserved_for_citation_display(tmp_path) -> None:
    settings = make_settings(tmp_path)
    service, _, _ = make_service(
        settings,
        [
            make_match(
                "a",
                0.9,
                {"text": "Arsenal won.", "season": "2003-04", "team": "Arsenal"},
            )
        ],
    )

    result = await service.retrieve("question")

    assert result.chunks[0].metadata["season"] == "2003-04"
