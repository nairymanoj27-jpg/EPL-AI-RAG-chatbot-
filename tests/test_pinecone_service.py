"""Read-only Pinecone wrapper tests (no network)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.errors import (
    DependencyAuthError,
    DependencyRateLimitError,
    RetrievalError,
)
from app.services.pinecone_service import PineconeService
from tests.conftest import make_settings


class FakeIndex:
    def __init__(self, *, matches=None, stats=None, error: Exception | None = None) -> None:
        self.matches = matches or []
        self.stats = stats
        self.error = error
        self.query_calls: list[dict] = []
        self.stats_calls = 0

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(matches=self.matches)

    def describe_index_stats(self):
        self.stats_calls += 1
        if self.error:
            raise self.error
        return self.stats


class FakePinecone:
    def __init__(self, index: FakeIndex) -> None:
        self._index = index
        self.inference = object()
        self.index_calls: list[dict] = []
        self.closed = False

    def Index(self, **kwargs):
        self.index_calls.append(kwargs)
        return self._index

    def close(self) -> None:
        self.closed = True


class FakeUpstreamError(Exception):
    def __init__(
        self, status_code: int, message: str = "upstream detail that must not leak"
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


def make_service(tmp_path, index: FakeIndex) -> tuple[PineconeService, FakePinecone]:
    client = FakePinecone(index)
    settings = make_settings(tmp_path, pinecone_namespace="epl", top_k=7)
    return PineconeService(settings, client=client), client


def test_query_uses_read_only_data_plane_options(tmp_path) -> None:
    index = FakeIndex(
        matches=[
            SimpleNamespace(
                id="vec-1", score=0.87, metadata={"text": "Arsenal won."}
            ),
            {"id": "vec-2", "score": 0.5, "metadata": {"text": "Chelsea lost."}},
        ]
    )
    service, client = make_service(tmp_path, index)

    matches = service.query([0.1, 0.2], top_k=7)

    assert [match.id for match in matches] == ["vec-1", "vec-2"]
    assert matches[0].score == 0.87
    assert matches[1].metadata["text"] == "Chelsea lost."
    assert index.query_calls == [
        {
            "vector": [0.1, 0.2],
            "top_k": 7,
            "namespace": "epl",
            "include_metadata": True,
            "include_values": False,
        }
    ]
    assert client.index_calls == [{"name": "test-index"}]


def test_index_is_resolved_lazily_once(tmp_path) -> None:
    index = FakeIndex(matches=[])
    service, client = make_service(tmp_path, index)

    assert client.index_calls == []
    service.query([0.1], top_k=1)
    service.query([0.1], top_k=1)

    assert client.index_calls == [{"name": "test-index"}]


def test_host_is_used_instead_of_index_name(tmp_path) -> None:
    index = FakeIndex(matches=[])
    client = FakePinecone(index)
    settings = make_settings(
        tmp_path, pinecone_host="https://my-index.svc.pinecone.io", top_k=1
    )
    service = PineconeService(settings, client=client)

    service.query([0.1], top_k=1)

    assert client.index_calls == [{"host": "https://my-index.svc.pinecone.io"}]


def test_query_namespace_can_be_overridden_for_tooling(tmp_path) -> None:
    index = FakeIndex(matches=[])
    service, _ = make_service(tmp_path, index)

    service.query([0.1], top_k=1, namespace="other")

    assert index.query_calls[0]["namespace"] == "other"


def test_auth_error_is_mapped_and_does_not_leak_message(tmp_path) -> None:
    service, _ = make_service(tmp_path, FakeIndex(error=FakeUpstreamError(401)))

    with pytest.raises(DependencyAuthError) as excinfo:
        service.query([0.1], top_k=1)

    assert "upstream detail" not in str(excinfo.value)


def test_rate_limit_error_is_mapped(tmp_path) -> None:
    service, _ = make_service(tmp_path, FakeIndex(error=FakeUpstreamError(429)))

    with pytest.raises(DependencyRateLimitError):
        service.query([0.1], top_k=1)


def test_monthly_embedding_quota_error_is_actionable(tmp_path) -> None:
    error = FakeUpstreamError(
        429,
        "You've reached the embedding token limit (5000000) for model "
        "multilingual-e5-large. RESOURCE_EXHAUSTED",
    )
    service, _ = make_service(tmp_path, FakeIndex(error=error))

    with pytest.raises(DependencyRateLimitError) as excinfo:
        service.query([0.1], top_k=1)

    message = str(excinfo.value)
    assert "quota" in message.lower()
    assert "5000000" not in message
    assert "upstream detail" not in message
    assert excinfo.value.retry_after == 3600


def test_unknown_error_becomes_retrieval_error(tmp_path) -> None:
    service, _ = make_service(tmp_path, FakeIndex(error=RuntimeError("boom")))

    with pytest.raises(RetrievalError):
        service.describe_stats()


def test_describe_stats_returns_sdk_object(tmp_path) -> None:
    stats = SimpleNamespace(dimension=1024, metric="cosine")
    service, _ = make_service(tmp_path, FakeIndex(stats=stats))

    assert service.describe_stats() is stats


def test_wrapper_exposes_no_mutation_methods() -> None:
    for name in (
        "upsert",
        "delete",
        "update",
        "create_index",
        "delete_index",
        "configure_index",
    ):
        assert not hasattr(PineconeService, name)


def test_close_closes_client(tmp_path) -> None:
    service, client = make_service(tmp_path, FakeIndex())

    service.close()

    assert client.closed is True
