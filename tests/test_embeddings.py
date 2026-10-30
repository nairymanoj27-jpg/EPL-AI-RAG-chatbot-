"""Embedding wrapper tests: E5 query convention and dimension validation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.errors import EmbeddingError
from app.services.embeddings import EmbeddingService
from tests.conftest import make_settings


@dataclass
class FakeEmbedding:
    values: list[float]


@dataclass
class FakeEmbeddingResult:
    data: list[FakeEmbedding]


class FakeInference:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    def embed(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result


def test_query_embedding_uses_query_input_type_without_manual_prefix(tmp_path) -> None:
    settings = make_settings(tmp_path, embedding_dimension=3)
    inference = FakeInference(FakeEmbeddingResult([FakeEmbedding([1.0, 2.0, 3.0])]))
    service = EmbeddingService(settings, inference)

    vector = service.embed_query("  Who won the league?  ")

    assert vector == [1.0, 2.0, 3.0]
    call = inference.calls[0]
    assert call["model"] == "multilingual-e5-large"
    assert call["inputs"] == "Who won the league?"
    assert call["parameters"]["input_type"] == "query"
    assert "query:" not in call["inputs"]


def test_embedding_dimension_mismatch_is_a_hard_error(tmp_path) -> None:
    settings = make_settings(tmp_path, embedding_dimension=1024)
    inference = FakeInference(FakeEmbeddingResult([FakeEmbedding([0.1, 0.2])]))
    service = EmbeddingService(settings, inference)

    with pytest.raises(EmbeddingError) as excinfo:
        service.embed_query("question")

    assert "dimension mismatch" in str(excinfo.value).lower()


def test_empty_embedding_is_rejected(tmp_path) -> None:
    settings = make_settings(tmp_path, embedding_dimension=None)
    service = EmbeddingService(settings, FakeInference(FakeEmbeddingResult([])))

    with pytest.raises(EmbeddingError):
        service.embed_query("question")


def test_empty_question_is_rejected_without_call(tmp_path) -> None:
    settings = make_settings(tmp_path)
    inference = FakeInference(FakeEmbeddingResult([FakeEmbedding([0.1])]))
    service = EmbeddingService(settings, inference)

    with pytest.raises(EmbeddingError):
        service.embed_query("   ")

    assert inference.calls == []


def test_embedding_response_can_be_a_plain_dict(tmp_path) -> None:
    settings = make_settings(tmp_path, embedding_dimension=2)
    service = EmbeddingService(
        settings,
        FakeInference({"data": [{"values": [0.25, 0.75]}]}),
    )

    assert service.embed_query("question") == [0.25, 0.75]
