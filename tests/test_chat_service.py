"""Chat orchestration tests: grounding, citations, memory, and follow-ups."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.errors import RequestValidationAppError
from app.services.chat_service import ChatService
from app.services.conversation_store import SQLiteConversationStore
from app.services.prompt_builder import NO_EVIDENCE_ANSWER
from app.services.retrieval import RetrievalResult, RetrievedChunk
from tests.conftest import make_settings


def chunk(citation: str = "S1", text: str = "Arsenal finished with 89 points.") -> RetrievedChunk:
    return RetrievedChunk(
        id=f"vec-{citation}",
        citation_id=citation,
        rank=int(citation[1:]),
        score=0.9,
        text=text,
        metadata={"text": text},
    )


class FakeRetrieval:
    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.chunks = [chunk()] if chunks is None else chunks
        self.queries: list[tuple[str, int | None]] = []

    async def retrieve(self, question: str, *, top_k: int | None = None) -> RetrievalResult:
        self.queries.append((question, top_k))
        return RetrievalResult(chunks=list(self.chunks), matches_received=len(self.chunks))


class FakeGemini:
    def __init__(self, answer: str = "Arsenal finished with 89 points [S1].") -> None:
        self.answer = answer
        self.calls: list[dict[str, str]] = []

    async def generate_answer(self, *, system_instruction: str, user_prompt: str) -> str:
        self.calls.append(
            {"system_instruction": system_instruction, "user_prompt": user_prompt}
        )
        return self.answer


def make_service(
    tmp_path: Path,
    *,
    chunks: list[RetrievedChunk] | None = None,
    answer: str = "Arsenal finished with 89 points [S1].",
    store: SQLiteConversationStore | None = None,
    **settings_overrides,
) -> tuple[ChatService, FakeRetrieval, FakeGemini, SQLiteConversationStore | None]:
    settings = make_settings(tmp_path, **settings_overrides)
    if store is None and "conversation_storage" not in settings_overrides:
        store = SQLiteConversationStore(settings.database_path)
    retrieval = FakeRetrieval(chunks)
    gemini = FakeGemini(answer)
    service = ChatService(settings, retrieval, gemini, conversation_store=store)
    return service, retrieval, gemini, store


@pytest.mark.asyncio
async def test_grounded_answer_returns_cited_sources(tmp_path) -> None:
    service, _retrieval, gemini, store = make_service(tmp_path)

    result = await service.chat("How many points did Arsenal get?")

    assert result.answer == "Arsenal finished with 89 points [S1]."
    assert result.evidence_found is True
    assert result.cited_source_ids == ["S1"]
    assert [source.citation_id for source in result.sources] == ["S1"]
    assert result.warnings == []
    assert len(gemini.calls) == 1
    assert "<evidence>" in gemini.calls[0]["user_prompt"]
    assert store is not None
    assert len(store.get_history(result.conversation_id)) == 2


@pytest.mark.asyncio
async def test_no_evidence_does_not_call_gemini(tmp_path) -> None:
    service, _, gemini, store = make_service(tmp_path, chunks=[])

    result = await service.chat("Who won the 2050 season?")

    assert result.answer == NO_EVIDENCE_ANSWER
    assert result.evidence_found is False
    assert result.sources == []
    assert gemini.calls == []
    assert store is not None
    history = store.get_history(result.conversation_id)
    assert history[0].content == "Who won the 2050 season?"
    assert history[1].content == NO_EVIDENCE_ANSWER


@pytest.mark.asyncio
async def test_unknown_citation_produces_warning(tmp_path) -> None:
    service, _, _, _ = make_service(tmp_path, answer="Arsenal won [S9].")

    result = await service.chat("Who won?")

    assert result.cited_source_ids == []
    assert any("not provided" in warning and "S9" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_answer_without_citation_produces_warning(tmp_path) -> None:
    service, _, _, _ = make_service(tmp_path, answer="Arsenal won the league.")

    result = await service.chat("Who won?")

    assert result.cited_source_ids == []
    assert any("did not cite" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_follow_up_uses_previous_question_for_retrieval(tmp_path) -> None:
    service, retrieval, gemini, store = make_service(tmp_path)
    assert store is not None
    store.append_exchange(
        "conversation-one",
        "Which team won the league in 2023-24?",
        "Arsenal did [S1].",
    )

    result = await service.chat(
        "Who was their top scorer?", conversation_id="conversation-one"
    )

    retrieval_query = retrieval.queries[0][0]
    assert "Which team won the league in 2023-24?" in retrieval_query
    assert "Who was their top scorer?" in retrieval_query
    assert "Which team won the league in 2023-24?" in gemini.calls[0]["user_prompt"]
    assert result.conversation_id == "conversation-one"


@pytest.mark.asyncio
async def test_unrelated_short_question_does_not_reuse_previous_topic(tmp_path) -> None:
    service, retrieval, _, store = make_service(tmp_path)
    assert store is not None
    store.append_exchange("conversation-one", "Which team won the league?", "Arsenal [S1].")

    await service.chat(
        "How many goals did Salah score?", conversation_id="conversation-one"
    )

    assert retrieval.queries[0][0] == "How many goals did Salah score?"


@pytest.mark.asyncio
async def test_conversations_are_isolated(tmp_path) -> None:
    service, retrieval, _, store = make_service(tmp_path)
    assert store is not None
    store.append_exchange("conversation-a", "Arsenal secret question", "answer")

    await service.chat("Who was their top scorer?", conversation_id="conversation-b")

    assert retrieval.queries[0][0] == "Who was their top scorer?"


@pytest.mark.asyncio
async def test_generated_conversation_id_is_safe(tmp_path) -> None:
    service, _, _, _ = make_service(tmp_path)

    result = await service.chat("Who won?")

    assert re.fullmatch(r"[a-f0-9]{32}", result.conversation_id)


@pytest.mark.asyncio
async def test_top_k_is_bounded_by_configuration(tmp_path) -> None:
    service, retrieval, _, _ = make_service(tmp_path, max_top_k=3, top_k=2)

    with pytest.raises(RequestValidationAppError):
        await service.chat("question", top_k=4)

    assert retrieval.queries == []


@pytest.mark.asyncio
async def test_context_chunk_limit_limits_returned_sources(tmp_path) -> None:
    service, _, _, _ = make_service(
        tmp_path,
        chunks=[chunk("S1"), chunk("S2"), chunk("S3")],
        max_context_chunks=2,
        answer="Answer [S1].",
    )

    result = await service.chat("question")

    assert [source.citation_id for source in result.sources] == ["S1", "S2"]
    assert any("Only the top 2" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_history_load_failure_does_not_block_answer(tmp_path) -> None:
    class BrokenStore:
        def get_history(self, *args, **kwargs):
            raise RuntimeError("disk gone")

        def append_exchange(self, *args, **kwargs):
            raise RuntimeError("disk gone")

    service, _, gemini, _ = make_service(tmp_path, store=BrokenStore())  # type: ignore[arg-type]

    result = await service.chat("Who won?")

    assert result.answer
    assert len(gemini.calls) == 1
    assert any("history could not be loaded" in warning.lower() for warning in result.warnings)
    assert any("could not be saved" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_memory_can_be_disabled(tmp_path) -> None:
    service, _, _, _ = make_service(tmp_path, conversation_storage="none")

    result = await service.chat("Who won?")

    assert result.answer
    assert result.conversation_id


@pytest.mark.asyncio
async def test_warnings_are_deduplicated(tmp_path) -> None:
    service, _, _, _ = make_service(
        tmp_path,
        chunks=[chunk()],
        answer="Answer without citations.",
    )

    result = await service.chat("question")

    assert len(result.warnings) == len(set(result.warnings))
