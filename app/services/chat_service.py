"""End-to-end chat orchestration.

Retrieval is always performed for the current question.  Conversation history
is used only to make follow-up questions retrievable and to give the model
context; previous assistant messages are never treated as evidence.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field

from app.config import Settings
from app.errors import RequestValidationAppError
from app.services.conversation_store import ConversationTurn, SQLiteConversationStore
from app.services.gemini_service import GeminiService
from app.services.prompt_builder import (
    NO_EVIDENCE_ANSWER,
    build_prompt,
    extract_citation_ids,
    find_unknown_citations,
)
from app.services.retrieval import RetrievalService, RetrievedChunk

logger = logging.getLogger(__name__)

_FOLLOW_UP_WORDS = {
    "he",
    "her",
    "hers",
    "him",
    "his",
    "it",
    "its",
    "they",
    "them",
    "their",
    "this",
    "that",
    "these",
    "those",
    "there",
    "then",
    "same",
    "more",
    "also",
}


@dataclass
class ChatResult:
    answer: str
    conversation_id: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    cited_source_ids: list[str] = field(default_factory=list)
    evidence_found: bool = False
    warnings: list[str] = field(default_factory=list)


class ChatService:
    def __init__(
        self,
        settings: Settings,
        retrieval_service: RetrievalService,
        gemini_service: GeminiService,
        conversation_store: SQLiteConversationStore | None = None,
    ) -> None:
        self._settings = settings
        self._retrieval = retrieval_service
        self._gemini = gemini_service
        self._store = conversation_store

    async def chat(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        top_k: int | None = None,
    ) -> ChatResult:
        conversation_id = conversation_id or uuid.uuid4().hex
        top_k = self._resolve_top_k(top_k)
        warnings: list[str] = []

        history = await self._load_history(conversation_id, warnings)
        retrieval_query = self._contextualize_question(question, history)

        retrieval = await self._retrieval.retrieve(retrieval_query, top_k=top_k)
        warnings.extend(retrieval.warnings)

        if not retrieval.chunks:
            return await self._finish(
                conversation_id=conversation_id,
                question=question,
                answer=NO_EVIDENCE_ANSWER,
                sources=[],
                cited=[],
                evidence_found=False,
                warnings=warnings,
            )

        prompt = build_prompt(
            question,
            retrieval.chunks,
            history=history,
            max_context_characters=self._settings.max_context_characters,
            max_context_chunks=self._settings.max_context_chunks,
            max_history_turns=self._settings.conversation_max_turns,
            max_history_characters=self._settings.conversation_max_characters,
        )
        warnings.extend(prompt.warnings)

        if not prompt.included_chunks:
            warnings.append(
                "The context budget was too small to include any retrieved chunk. "
                "Increase MAX_CONTEXT_CHARACTERS or reduce metadata length."
            )
            return await self._finish(
                conversation_id=conversation_id,
                question=question,
                answer=NO_EVIDENCE_ANSWER,
                sources=[],
                cited=[],
                evidence_found=False,
                warnings=warnings,
            )

        answer = await self._gemini.generate_answer(
            system_instruction=prompt.system_instruction,
            user_prompt=prompt.user_prompt,
        )

        allowed_ids = [chunk.citation_id for chunk in prompt.included_chunks]
        cited = extract_citation_ids(answer, allowed_ids)
        unknown = find_unknown_citations(answer, allowed_ids)
        if unknown:
            warnings.append(
                "Gemini cited source identifiers that were not provided: "
                + ", ".join(unknown)
                + ". Treat those citations as unsupported."
            )
        if not cited and answer.strip() != NO_EVIDENCE_ANSWER:
            warnings.append(
                "Gemini did not cite any retrieved source. The answer may be unsupported; "
                "verify it against the returned source passages."
            )

        return await self._finish(
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            sources=list(prompt.included_chunks),
            cited=cited,
            evidence_found=True,
            warnings=warnings,
        )

    async def _finish(
        self,
        *,
        conversation_id: str,
        question: str,
        answer: str,
        sources: list[RetrievedChunk],
        cited: list[str],
        evidence_found: bool,
        warnings: list[str],
    ) -> ChatResult:
        await self._save_exchange(conversation_id, question, answer, warnings)
        return ChatResult(
            answer=answer,
            conversation_id=conversation_id,
            sources=sources,
            cited_source_ids=cited,
            evidence_found=evidence_found,
            warnings=list(dict.fromkeys(warnings)),
        )

    def _resolve_top_k(self, requested: int | None) -> int:
        if requested is None:
            return min(self._settings.top_k, self._settings.max_top_k)
        if not 1 <= requested <= self._settings.max_top_k:
            raise RequestValidationAppError(
                f"top_k must be between 1 and {self._settings.max_top_k}."
            )
        return requested

    async def _load_history(
        self, conversation_id: str, warnings: list[str]
    ) -> list[ConversationTurn]:
        if self._store is None:
            return []
        try:
            return await asyncio.to_thread(
                self._store.get_history,
                conversation_id,
                max_turns=self._settings.conversation_max_turns,
                max_characters=self._settings.conversation_max_characters,
            )
        except Exception as exc:
            logger.warning("Conversation history load failed (%s)", type(exc).__name__)
            warnings.append(
                "Conversation history could not be loaded; this answer used no prior turns."
            )
            return []

    async def _save_exchange(
        self,
        conversation_id: str,
        question: str,
        answer: str,
        warnings: list[str],
    ) -> None:
        if self._store is None:
            return
        try:
            await asyncio.to_thread(
                self._store.append_exchange, conversation_id, question, answer
            )
        except Exception as exc:
            logger.warning("Conversation history save failed (%s)", type(exc).__name__)
            warnings.append(
                "The answer could not be saved to conversation history; the next turn may "
                "not remember this exchange."
            )

    def _contextualize_question(
        self, question: str, history: list[ConversationTurn]
    ) -> str:
        """Add a little prior context so short follow-ups retrieve useful chunks.

        This is deliberately a cheap heuristic rather than an extra LLM call.
        The original question is still the one shown to the answer model.
        """
        previous_user = next(
            (turn for turn in reversed(history) if turn.role == "user"), None
        )
        if previous_user is None:
            return question

        words = re.findall(r"[A-Za-z0-9']+", question.lower())
        starts_like_follow_up = question.lower().lstrip().startswith(
            ("and ", "what about", "how about", "plus ")
        )
        looks_like_follow_up = (
            any(word in _FOLLOW_UP_WORDS for word in words)
            or len(words) <= 4
            or starts_like_follow_up
        )
        if not looks_like_follow_up:
            return question

        context = previous_user.content.strip()[-500:]
        return f"{context} {question}"[:800]


__all__ = ["ChatResult", "ChatService"]
