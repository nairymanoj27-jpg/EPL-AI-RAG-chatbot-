"""Chat, retrieval, and conversation-history endpoints."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Path

from app.config import Settings
from app.dependencies import (
    get_app_settings,
    get_chat_service,
    get_conversation_store,
    get_retrieval_service,
    require_api_key,
)
from app.errors import (
    ConversationMemoryDisabledError,
    FeatureDisabledError,
    RequestValidationAppError,
)
from app.schemas import (
    CONVERSATION_ID_PATTERN,
    ChatRequest,
    ChatResponse,
    ClearConversationResponse,
    RetrieveRequest,
    RetrieveResponse,
    SourceChunk,
)
from app.services.chat_service import ChatService
from app.services.conversation_store import SQLiteConversationStore
from app.services.retrieval import RetrievalService, RetrievedChunk

router = APIRouter(tags=["chat"])


def _to_source(chunk: RetrievedChunk) -> SourceChunk:
    return SourceChunk(
        id=chunk.id,
        citation_id=chunk.citation_id,
        rank=chunk.rank,
        score=chunk.score,
        text=chunk.text,
        metadata=chunk.metadata,
    )


def _check_question_length(question: str, settings: Settings) -> None:
    if len(question) > settings.max_question_characters:
        raise RequestValidationAppError(
            f"question must be at most {settings.max_question_characters} characters."
        )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a grounded question",
    dependencies=[Depends(require_api_key)],
)
async def chat(
    payload: ChatRequest,
    settings: Settings = Depends(get_app_settings),
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Retrieve evidence from Pinecone and answer with Gemini.

    Send the returned ``conversation_id`` on later requests to continue the
    same conversation.  Omit it to start a new one.
    """
    _check_question_length(payload.question, settings)
    result = await chat_service.chat(
        payload.question,
        conversation_id=payload.conversation_id,
        top_k=payload.top_k,
    )
    return ChatResponse(
        answer=result.answer,
        conversation_id=result.conversation_id,
        sources=[_to_source(chunk) for chunk in result.sources],
        cited_source_ids=result.cited_source_ids,
        evidence_found=result.evidence_found,
        model=settings.gemini_model,
        warnings=result.warnings,
    )


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Inspect retrieval results without calling Gemini",
    dependencies=[Depends(require_api_key)],
)
async def retrieve(
    payload: RetrieveRequest,
    settings: Settings = Depends(get_app_settings),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> RetrieveResponse:
    """Return the chunks that would be used as evidence.

    This endpoint is useful for debugging and threshold calibration.  It can
    expose indexed document text, so it can be disabled with
    ``ENABLE_RETRIEVE_ENDPOINT=false``.
    """
    if not settings.enable_retrieve_endpoint:
        raise FeatureDisabledError()
    _check_question_length(payload.question, settings)
    result = await retrieval_service.retrieve(payload.question, top_k=payload.top_k)
    return RetrieveResponse(
        question=payload.question,
        namespace=settings.namespace,
        count=len(result.chunks),
        matches=[_to_source(chunk) for chunk in result.chunks],
        warnings=list(dict.fromkeys(result.warnings)),
    )


@router.delete(
    "/conversations/{conversation_id}",
    response_model=ClearConversationResponse,
    summary="Clear a conversation's stored history",
    dependencies=[Depends(require_api_key)],
)
async def clear_conversation(
    conversation_id: Annotated[str, Path(pattern=CONVERSATION_ID_PATTERN)],
    store: SQLiteConversationStore | None = Depends(get_conversation_store),
) -> ClearConversationResponse:
    """Delete all stored turns for one conversation ID.

    Clearing an unknown conversation is not an error; ``cleared`` is simply
    ``false``.  Other conversations are never affected.
    """
    if store is None:
        raise ConversationMemoryDisabledError()
    cleared = await asyncio.to_thread(store.clear, conversation_id)
    return ClearConversationResponse(conversation_id=conversation_id, cleared=cleared)


__all__ = ["router"]
