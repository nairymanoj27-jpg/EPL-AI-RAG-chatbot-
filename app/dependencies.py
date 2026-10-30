"""Shared FastAPI dependencies and API-key authentication."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from app.config import Settings
from app.services.chat_service import ChatService
from app.services.conversation_store import SQLiteConversationStore
from app.services.readiness import ReadinessService
from app.services.retrieval import RetrievalService


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_retrieval_service(request: Request) -> RetrievalService:
    return request.app.state.retrieval_service


def get_conversation_store(request: Request) -> SQLiteConversationStore | None:
    return request.app.state.conversation_store


def get_readiness_service(request: Request) -> ReadinessService:
    return request.app.state.readiness_service


_api_key_header = APIKeyHeader(
    name="X-API-Key",
    scheme_name="AppApiKey",
    description="Optional application key. Required only when APP_API_KEY is configured.",
    auto_error=False,
)


async def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Security(_api_key_header)] = None,
) -> None:
    """Require ``X-API-Key`` when ``APP_API_KEY`` is configured.

    Comparison is constant-time.  A missing key yields 401; the response never
    echoes the expected value.
    """
    expected = request.app.state.settings.api_key
    if expected is None:
        return
    supplied = x_api_key or ""
    if not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


ApiKeyDependency = Annotated[None, Depends(require_api_key)]
