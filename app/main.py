"""FastAPI application factory and lifespan wiring."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import PROJECT_ROOT, Settings, get_settings
from app.errors import AppError
from app.middleware import BodySizeLimitMiddleware, RequestIDMiddleware
from app.routes import chat as chat_routes
from app.routes import system as system_routes
from app.services.chat_service import ChatService
from app.services.conversation_store import SQLiteConversationStore
from app.services.embeddings import EmbeddingService
from app.services.gemini_service import GeminiService
from app.services.pinecone_service import PineconeService
from app.services.readiness import ReadinessService
from app.services.retrieval import RetrievalService
from app.utils.logging_config import configure_logging, request_id_var

logger = logging.getLogger(__name__)

API_DESCRIPTION = """
A retrieval-augmented chatbot over an existing English Premier League vector
index in Pinecone.

* Answers are grounded **only** in retrieved passages and cite them as `[S1]`,
  `[S2]`, ...
* The service never writes to Pinecone.
* `POST /chat` requires a `GEMINI_API_KEY` and `PINECONE_API_KEY` configured on
  the server; secrets are never accepted from the browser.
"""


def _current_request_id() -> str | None:
    value = request_id_var.get()
    return None if value in ("", "-") else value


def _error_payload(code: str, message: str) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": _current_request_id(),
        }
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    ``settings`` is injectable so tests can construct an app without a real
    ``.env``.  Clients are created eagerly but connect lazily, so importing or
    starting the app performs no paid or network calls.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    pinecone_service = PineconeService(settings)
    embedding_service = EmbeddingService(settings, pinecone_service.inference)
    retrieval_service = RetrievalService(settings, embedding_service, pinecone_service)
    gemini_service = GeminiService(settings)
    conversation_store = (
        SQLiteConversationStore(settings.database_path)
        if settings.conversation_storage == "sqlite"
        else None
    )
    chat_service = ChatService(
        settings,
        retrieval_service,
        gemini_service,
        conversation_store=conversation_store,
    )
    readiness_service = ReadinessService(settings, pinecone_service, gemini_service)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        for warning in settings.startup_warnings():
            logger.warning("Configuration: %s", warning)
        logger.info(
            "Starting %s v%s (env=%s, index=%s, namespace=%r)",
            settings.app_name,
            settings.app_version,
            settings.app_env,
            settings.pinecone_index_name,
            settings.namespace,
        )
        try:
            yield
        finally:
            await gemini_service.aclose()
            await asyncio.to_thread(pinecone_service.close)
            if conversation_store is not None:
                conversation_store.close()
            logger.info("Shutdown complete")

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=API_DESCRIPTION,
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    app.state.settings = settings
    app.state.pinecone_service = pinecone_service
    app.state.embedding_service = embedding_service
    app.state.retrieval_service = retrieval_service
    app.state.gemini_service = gemini_service
    app.state.chat_service = chat_service
    app.state.conversation_store = conversation_store
    app.state.readiness_service = readiness_service
    app.state.frontend_mounted = False

    # Middleware is applied bottom-up: the last added runs first.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key"],
        expose_headers=["X-Request-ID"],
        max_age=600,
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)
    app.add_middleware(RequestIDMiddleware)

    app.include_router(system_routes.router)
    app.include_router(chat_routes.router)

    @app.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after is not None else None
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(exc.code, exc.message),
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = []
        for error in exc.errors():
            location = ".".join(
                str(part) for part in error.get("loc", ()) if part not in ("body", "query")
            ) or "request"
            details.append(
                {
                    "location": location,
                    "message": str(error.get("msg", "Invalid value")),
                    "type": str(error.get("type", "value_error")),
                }
            )
        payload = _error_payload("invalid_request", "The request is invalid.")
        payload["details"] = details
        return JSONResponse(status_code=422, content=payload)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        _request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        message = exc.detail if isinstance(exc.detail, str) else "The request failed."
        code = "not_found" if exc.status_code == 404 else "http_error"
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(code, message),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled application error (%s)", type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content=_error_payload(
                "internal_error", "An unexpected error occurred. Reference the request ID."
            ),
        )

    frontend_dir = PROJECT_ROOT / "frontend"
    if settings.serve_frontend and (frontend_dir / "index.html").is_file():
        app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="ui")
        app.state.frontend_mounted = True
    elif settings.serve_frontend:
        logger.warning("Frontend directory not found at %s; /ui is disabled", frontend_dir)

    return app


__all__ = ["create_app"]
