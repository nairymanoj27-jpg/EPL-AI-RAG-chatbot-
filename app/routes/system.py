"""Operational endpoints: application info, liveness, and readiness."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response, status

from app.config import Settings
from app.dependencies import get_app_settings, get_readiness_service
from app.schemas import AppInfoResponse, HealthResponse, ReadyResponse
from app.services.readiness import ReadinessService

router = APIRouter(tags=["system"])


@router.get("/", response_model=AppInfoResponse, summary="Application metadata")
async def root(
    request: Request,
    settings: Settings = Depends(get_app_settings),
) -> AppInfoResponse:
    """Human-friendly entry point that points at the UI and API docs."""
    ui_url = "/ui/" if getattr(request.app.state, "frontend_mounted", False) else None
    return AppInfoResponse(
        name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        status="ok",
        docs_url="/docs",
        ui_url=ui_url,
    )


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health(settings: Settings = Depends(get_app_settings)) -> HealthResponse:
    """Process liveness only.

    This deliberately performs no network calls so orchestrators can restart a
    wedged process without depending on Pinecone or Gemini availability.
    """
    return HealthResponse(
        name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness probe (Pinecone + Gemini)",
    responses={503: {"model": ReadyResponse, "description": "A required dependency is not ready"}},
)
async def ready(
    response: Response,
    settings: Settings = Depends(get_app_settings),
    readiness: ReadinessService = Depends(get_readiness_service),
) -> ReadyResponse:
    """Check that dependencies are configured and reachable.

    Results are cached for ``READINESS_CACHE_SECONDS`` to avoid hammering
    upstream services.  Returns HTTP 503 when a required check fails.
    """
    is_ready, checks = await readiness.check()
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(ready=is_ready, version=settings.app_version, checks=checks)


__all__ = ["router"]
