"""Google Gemini answer generation via the current ``google-genai`` SDK."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.config import Settings
from app.errors import (
    DependencyAuthError,
    DependencyRateLimitError,
    DependencyTimeoutError,
    DependencyUnavailableError,
    EmptyGenerationError,
    GenerationError,
)

logger = logging.getLogger(__name__)


class GeminiService:
    """Thin, testable wrapper around the Gemini asynchronous client."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client or genai.Client(
            api_key=settings.gemini_api_key.get_secret_value(),
            http_options=types.HttpOptions(
                timeout=int(settings.request_timeout_seconds * 1000),
                retry_options=types.HttpRetryOptions(
                    attempts=2,
                    initial_delay=0.5,
                    max_delay=2.0,
                    http_status_codes=[429, 500, 502, 503, 504],
                ),
            ),
        )

    @property
    def model(self) -> str:
        return self._settings.gemini_model

    async def generate_answer(self, *, system_instruction: str, user_prompt: str) -> str:
        """Generate a grounded answer.

        Raises:
            GenerationError / EmptyGenerationError / Dependency*: on upstream
                failures.  The caller must not fabricate an answer when this
                raises.
        """
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self._settings.gemini_temperature,
            max_output_tokens=self._settings.gemini_max_output_tokens,
        )
        try:
            async with asyncio.timeout(self._settings.request_timeout_seconds):
                response = await self._client.aio.models.generate_content(
                    model=self._settings.gemini_model,
                    contents=user_prompt,
                    config=config,
                )
        except TimeoutError as exc:
            raise DependencyTimeoutError("Gemini timed out while generating an answer.") from exc
        except genai_errors.APIError as exc:
            raise self._map_api_error(exc, action="Gemini generation") from exc
        except Exception as exc:
            logger.warning("Gemini generation raised %s", type(exc).__name__)
            raise GenerationError(
                "Gemini answer generation failed. Check the server logs for the error type."
            ) from exc

        try:
            text = response.text
        except Exception as exc:
            logger.warning("Gemini response text was unreadable (%s)", type(exc).__name__)
            raise EmptyGenerationError() from exc
        if not isinstance(text, str) or not text.strip():
            raise EmptyGenerationError()
        return text.strip()

    async def probe(self) -> None:
        """Lightweight, non-billable readiness probe (model metadata lookup)."""
        try:
            async with asyncio.timeout(self._settings.request_timeout_seconds):
                await self._client.aio.models.get(model=self._settings.gemini_model)
        except TimeoutError as exc:
            raise DependencyTimeoutError("Gemini readiness check timed out.") from exc
        except genai_errors.APIError as exc:
            raise self._map_api_error(exc, action="Gemini readiness check") from exc
        except Exception as exc:
            logger.warning("Gemini readiness probe raised %s", type(exc).__name__)
            raise DependencyUnavailableError(
                "Gemini readiness check failed. Check GEMINI_API_KEY and network access."
            ) from exc

    @staticmethod
    def _map_api_error(exc: genai_errors.APIError, *, action: str):
        code = getattr(exc, "code", None)
        logger.warning("%s failed with Gemini API code %s", action, code)
        if code in (401, 403):
            return DependencyAuthError(
                "Gemini rejected the API key. Check GEMINI_API_KEY and its permissions."
            )
        if code == 429:
            return DependencyRateLimitError(
                "Gemini rate limit reached. Wait a moment and retry.", retry_after=10
            )
        if code in (408, 504):
            return DependencyTimeoutError("Gemini timed out.")
        if code in (400, 404):
            return GenerationError(
                "Gemini rejected the request. Check GEMINI_MODEL and generation settings."
            )
        if isinstance(code, int) and code >= 500:
            return DependencyUnavailableError(
                "Gemini is temporarily unavailable. Retry shortly."
            )
        return GenerationError(
            "Gemini answer generation failed. Check the server logs for the error type."
        )

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            try:
                await asyncio.to_thread(close)
            except Exception:
                logger.debug("Error while closing Gemini client", exc_info=True)
