"""Gemini wrapper tests with a fake google-genai client."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from app.errors import (
    DependencyAuthError,
    DependencyRateLimitError,
    DependencyTimeoutError,
    DependencyUnavailableError,
    EmptyGenerationError,
    GenerationError,
)
from app.services.gemini_service import GeminiService
from tests.conftest import make_settings


class FakeModels:
    def __init__(self, *, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.generate_calls: list[dict] = []
        self.get_calls: list[dict] = []

    async def generate_content(self, **kwargs):
        self.generate_calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response

    async def get(self, **kwargs):
        self.get_calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(name=kwargs.get("model"))


class FakeClient:
    def __init__(self, models: FakeModels) -> None:
        self.aio = SimpleNamespace(models=models)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def make_service(tmp_path, models: FakeModels) -> tuple[GeminiService, FakeClient]:
    client = FakeClient(models)
    service = GeminiService(make_settings(tmp_path), client=client)
    return service, client


@pytest.mark.asyncio
async def test_generate_answer_passes_grounded_configuration(tmp_path) -> None:
    models = FakeModels(response=SimpleNamespace(text="Grounded answer [S1]."))
    service, _ = make_service(tmp_path, models)

    answer = await service.generate_answer(
        system_instruction="SYSTEM", user_prompt="PROMPT"
    )

    assert answer == "Grounded answer [S1]."
    call = models.generate_calls[0]
    assert call["model"] == "gemini-flash-latest"
    assert call["contents"] == "PROMPT"
    assert call["config"].system_instruction == "SYSTEM"
    assert call["config"].temperature == 0.2
    assert call["config"].max_output_tokens == 1024


@pytest.mark.asyncio
async def test_empty_generation_is_an_error(tmp_path) -> None:
    models = FakeModels(response=SimpleNamespace(text="   "))
    service, _ = make_service(tmp_path, models)

    with pytest.raises(EmptyGenerationError):
        await service.generate_answer(system_instruction="S", user_prompt="P")


@pytest.mark.asyncio
async def test_missing_text_is_an_error(tmp_path) -> None:
    models = FakeModels(response=SimpleNamespace(text=None))
    service, _ = make_service(tmp_path, models)

    with pytest.raises(EmptyGenerationError):
        await service.generate_answer(system_instruction="S", user_prompt="P")


@pytest.mark.asyncio
async def test_unreadable_response_text_is_a_safe_error(tmp_path) -> None:
    class BlockedResponse:
        @property
        def text(self):
            raise ValueError("blocked by safety filter")

    service, _ = make_service(tmp_path, FakeModels(response=BlockedResponse()))

    with pytest.raises(EmptyGenerationError) as excinfo:
        await service.generate_answer(system_instruction="S", user_prompt="P")

    assert "safety filter" not in str(excinfo.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (401, DependencyAuthError),
        (403, DependencyAuthError),
        (429, DependencyRateLimitError),
        (504, DependencyTimeoutError),
        (500, DependencyUnavailableError),
        (503, DependencyUnavailableError),
        (404, GenerationError),
        (400, GenerationError),
    ],
)
async def test_api_errors_are_mapped_to_safe_errors(tmp_path, code, expected) -> None:
    error = genai_errors.APIError(code, {"error": {"message": "upstream detail"}})
    service, _ = make_service(tmp_path, FakeModels(error=error))

    with pytest.raises(expected) as excinfo:
        await service.generate_answer(system_instruction="S", user_prompt="P")

    # Upstream messages are never handed to the client verbatim.
    assert "upstream detail" not in str(excinfo.value)
    if isinstance(excinfo.value, DependencyRateLimitError):
        assert excinfo.value.retry_after == 10


@pytest.mark.asyncio
async def test_generic_exception_becomes_generation_error(tmp_path) -> None:
    service, _ = make_service(tmp_path, FakeModels(error=RuntimeError("socket failure")))

    with pytest.raises(GenerationError) as excinfo:
        await service.generate_answer(system_instruction="S", user_prompt="P")

    assert "socket failure" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_builtin_timeout_becomes_timeout_error(tmp_path) -> None:
    service, _ = make_service(tmp_path, FakeModels(error=TimeoutError()))

    with pytest.raises(DependencyTimeoutError):
        await service.generate_answer(system_instruction="S", user_prompt="P")


@pytest.mark.asyncio
async def test_probe_uses_non_billable_model_lookup(tmp_path) -> None:
    models = FakeModels()
    service, _ = make_service(tmp_path, models)

    await service.probe()

    assert models.get_calls == [{"model": "gemini-flash-latest"}]
    assert models.generate_calls == []


@pytest.mark.asyncio
async def test_probe_maps_authentication_error(tmp_path) -> None:
    error = genai_errors.APIError(403, {"error": {"message": "forbidden"}})
    service, _ = make_service(tmp_path, FakeModels(error=error))

    with pytest.raises(DependencyAuthError):
        await service.probe()


@pytest.mark.asyncio
async def test_client_is_closed(tmp_path) -> None:
    service, client = make_service(tmp_path, FakeModels())

    await service.aclose()

    assert client.closed is True
