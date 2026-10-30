"""Shared test fixtures.

No test performs network access: Pinecone and Gemini clients are constructed
with fake credentials but never called, and API tests use dependency overrides
or fake service objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "_env_file": None,
        "pinecone_api_key": "test-pinecone-key",
        "pinecone_index_name": "test-index",
        "gemini_api_key": "test-gemini-key",
        "app_env": "test",
        "conversation_db_path": str(tmp_path / "conversations.sqlite3"),
        "readiness_cache_seconds": 0,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def app(settings: Settings):
    return create_app(settings)


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client
